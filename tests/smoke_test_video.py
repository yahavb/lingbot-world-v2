#!/usr/bin/env python3
"""Smoke test for LingBot-World v2 inference.

Runs the model's bring-up pipeline (run_fast_neuron.sh by default) and
sanity-checks the resulting mp4: right frame count / fps / resolution, not
blank, not frozen. Doesn't check semantic quality -- just "did inference
produce a video that looks like a video".

Usage:
    # full run: executes run_fast_neuron.sh, then checks the resulting video
    python3 tests/smoke_test_video.py

    # same, with explicit pipeline/gate options
    python3 tests/smoke_test_video.py --eworld 4 --lnc 2 --save-file output/x.mp4

    # check an already-generated video instead of running inference
    python3 tests/smoke_test_video.py --skip-run --save-file output/existing.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(script, eworld, lnc, save_file):
    cmd = [str(script), str(eworld), str(lnc), str(save_file)]
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)

    # torch-neuronx can segfault during teardown after the real work is
    # already done -- check the output file, not the return code, same as
    # run_fast_neuron.sh does for $LATENTS_PATH.
    if not Path(save_file).exists():
        print(f"pipeline did not produce {save_file} (exit code {proc.returncode})", file=sys.stderr)
        sys.exit(1)
    return Path(save_file)


def read_all_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read sequentially, never seek: CAP_PROP_POS_FRAMES on a GOP-encoded
    # H.264 mp4 snaps to the nearest keyframe and silently duplicates
    # frames, which would corrupt both the frame count and the entropy/diff
    # metrics below.
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    return {"playback_fps": fps, "width": width, "height": height, "frame_count": len(frames)}, frames


def shannon_entropy(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum()
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def check_video(
    path,
    expected_frame_num=None,
    expected_fps=None,
    expected_area=None,
    area_tolerance=0.2,
    min_entropy=3.0,
    min_frame_diff=1.0,
):
    checks = {}
    meta = {}
    path = Path(path)

    checks["file_exists"] = path.exists() and path.stat().st_size > 1024
    if not checks["file_exists"]:
        return checks, meta

    meta, frames = read_all_frames(path)
    checks["opens_and_has_frames"] = len(frames) > 0
    if not checks["opens_and_has_frames"]:
        return checks, meta

    if expected_frame_num is not None:
        checks["frame_count_matches"] = meta["frame_count"] == expected_frame_num
    if expected_fps is not None:
        checks["fps_matches"] = abs(meta["playback_fps"] - expected_fps) < 0.5
    if expected_area is not None:
        actual_area = meta["width"] * meta["height"]
        meta["area"] = actual_area
        checks["resolution_in_range"] = abs(actual_area - expected_area) / expected_area < area_tolerance

    entropies = [shannon_entropy(f) for f in frames]
    meta["mean_entropy"] = round(float(np.mean(entropies)), 2)
    checks["not_blank"] = meta["mean_entropy"] > min_entropy

    diffs = [
        float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))))
        for i in range(1, len(frames))
    ]
    meta["mean_frame_diff"] = round(float(np.mean(diffs)), 2) if diffs else 0.0
    meta["max_frame_diff"] = round(float(np.max(diffs)), 2) if diffs else 0.0
    # "not frozen" means at least one consecutive pair actually changes --
    # a model that just re-emits the input image would fail this.
    checks["not_frozen"] = bool(diffs) and max(diffs) > min_frame_diff

    # REPORTED, NOT GATED. Every check above passes on noise: noise has high entropy so
    # it is not blank, and high frame differences so it is not frozen. Six consecutive
    # cp4 runs reported GATE: PASS on output that was visually wrong.
    #
    # I first added this as a not_noise CHECK on spatial smoothness, then measured it on
    # the actual files: the good cp1 output is 4.01 and the bad cp4 output is 10.20,
    # while uniform noise is ~85. A threshold that catches uniform noise passes both of
    # ours, so gating on it would be false comfort. It is metadata instead, so a 2.5x
    # jump is at least visible in the log next to entropy and frame_diff.
    #
    # This gate cannot certify quality. It checks shape and liveness. Deciding whether
    # output is right still needs the mp4 looked at, or a reference to diff against --
    # and CLAUDE.md notes no such reference or agreed tolerance exists yet.
    grads_h = [
        float(np.mean(np.abs(f[:, 1:].astype(np.int16) - f[:, :-1].astype(np.int16))))
        for f in frames
    ]
    grads_v = [
        float(np.mean(np.abs(f[1:, :].astype(np.int16) - f[:-1, :].astype(np.int16))))
        for f in frames
    ]
    meta["mean_spatial_grad_h"] = round(float(np.mean(grads_h)), 2)
    meta["mean_spatial_grad_v"] = round(float(np.mean(grads_v)), 2)

    return checks, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", default=str(REPO_ROOT / "run_fast_neuron.sh"),
                    help="Pipeline script to run (EWORLD, LNC, SAVE_FILE positional args).")
    p.add_argument("--eworld", type=int, default=4)
    p.add_argument("--lnc", type=int, default=2)
    p.add_argument("--save-file", default=str(REPO_ROOT / "output" / "smoke_test.mp4"))
    p.add_argument("--skip-run", action="store_true",
                    help="Check an existing video at --save-file instead of running inference.")
    p.add_argument("--expected-frame-num", type=int, default=21,
                    help="Must match --frame_num used inside the pipeline script.")
    p.add_argument("--expected-fps", type=float, default=16.0,
                    help="wan_shared_cfg.sample_fps (wan/configs/shared_config.py).")
    p.add_argument("--size", default="480*832",
                    help="Must match --size used inside the pipeline script (WAN reports it as an area target, "
                         "not exact WxH, since output follows the input image's aspect ratio).")
    p.add_argument("--area-tolerance", type=float, default=0.2)
    p.add_argument("--min-entropy", type=float, default=3.0)
    p.add_argument("--min-frame-diff", type=float, default=1.0)
    args = p.parse_args()

    width, height = (int(x) for x in args.size.split("*"))
    expected_area = width * height

    if args.skip_run:
        video_path = Path(args.save_file)
        if not video_path.exists():
            print(f"--skip-run given but {video_path} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        video_path = run_pipeline(args.script, args.eworld, args.lnc, args.save_file)

    checks, meta = check_video(
        video_path,
        expected_frame_num=args.expected_frame_num,
        expected_fps=args.expected_fps,
        expected_area=expected_area,
        area_tolerance=args.area_tolerance,
        min_entropy=args.min_entropy,
        min_frame_diff=args.min_frame_diff,
    )

    passed = bool(checks) and all(checks.values())

    print(f"\n{'PASS' if passed else 'FAIL'}  {video_path}")
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    print("metrics:", meta)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
