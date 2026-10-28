"""Side-by-side frames of two videos, with per-frame PSNR of b against a.

python3 kiro/frames_cmp.py a.mp4 b.mp4 out.png [frame ...]   (default: first, middle, last)
Row 1 is a, row 2 is b, row 3 is |a - b| x4. The PNG is downscaled 2x.
"""
import sys

import numpy as np


def read_all(path):
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f[:, :, ::-1])
        cap.release()
        if frames:
            return np.stack(frames)
    import imageio.v3 as iio
    return np.asarray(iio.imread(path))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    a_path, b_path, out = sys.argv[1:4]
    a, b = read_all(a_path), read_all(b_path)
    n = min(len(a), len(b))
    idx = [int(i) for i in sys.argv[4:]] or [0, n // 2, n - 1]
    print(f"frames: a={len(a)} b={len(b)} hw={a.shape[1:3]}")
    print("psnr(b vs a), every 8th frame: "
          + " ".join(f"{i}:{psnr(a[i], b[i]):.1f}" for i in range(0, n, 8)))
    rows = [np.concatenate([v[i] for i in idx], axis=1) for v in (a, b)]
    rows.append(np.concatenate(
        [np.clip(np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
         for i in idx], axis=1))
    img = np.ascontiguousarray(np.concatenate(rows, axis=0)[::2, ::2])
    from PIL import Image
    Image.fromarray(img).save(out)
    print(f"wrote {out} {img.shape}")


if __name__ == "__main__":
    main()
