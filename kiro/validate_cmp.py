"""Comparisons and verdict for kiro/k8s_validate.sh.  python3 kiro/validate_cmp.py <out_dir>

<out_dir>/<run>/ holds latents.pt (final latent chunks), frames.pt (uint8 [F, H, W, 3]),
out.mp4 and dump/c0s0.pt (chunk 0, step 0 noise prediction); vae_fp32_<run>/frames.pt is the
fp32 CPU decode of that run's latents; timing.txt has the GEN / VAE ms of each run.
"""
import filecmp
import math
import os
import re
import sys

import torch

OUT = sys.argv[1]
FRAME_NUM, CHUNK_LAT = 81, 3


def load(run, name):
    p = os.path.join(OUT, run, name)
    return torch.load(p, map_location="cpu") if os.path.exists(p) else None


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def psnr(a, b):
    mse = (a.double() - b.double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def frame_stats(fr, ref):
    per = [psnr(fr[i], ref[i]) for i in range(ref.shape[0])]
    worst = min(range(len(per)), key=lambda i: per[i])
    lsb = int((fr.short() - ref.short()).abs().max())
    return psnr(fr, ref), per[worst], worst, lsb


def lat_chunks(lat, ref):
    fdim = next(i for i, s in enumerate(ref.shape) if s == (FRAME_NUM - 1) // 4 + 1)
    n = ref.shape[fdim] // CHUNK_LAT
    return [rel(lat.narrow(fdim, k * CHUNK_LAT, CHUNK_LAT), ref.narrow(fdim, k * CHUNK_LAT, CHUNK_LAT))
            for k in range(n)]


timing = {}
if os.path.exists(os.path.join(OUT, "timing.txt")):
    for line in open(os.path.join(OUT, "timing.txt")):
        m = re.match(r"(\S+) GEN_ms=(\S+) VAE_ms=(\S+)", line)
        if m:
            g, v = (float(x) if x != "NA" else None for x in m.groups()[1:])
            timing[m.group(1)] = (g, v)
print("=== latency (steady state) ===")
for r, (g, v) in timing.items():
    tot = (g + v) / 1000 if g is not None and v is not None else None
    fps = FRAME_NUM / tot if tot else None
    print(f"  {r:10} GEN {g} ms  VAE {v} ms  total {tot and round(tot, 3)} s  fps {fps and round(fps, 2)}")

res = {}
print("=== bit-level ===")
def same_t(a, b):
    return a is not None and b is not None and a.shape == b.shape and torch.equal(a, b)
lm, lo = load("main_code", "latents.pt"), load("off", "latents.pt")
mp_main, mp_off = (os.path.join(OUT, r, "out.mp4") for r in ("main_code", "off"))
res["main_off"] = (same_t(lm, lo), os.path.exists(mp_main) and os.path.exists(mp_off)
                   and filecmp.cmp(mp_main, mp_off, shallow=False))
print(f"  main_code vs off: latents identical {res['main_off'][0]}, mp4 bytes identical {res['main_off'][1]}")
la, lb = load("loop", "latents.pt"), load("loop_rep", "latents.pt")
fa, fb = load("loop", "frames.pt"), load("loop_rep", "frames.pt")
res["det"] = same_t(la, lb) and same_t(fa, fb)
print(f"  loop vs loop_rep: latents identical {same_t(la, lb)}, frames identical {same_t(fa, fb)}")

print("=== vs exact (ATTENTION_BACKEND=sdpa, bf16, the job's layout) ===")
le, fe, ce = load("exact", "latents.pt"), load("exact", "frames.pt"), load("exact", "dump/c0s0.pt")
for r in ("off", "loop"):
    l, f, c = load(r, "latents.pt"), load(r, "frames.pt"), load(r, "dump/c0s0.pt")
    parts = []
    if l is not None and le is not None and l.shape == le.shape:
        ch = lat_chunks(l, le)
        res[f"lat_{r}"] = rel(l, le)
        parts.append(f"latent rel-RMS {res[f'lat_{r}']:.4f} (per chunk {' '.join(f'{x:.3f}' for x in ch)})")
    if f is not None and fe is not None and f.shape == fe.shape:
        p, w, wi, lsb = frame_stats(f, fe)
        res[f"psnr_{r}"], res[f"worst_{r}"] = p, w
        parts.append(f"video PSNR {p:.2f} dB, worst frame {w:.2f} dB (#{wi}), max |diff| {lsb} LSB")
    if c is not None and ce is not None and c.shape == ce.shape:
        res[f"s0_{r}"] = rel(c, ce)
        parts.append(f"c0s0 rel-RMS {res[f's0_{r}']:.5f}")
    print(f"  {r:5}: " + (" | ".join(parts) or "missing"))

print("=== VAE vs fp32 CPU decode of the same latents ===")
for r in ("off", "loop"):
    f, ref = load(r, "frames.pt"), load(f"vae_fp32_{r}", "frames.pt")
    if f is not None and ref is not None and f.shape == ref.shape:
        p, w, wi, lsb = frame_stats(f, ref)
        res[f"vae_{r}"] = p
        print(f"  {r:5}: PSNR {p:.2f} dB, worst frame {w:.2f} dB (#{wi}), max |diff| {lsb} LSB")
    else:
        print(f"  {r:5}: missing")


print("=== c0s0 vs fp32 CPU DiT ===")
fp = os.path.join(OUT, "step0_fp32.txt")
if os.path.exists(fp):
    for line in open(fp):
        m = re.search(r"/(\w+)/dump/c0s0\.pt: c0s0 rel-RMS vs fp32 CPU = ([0-9.]+)", line)
        if m:
            res[f"fp32_{m.group(1)}"] = float(m.group(2))
    print("  " + "  ".join(f"{k[5:]} {v:.5f}" for k, v in res.items() if k.startswith("fp32_")) or "  none")


def chk(name, ok, detail):
    st = "N/A " if ok is None else ("PASS" if ok else "WARN")
    print(f"  {st}  {name:34} {detail}")


g = lambda k: res.get(k)
print("════════ VALIDATION VERDICT ════════")
tl, to, tm = (timing.get(r, (None, None)) for r in ("loop", "off", "main_code"))
tot = lambda t: (t[0] + t[1]) / 1000 if None not in t else None
chk("total time (loop vs main_code)", None if None in (tot(tl), tot(tm)) else tot(tl) < tot(tm),
    f"loop {tot(tl)} s | off {tot(to)} s | main_code {tot(tm)} s")
chk("c0s0 vs fp32 CPU DiT (rel-RMS)", None if None in (g("fp32_loop"), g("fp32_off")) else g("fp32_loop") <= g("fp32_off") * 1.10,
    f"loop {g('fp32_loop')} | off {g('fp32_off')} | exact (sdpa bf16) {g('fp32_exact')}")
chk("c0s0 vs exact (rel-RMS)", None if None in (g("s0_loop"), g("s0_off")) else g("s0_loop") <= g("s0_off") * 1.10,
    f"loop {g('s0_loop')} | off {g('s0_off')}")
chk("video vs exact trajectory", None if None in (g("psnr_loop"), g("psnr_off")) else g("psnr_loop") >= g("psnr_off") - 1.0,
    f"loop {g('psnr_loop')} dB | off {g('psnr_off')} dB (worst loop frame {g('worst_loop')})")
chk("VAE vs fp32 CPU decode", None if None in (g("vae_loop"), g("vae_off")) else g("vae_loop") >= g("vae_off") - 1.0,
    f"loop {g('vae_loop')} dB | off {g('vae_off')} dB")
chk("main == flags-off (latents, mp4)", None if lm is None or lo is None else all(res["main_off"]), str(res["main_off"]))
chk("deterministic (loop x2)", None if la is None or lb is None else res["det"], str(res["det"]))
