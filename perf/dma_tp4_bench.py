"""Does the static DMA transpose move more bytes per second with 4-byte elements? The attention's P tiles
[128 q, 512 k] bf16 transposed as bf16 (today) vs viewed as fp32 [128 q, 256 k-pairs] (each element a pair of
neighbouring keys), same bytes. SBUF->SBUF, one logical core; NT tiles per call, timed minus a no-transpose call.
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/dma_tp4_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

NT = int(os.environ.get("NT", "64"))
REPS = int(os.environ.get("REPS", "10"))
DEV = torch.device("neuron:0")


def make(mode, reps_in):
    @nki.jit
    def k(p_hbm):
        out = nl.ndarray((128, NT, 512), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        p_sb = nl.ndarray((128, NT, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=p_sb, src=p_hbm[:, 0:NT, :])
        pt = nl.ndarray((128, NT, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        row = NT * 512
        for _ in range(reps_in):
            for t in range(NT):
                if mode == "bf16":
                    nisa.dma_transpose(dst=pt.ap([[row, 128], [1, 1], [128, 4], [1, 128]], offset=t * 512),
                                       src=p_sb.ap([[row, 128], [1, 1], [128, 4], [1, 128]], offset=t * 512))
                elif mode == "fp32":
                    # 4-byte elements: P[q, 2j:2j+2] as one element; [128 q, 256 j] -> [128 j, 2 halves, 128 q]
                    p32 = p_sb.view(nl.float32)          # [128, NT, 256]
                    t32 = pt.view(nl.float32)
                    if t == 0:
                        print("[dtp4] view shape", tuple(p32.shape), flush=True)
                    row32 = 1
                    for dim_ in p32.shape[1:]:
                        row32 *= int(dim_)
                    pstep = int(os.environ.get("PSTEP_MUL", "2")) * row32
                    nisa.dma_transpose(dst=t32.ap([[pstep, 128], [1, 1], [128, 2], [1, 128]], offset=t * 256),
                                       src=p32.ap([[pstep, 128], [1, 1], [128, 2], [1, 128]], offset=t * 256))
        nisa.dma_copy(dst=out, src=pt)
        return out
    return k


def make32(reps_in):
    @nki.jit
    def k(p_hbm):        # p_hbm: [128, NT + i, 256] fp32 (bf16 pairs bit-cast on the host)
        out = nl.ndarray((128, NT, 256), dtype=nl.float32, buffer=nl.shared_hbm)
        p_sb = nl.ndarray((128, NT, 256), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=p_sb, src=p_hbm[:, 0:NT, :])
        pt = nl.ndarray((128, NT, 256), dtype=nl.float32, buffer=nl.sbuf)
        row = NT * 256
        for _ in range(reps_in):
            for t in range(NT):
                nisa.dma_transpose(dst=pt.ap([[row, 128], [1, 1], [128, 2], [1, 128]], offset=t * 256),
                                   src=p_sb.ap([[row, 128], [1, 1], [128, 2], [1, 128]], offset=t * 256))
        nisa.dma_copy(dst=out, src=pt)
        return out
    return k


def bench(fn, *a):
    for _ in range(2):
        y = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


g = torch.Generator().manual_seed(0)
res = {}
for i, (mode, reps_in) in enumerate((("bf16", 1), ("bf16", 5), ("fp32", 1), ("fp32", 5))):
    full = torch.randn(128, NT + i, 512, generator=g).to(torch.bfloat16)
    p = full[:, :NT].contiguous()
    try:
        if mode == "fp32":
            us, y = bench(make32(reps_in), full.view(torch.float32).to(DEV))
            y = y.cpu().view(torch.bfloat16)
        else:
            us, y = bench(make(mode, reps_in), full.to(DEV))
        res[(mode, reps_in)] = us
        if mode == "bf16":
            ref = p.float().view(128, NT, 4, 128).permute(3, 1, 2, 0).reshape(128, NT, 512)
        else:
            # pt32[j, t, h*128 + q] = P32[q, t, h*128 + j]: as bf16, pt[j, t, (h*128 + q)*2 + e] = P[q, t, (h*128 + j)*2 + e]
            ref = p.float().view(128, NT, 2, 128, 2).permute(3, 1, 2, 0, 4).reshape(128, NT, 512)
        ok = torch.allclose(y.float().cpu(), ref)
        print(f"[dtp4] {mode} x{reps_in}: {us:8.1f} us/call  correct={ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dtp4] {mode} x{reps_in} FAILED: {str(e)[:600]}", flush=True)
mb = NT * 128 * 512 * 2 / 1e6
for mode in ("bf16", "fp32"):
    if (mode, 1) in res and (mode, 5) in res:
        per = (res[(mode, 5)] - res[(mode, 1)]) / 4
        print(f"[dtp4] {mode}: {per:7.1f} us per {mb:.1f} MB pass -> {mb / per * 1e3:6.1f} GB/s", flush=True)
print("[dtp4] done", flush=True)
