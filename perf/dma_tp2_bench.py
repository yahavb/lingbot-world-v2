"""Can hardware-DGE DMA transposes (16-partition sources) run beside the default (static-descriptor) ones on another
queue? SBUF->SBUF bf16, one logical core. Variants: default 4-D [128 q, 512 k] tiles; hwdge 2-D [16, 128] pieces;
both at once on different halves of the data.
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/dma_tp2_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

NT = int(os.environ.get("NT", "64"))       # [128, 512] tiles per call
REPS = int(os.environ.get("REPS", "10"))
DEV = torch.device("neuron:0")


def make(mode):
    @nki.jit
    def k(p_hbm):
        out = nl.ndarray((128, NT, 512), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        p_sb = nl.ndarray((128, NT, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=p_sb, src=p_hbm[:, 0:NT, :])
        pt = nl.ndarray((128, NT, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        row = NT * 512
        for t in range(NT):
            use_hw = mode == "hwdge" or (mode == "both" and t % 2 == 1)
            if not use_hw:
                nisa.dma_transpose(dst=pt.ap([[row, 128], [1, 1], [128, 4], [1, 128]], offset=t * 512),
                                   src=p_sb.ap([[row, 128], [1, 1], [128, 4], [1, 128]], offset=t * 512))
            else:
                # 2-D hardware-DGE pieces: [16 q, 128 k] -> [128 k, 16 q]
                for sub in range(4):
                    for r in range(8):
                        nisa.dma_transpose(
                            dst=pt.ap([[row, 128], [1, 16]], offset=t * 512 + sub * 128 + r * 16),
                            src=p_sb.ap([[row, 16], [1, 128]], offset=r * 16 * row + t * 512 + sub * 128),
                            dge_mode=nisa.dge_mode.hwdge)
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
for i, mode in enumerate(("default", "hwdge", "both")):
    full = torch.randn(128, NT + i, 512, generator=g).to(torch.bfloat16)   # distinct shapes: distinct cache keys
    p = full[:, :NT].contiguous()
    pd = full.to(DEV)
    try:
        us, y = bench(make(mode), pd)
        ref = p.float().view(128, NT, 4, 128).permute(3, 1, 2, 0).reshape(128, NT, 512)
        ok = torch.allclose(y.float().cpu(), ref)
        mb = NT * 128 * 512 * 2 / 1e6
        print(f"[dtp2] {mode:8s} {us:8.1f} us/call ({mb:.1f} MB, + load/store)  correct={ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dtp2] {mode:8s} FAILED: {str(e).splitlines()[0][:220]}", flush=True)
print("[dtp2] done", flush=True)
