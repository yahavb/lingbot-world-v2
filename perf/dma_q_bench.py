"""HBM->SBUF DMA throughput by descriptor mode / issuing engine, one logical core: do copies issued on several
queues (static, HWDGE on Sync, HWDGE on Scalar, SWDGE on GpSimd) add up?
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/dma_q_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

NT = int(os.environ.get("NT", "64"))    # [128, 2048] bf16 tiles (512 KB each)
F = 2048
REPS = int(os.environ.get("REPS", "10"))
DEV = torch.device("neuron:0")
MODES = {
    "static": [{}],
    "hw_sync": [{"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync}],
    "hw_act": [{"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}],
    "sw": [{"dge_mode": nisa.dge_mode.swdge}],
    "sync+act": [{"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync},
                 {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}],
    "static+act+sw": [{}, {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar},
                      {"dge_mode": nisa.dge_mode.swdge}],
    "all4": [{}, {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.sync},
             {"dge_mode": nisa.dge_mode.hwdge, "engine": nisa.engine.scalar}, {"dge_mode": nisa.dge_mode.swdge}],
}


def make(mode, reps_in):
    kws = MODES[mode]

    @nki.jit
    def k(x_hbm):
        out = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        bufs = [nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
        acc = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for r in range(reps_in):
            for t in range(NT):
                nisa.dma_copy(dst=bufs[t % 8], src=x_hbm[t, 0:128, :], **kws[(r * NT + t) % len(kws)])
                if t % 8 == 7:
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=bufs[t % 8], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
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
    return (time.perf_counter() - t0) / REPS * 1e6


g = torch.Generator().manual_seed(0)
xi = 0
for mode in MODES:
    try:
        ts = []
        for reps_in in (1, 4):
            xi += 1
            x = torch.randn(NT, 128 + xi, F, generator=g).to(torch.bfloat16)      # distinct shapes: distinct kernels
            ts.append(bench(make(mode, reps_in), x.to(DEV)))
        mb = NT * 128 * F * 2 / 1e6
        per = (ts[1] - ts[0]) / 3
        print(f"[dmaq] {mode:14s} {mb / per * 1e3:7.1f} GB/s ({per:.1f} us per {mb:.0f} MB)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dmaq] {mode:14s} FAILED: {str(e).splitlines()[0][:200]}", flush=True)
print("[dmaq] done", flush=True)
