"""SBUF->SBUF DMA transpose throughput of the attention's P -> P^T tiles ([128 q, 512 k] bf16 -> [128 k, 4, 128 q],
the attention_cte_lb 4-D form), one logical core: does spreading the transposes over DGE modes / priorities run
them on more queues in parallel? The block profile had every transpose on one queue at ~222 GB/s, busy 73% of the
attention window.
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/dma_tp_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

NT = int(os.environ.get("NT", "256"))      # transposes per call
REPS = int(os.environ.get("REPS", "10"))
DEV = torch.device("neuron:0")


def make(mode):
    @nki.jit
    def k(p_hbm):
        out = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        nb = 8
        p_sb = nl.ndarray((128, nb, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=p_sb, src=p_hbm.ap(pattern=[[512, 128], [0, nb], [1, 512]], offset=0))
        pt = nl.ndarray((128, nb, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(NT):
            b = i % nb
            kw = {}
            if mode == "swdge" or (mode == "alt" and i % 2 == 1):
                kw["dge_mode"] = nisa.dge_mode.swdge
            elif mode == "hwdge":
                kw["dge_mode"] = nisa.dge_mode.hwdge
            if mode == "prio":
                kw["priority"] = i % 4
            if mode in ("hw16", "mix") and not (mode == "mix" and i % 2 == 0):
                # hardware-DGE form: 16 query rows per instruction, 8 per tile
                for r in range(8):
                    nisa.dma_transpose(
                        dst=pt.ap([[nb * 512, 128], [1, 1], [128, 4], [1, 16]], offset=b * 512 + r * 16),
                        src=p_sb.ap([[nb * 512, 16], [1, 1], [128, 4], [1, 128]], offset=r * 16 * nb * 512 + b * 512),
                        dge_mode=nisa.dge_mode.hwdge)
                continue
            nisa.dma_transpose(
                dst=pt.ap([[nb * 512, 128], [1, 1], [128, 4], [1, 128]], offset=b * 512),
                src=p_sb.ap([[nb * 512, 128], [1, 1], [128, 4], [1, 128]], offset=b * 512),
                **kw)
        res = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=res, src=pt[:, nb - 1, :])
        nisa.dma_copy(dst=out, src=res)
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
for i, mode in enumerate(("default", "hw16", "mix", "prio")):
    pfull = torch.randn(128 + i, 512, generator=g).to(torch.bfloat16)   # distinct shapes: distinct cache keys
    p = pfull[:128]
    pd = pfull.to(DEV)
    try:
        us, y = bench(make(mode), pd)
        ref = p.float().view(128, 4, 128).permute(2, 1, 0).reshape(128, 512)
        ok = torch.allclose(y.float().cpu(), ref)
        mb = NT * 128 * 512 * 2 / 1e6
        print(f"[dtp] {mode:8s} {us:8.1f} us/call  {mb / us * 1e3:7.1f} GB/s  correct={ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dtp] {mode:8s} FAILED: {str(e).splitlines()[0][:220]}", flush=True)
print("[dtp] done", flush=True)
