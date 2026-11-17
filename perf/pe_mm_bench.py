"""Tensor-engine cost of the attention PV matmul shapes (one logical core, eager, long loops):
  bf16_128    stationary V [128 k, 128 d] bf16, moving P^T [128 k, 128 q]  (the kernel today, tp_out)
  bf16_512    the same with 512 query columns (4 Q groups per instruction)
  fp8dr_128   double_row fp8: stationary [128, 2, 128], moving [128, 2, 128] (256 keys per instruction)
  fp8dr_512   double_row fp8 with 512 query columns
Each runs the same number of keys x queries; prints us per call and per 1k keys of one 128-query group.
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/pe_mm_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

KT = int(os.environ.get("KT", "64"))      # 128-key tiles per pass
REPS_IN = int(os.environ.get("REPS_IN", "16"))
REPS = int(os.environ.get("REPS", "10"))
DEV = torch.device("neuron:0")


def make(nq, fp8):
    @nki.jit
    def k(v, pt):
        # v: [KT*128, 128] (keys x d), pt: [KT*128, nq] (keys x queries) in HBM
        out = nl.ndarray((128, nq), dtype=nl.float32, buffer=nl.shared_hbm)
        dt = nl.float8_e4m3fn if fp8 else nl.bfloat16
        if fp8:
            v_sb = nl.ndarray((128, KT // 2, 2, 128), dtype=dt, buffer=nl.sbuf)
            p_sb = nl.ndarray((128, KT // 2, 2, nq), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[256, 128], [256 * 128, KT // 2], [128, 2], [1, 128]], offset=0))
            nisa.dma_copy(dst=p_sb, src=pt.ap(pattern=[[2 * nq, 128], [256 * nq, KT // 2], [nq, 2], [1, nq]],
                                              offset=0))
        else:
            v_sb = nl.ndarray((128, KT, 128), dtype=dt, buffer=nl.sbuf)
            p_sb = nl.ndarray((128, KT, nq), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[128, 128], [128 * 128, KT], [1, 128]], offset=0))
            nisa.dma_copy(dst=p_sb, src=pt.ap(pattern=[[nq, 128], [128 * nq, KT], [1, nq]], offset=0))
        ps = nl.ndarray((128, nq), dtype=nl.float32, buffer=nl.psum)
        n_inst = KT // 2 if fp8 else KT
        for r in nl.sequential_range(REPS_IN):
            for t in range(n_inst):
                if fp8:
                    nisa.nc_matmul(ps[0:128, 0:nq], v_sb[0:128, t, 0:2, 0:128], p_sb[0:128, t, 0:2, 0:nq],
                                   perf_mode=nisa.matmul_perf_mode.double_row, accumulate=(t > 0))
                else:
                    nisa.nc_matmul(ps[0:128, 0:nq], v_sb[0:128, t, 0:128], p_sb[0:128, t, 0:nq], accumulate=(t > 0))
        o_sb = nl.ndarray((128, nq), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=ps)
        nisa.dma_copy(dst=out, src=o_sb)
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
v = torch.randn(KT * 128, 128, generator=g) * 0.5
pt = torch.rand(KT * 128, 512, generator=g)
ref = v.float().t() @ pt.float()                                # [d, q]
for name, nq, fp8 in (("bf16_128", 128, False), ("bf16_512", 512, False), ("fp8dr_128", 128, True),
                      ("fp8dr_512", 512, True)):
    dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    # pt sliced to nq columns so every variant has its own input shapes (the eager NKI cache keys on them)
    vd, pd = v.to(dt).to(DEV), pt[:, :nq].contiguous().to(dt).to(DEV)
    try:
        us, y = bench(make(nq, fp8), vd, pd)
        r = ref[:, :nq]
        rel = ((y.float().cpu() - r).norm() / r.norm()).item()
        groups = nq // 128
        per = us / (REPS_IN * KT * 128 * groups / 1000.0)
        print(f"[pe] {name:10s} {us:8.1f} us/call  {per * 1000:7.1f} ns per 1k keys x 128 queries  rel {rel:.2e}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[pe] {name:10s} FAILED: {str(e).splitlines()[0][:200]}", flush=True)
print("[pe] done", flush=True)
