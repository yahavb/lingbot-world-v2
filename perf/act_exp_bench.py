"""Why is the S^T attention's exp (Scalar, PSUM [128, 512] fp32 -> SBUF bf16) ~0.5 us in-kernel? One logical core,
NT key tiles per call, cost per tile = slope between two NT. Variants add one consumer at a time:
  mm1_exp        MM1 (bf16 128x128 x 512) -> exp
  mm1_exp_mm2    + MM2 (V^T x P^T, 512 columns, accumulating into one PSUM bank)
  mm1_exp_add    + Vector add of P^T into a bf16 accumulator
  full           MM1 -> exp -> MM2 + add (the kernel's pattern)
  full_altexp    the same with every other exp on Vector (exponential)
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/act_exp_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_helpers import PSUM_BANK_SIZE

REPS = int(os.environ.get("REPS", "10"))
DEV = torch.device("neuron:0")


def make(mode, nt):
    @nki.jit
    def k(qt, kt, vt):            # qt [128 d, 512 q], kt [128 d, 128 keys], vt [128 keys, 128 d]  (bf16)
        out = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
        q_sb = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        k_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        v_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb, src=qt[0:128, 0:512])
        nisa.dma_copy(dst=k_sb, src=kt[0:128, 0:128])
        nisa.dma_copy(dst=v_sb, src=vt[0:128, 0:128])
        o_ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
        acc = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        rsink = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
        pend = []
        for t in range(nt):
            s_ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum, address=(0, (2 + t % 6) * PSUM_BANK_SIZE))
            nisa.nc_matmul(s_ps, stationary=k_sb, moving=q_sb)
            p = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            if mode == "full_altexp" and t % 2 == 1:
                nisa.exponential(dst=p, src=s_ps, max_value=1.0)
            elif mode in ("e_s1", "e_s1_add"):
                nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=-1.0, scale=1.0)
            elif mode in ("e_red", "e_red_add"):
                nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=-1.0, scale=1.0, reduce_op=nl.add,
                                reduce_res=rsink[:, t % 8:t % 8 + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
            elif mode == "e_vec":
                nisa.exponential(dst=p, src=s_ps, max_value=1.0)
            else:
                nisa.activation(dst=p, op=nl.exp, data=s_ps, bias=-1.0, scale=0.01)
            if mode in ("mm1_exp_add", "full", "full_altexp", "e_s1_add", "e_red_add"):
                nisa.tensor_tensor(dst=acc, data1=acc, data2=p, op=nl.add, engine=nisa.engine.vector)
            if mode in ("mm1_exp_mm2", "full", "full_altexp", "e_s1", "e_red", "e_vec", "e_s1_add", "e_red_add"):
                pend.append(p)
                if len(pend) > 2:
                    nisa.nc_matmul(o_ps, stationary=v_sb, moving=pend.pop(0), accumulate=True)
            if mode == "mm1_exp":
                pend.append(p)
                if len(pend) > 4:
                    pend.pop(0)
        for p in pend:
            if mode in ("mm1_exp_mm2", "full", "full_altexp", "e_s1", "e_red", "e_vec", "e_s1_add", "e_red_add"):
                nisa.nc_matmul(o_ps, stationary=v_sb, moving=p, accumulate=True)
        o_sb = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=o_ps)
        a32 = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
        if mode == "mm1_exp":
            nisa.tensor_copy(dst=a32, src=pend[-1])        # keep the exps alive (their last output is read)
        else:
            nisa.tensor_copy(dst=a32, src=acc)
        nisa.tensor_tensor(dst=o_sb, data1=o_sb, data2=a32, op=nl.add)
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
    return (time.perf_counter() - t0) / REPS * 1e6


g = torch.Generator().manual_seed(0)
xi = 0
for mode in os.environ.get("MODES", "mm1_exp_mm2,mm1_exp_add,full,full_altexp").split(","):
    try:
        ts = []
        for nt in [int(v) for v in os.environ.get("NTS", "64,448").split(",")]:
            xi += 1
            qt = (torch.randn(128, 512 + xi, generator=g) * 0.1).to(torch.bfloat16)      # distinct shapes
            kt = (torch.randn(128, 128 + xi, generator=g) * 0.1).to(torch.bfloat16)
            vt = (torch.randn(128 + xi, 128, generator=g) * 0.1).to(torch.bfloat16)
            ts.append(bench(make(mode, nt), qt.to(DEV), kt.to(DEV), vt.to(DEV)))
        if len(ts) > 1:
            print(f"[actx] {mode:12s} {(ts[1] - ts[0]) / (448 - 64) * 1000:7.1f} ns per key tile", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[actx] {mode:12s} FAILED: {str(e).splitlines()[0][:200]}", flush=True)
print("[actx] done", flush=True)
