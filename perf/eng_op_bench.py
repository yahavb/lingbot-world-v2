"""Per-instruction cost of the attention's elementwise ops on [128, 512] tiles, one logical core: exp from PSUM fp32
into bf16 SBUF on Scalar (activation, with / without the fused row sum), Vector (exponential) and GpSimd
(activation), and bf16 adds on Vector / GpSimd. Each op runs N times on rotating buffers; cost = slope between
two N.
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/eng_op_bench.py
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
NB = 6


def make(op, n):
    @nki.jit
    def k(x_hbm):
        out = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        x = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x, src=x_hbm[0:128, 0:512])
        ps = [nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum, address=(0, (2 + b) * PSUM_BANK_SIZE))
              for b in range(NB)]
        ones = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=0.01)
        for b in range(NB):
            nisa.nc_matmul(ps[b], stationary=ones, moving=x)
        outs = [nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(NB)]
        acc = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=acc, src=x)
        rs = nl.ndarray((128, NB), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(n):
            b = i % NB
            if op == "act_exp":
                nisa.activation(dst=outs[b], op=nl.exp, data=ps[b], bias=-1.0, scale=1.0)
            elif op == "act_exp_sum":
                nisa.activation(dst=outs[b], op=nl.exp, data=ps[b], bias=-1.0, scale=1.0, reduce_op=nl.add,
                                reduce_res=rs[:, b:b + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
            elif op == "vec_exp":
                nisa.exponential(dst=outs[b], src=ps[b], max_value=1.0)
            elif op == "gps_exp":
                nisa.activation(dst=outs[b], op=nl.exp, data=ps[b], bias=-1.0, scale=1.0, engine=nisa.engine.gpsimd)
            elif op == "vec_add":
                nisa.tensor_tensor(dst=outs[b], data1=outs[(b + 1) % NB], data2=x, op=nl.add, engine=nisa.engine.vector)
            elif op == "gps_add":
                nisa.tensor_tensor(dst=outs[b], data1=outs[(b + 1) % NB], data2=x, op=nl.add, engine=nisa.engine.gpsimd)
            elif op == "vec_add_acc":   # the kernel's pattern: one accumulator
                nisa.tensor_tensor(dst=acc, data1=acc, data2=outs[b], op=nl.add, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=outs[0], op=nl.add)
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
for op in ("act_exp", "act_exp_sum", "vec_exp", "gps_exp", "vec_add", "gps_add", "vec_add_acc"):
    try:
        ts = []
        for n in (48, 336):
            xi += 1
            x = torch.randn(128 + xi, 512, generator=g).to(torch.bfloat16).to(DEV)      # distinct shapes
            ts.append(bench(make(op, n), x))
        print(f"[eng] {op:12s} {(ts[1] - ts[0]) / (336 - 48) * 1000:7.1f} ns per [128, 512] instruction", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[eng] {op:12s} FAILED: {str(e).splitlines()[0][:200]}", flush=True)
print("[eng] done", flush=True)
