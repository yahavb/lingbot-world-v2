"""kernels/attention_cmax3.py vs a CPU fp32 softmax reference and vs attention_cmax2_lb, at the 2-chip per-rank
shape: 10 heads, 2176 queries, 3 cached chunks + 1 new chunk of 4320 keys, K rank-blocked (k_blocks=2), V
token-major. One logical core, eager.
  LINGBOT_CMAX_BANK_REUSE=1 PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/cmax3_bench.py
Env: N (10), LQ (2176), LK (4320), NC (3 cached chunks), QS (query scale, 4.0), C (offset, 30), REPS (10), CMP2 (1).
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
from kernels import attention_cmax3 as c3

N, D = int(os.environ.get("N", "10")), 128
LQ, LK, NC = int(os.environ.get("LQ", "2176")), int(os.environ.get("LK", "4320")), int(os.environ.get("NC", "3"))
QS, C = float(os.environ.get("QS", "4.0")), float(os.environ.get("C", "30"))
REPS = int(os.environ.get("REPS", "10"))
KB = 2
DEV = torch.device("neuron:0")
g = torch.Generator().manual_seed(0)
q = (torch.randn(N, D, LQ, generator=g) * QS / D ** 0.5).to(torch.bfloat16)          # pre-scaled, [N, d, Sq]
ks = [torch.randn(N, D, LK, generator=g).to(torch.bfloat16) for _ in range(NC + 1)]    # logical [N, d, Sk]
vs = [torch.randn(LK, N, D, generator=g).to(torch.bfloat16) for _ in range(NC + 1)]    # token-major [Sk, N, d]


def kmem(k):   # logical [N, d, Sk] -> rank-blocked memory [KB, N, d, Sk / KB], viewed back as [N, d, Sk]
    lb = LK // KB
    return k.view(N, D, KB, lb).permute(2, 0, 1, 3).contiguous().view(N, D, LK)


# reference: softmax over all chunks' keys
kk = torch.cat([k.float() for k in ks], dim=2)                     # [N, d, Sk_total]
vv = torch.cat([v.float().permute(1, 0, 2) for v in vs], dim=1)   # [N, Sk_total, d]
s = torch.einsum("ndq,ndk->nqk", q.float(), kk)
print(f"[c3] row max range {s.amax(-1).min():.2f} .. {s.amax(-1).max():.2f}, offset {C}", flush=True)
ref = torch.einsum("nqk,nkd->ndq", torch.softmax(s, -1), vv)      # [N, d, Sq]
del s
qd = q.to(DEV)
kd = [kmem(k).to(DEV) for k in ks]
vd = [v.to(DEV) for v in vs]


def bench(fn):
    for _ in range(2):
        y = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


def rel(a, b):
    a = a.float().cpu()
    return ((a - b).norm() / b.norm()).item()


part = {1: c3.cmax3_part_1, 2: c3.cmax3_part_2, 3: c3.cmax3_part_3}[NC]


def run3():
    o_acc, s_acc = part[2](qd, *kd[:NC], *vd[:NC], C, KB, True)
    return c3.cmax3_final_1[2](qd, kd[NC], vd[NC], o_acc, s_acc, C, KB, True)


us3, o3 = bench(run3)
print(f"[c3] cmax3 part{NC}+final1 {us3:8.1f} us  rel vs fp32 {rel(o3, ref):.2e}", flush=True)
TOK = int(os.environ.get("TOK", "0") or 0)
if TOK:
    # LINGBOT_C3_TOKOUT: the final call's token-major [Sq, TOK] output (heads x d, then zero columns)
    def run3t():
        o_acc, s_acc = part[2](qd, *kd[:NC], *vd[:NC], C, KB, True)
        return c3.cmax3_final_tok_1[2](qd, kd[NC], vd[NC], o_acc, s_acc, C, KB, True, TOK)
    us3t, o3t = bench(run3t)
    ot = o3t.float().cpu()
    ref_t = ref.permute(2, 0, 1).reshape(LQ, N * D)
    print(f"[c3] cmax3 part{NC}+final1 tok {us3t:8.1f} us  rel vs fp32 {rel(ot[:, :N * D], ref_t):.2e}  "
          f"pad max |x| {ot[:, N * D:].abs().max().item() if TOK > N * D else 0.0:.1e}", flush=True)
if os.environ.get("CMP2", "1") == "1":
    from kernels import attention_cmax2_lb as c2
    part2 = {1: c2.cmax2_part_1, 2: c2.cmax2_part_2, 3: c2.cmax2_part_3}[NC]

    def run2():
        acc = part2[2](qd, *kd[:NC], *vd[:NC], C, KB, True)
        return c2.cmax2_final_1[2](qd, kd[NC], vd[NC], acc, C, KB, True)
    try:
        us2, o2 = bench(run2)
        print(f"[c3] cmax2 part{NC}+final1 {us2:8.1f} us  rel vs fp32 {rel(o2.permute(0, 2, 1), ref):.2e}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[c3] cmax2 FAILED {str(e).splitlines()[0][:200]}", flush=True)
print("[c3] done", flush=True)
