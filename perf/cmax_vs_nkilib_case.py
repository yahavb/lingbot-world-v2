"""nkilib's attention_const_max in this repo's DiT self-attention, at the 2-chip per-rank shape and the production KV
cache layout: N heads, 2176 queries, a window of 4 chunks of L keys (3 cached + the new one), K chunks rank-blocked
over the CP group (k_blocks), V chunks token-major, offset c = 30 (the band that covers this model's row maxima).
  ours:      kernels/attention_cmax3.py cmax3_part_3 over the 3 cached chunks + cmax3_final_1 over the new one, on the
             chunks as they are
  nkilib:    the chunks re-laid out and concatenated into one head-major K [N, d, 4L] and one V [N, 4L, d] (what
             attention_const_max reads), then attention_const_max; and the kernel alone on prebuilt K / V
    NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES=0 LINGBOT_CMAX=3 LINGBOT_C3_SUMMODE=pef LINGBOT_C3_SUMGROUP=16 \
        LINGBOT_C3_DEPTH=3 python3 perf/cmax_vs_nkilib_case.py
Accuracy (CASEACC) on plain layouts (k_blocks=1, V plain) against fp32 CPU attention of head 0.
"""
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402
from kernels import attention_cmax3 as cm3  # noqa: E402
from nkilib.experimental.attention.attention_const_max import attention_const_max  # noqa: E402


def timeit(fn, reps):
    out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.neuron.synchronize()
    return out, (time.perf_counter() - t0) / reps * 1e6


def main():
    N, d, Sq, L, KB = (int(v) for v in os.environ.get("SHAPE", "10,128,2176,4320,2").split(","))
    c = float(os.environ.get("OFFSET", "30"))
    reps = int(os.environ.get("REPS", "20"))
    dev = torch.device("neuron:0")
    s = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(N, Sq, d).bfloat16()
    ks = [torch.randn(N, d, L).bfloat16() for _ in range(4)]       # logical [N, d, L] per chunk
    vs = [torch.randn(N, L, d).bfloat16() for _ in range(4)]       # logical [N, L, d] per chunk
    q_t = (q.float() * s).bfloat16().permute(0, 2, 1).contiguous().to(dev)
    q_u = q.permute(0, 2, 1).contiguous().to(dev)
    f_p3, f_f1 = wrap_nki(cm3.cmax3_part_3), wrap_nki(cm3.cmax3_final_1)
    f_nk = wrap_nki(attention_const_max)

    # accuracy, plain layouts
    kp = [k.to(dev) for k in ks]
    vp = [v.to(dev) for v in vs]
    kf = torch.cat([k[0].float() for k in ks], 1)                   # head 0 [d, 4L]
    vf = torch.cat([v[0].float() for v in vs], 0)                   # head 0 [4L, d]
    ref0 = torch.softmax(q[0].float() @ kf * s, -1) @ vf             # [Sq, d]

    def ours(qq, k0, k1, k2, k3, v0, v1, v2, v3, kb, vtm):
        o, sm = f_p3(qq, k0, k1, k2, v0, v1, v2, c, kb, vtm)
        return f_f1(qq, k3, v3, o, sm, c, kb, vtm)                   # [N, d, Sq]
    g_ours_plain = torch.compile(lambda *a: ours(*a, 1, False), backend="neuron", fullgraph=True, dynamic=False)
    o = g_ours_plain(q_t, *kp, *vp).float().cpu()
    acc_ours = ((o[0].T - ref0).norm() / ref0.norm()).item()
    kfull = torch.cat([k for k in kp], 2).contiguous()
    vfull = torch.cat([v for v in vp], 1).contiguous()
    g_nk = torch.compile(lambda a, b, cc: f_nk(a, b, cc, c, s), backend="neuron", fullgraph=True, dynamic=False)
    o = g_nk(q_u, kfull, vfull).float().cpu()
    acc_nk = ((o[0] - ref0).norm() / ref0.norm()).item()
    print(f"CASEACC c={c} Sk={4 * L} rel_l2 vs fp32: ours {acc_ours:.4e}  nkilib {acc_nk:.4e}", flush=True)

    # timing, production layouts: K chunks rank-blocked (memory [KB, N, d, L/KB]), V chunks token-major ([L, N, d])
    kr = [k.view(N, d, KB, L // KB).permute(2, 0, 1, 3).contiguous().to(dev) for k in ks]   # [KB, N, d, L/KB]
    vt = [v.permute(1, 0, 2).contiguous().to(dev) for v in vs]                              # [L, N, d]
    k_in = [k.view(N, d, L) for k in kr]          # the kernel's logical view of the rank-blocked memory
    v_in = [v.view(N, L, d) for v in vt]          # the kernel's logical view of the token-major memory
    g_ours = torch.compile(lambda *a: ours(*a, KB, True), backend="neuron", fullgraph=True, dynamic=False)
    _, us_ours = timeit(lambda: g_ours(q_t, *k_in, *v_in), reps)

    def nk_case(qq, k0, k1, k2, k3, v0, v1, v2, v3):
        kk = torch.cat([k.view(KB, N, d, L // KB).permute(1, 2, 0, 3).reshape(N, d, L) for k in (k0, k1, k2, k3)], 2)
        vv = torch.cat([v.view(L, N, d).permute(1, 0, 2) for v in (v0, v1, v2, v3)], 1)
        return f_nk(qq, kk.contiguous(), vv.contiguous(), c, s)    # [N, Sq, d]
    g_case = torch.compile(nk_case, backend="neuron", fullgraph=True, dynamic=False)
    _, us_case = timeit(lambda: g_case(q_u, *kr, *vt), reps)
    _, us_nk = timeit(lambda: g_nk(q_u, kfull, vfull), reps)
    print(f"CASETIME N={N} Sq={Sq} window={4} x {L} k_blocks={KB} c={c}: ours (chunks as stored) {us_ours:.1f} us | "
          f"nkilib incl. re-layout + concat {us_case:.1f} us | nkilib kernel alone {us_nk:.1f} us", flush=True)


if __name__ == "__main__":
    main()
