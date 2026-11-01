"""Replay one rank's constant-offset attention from a block_bench DUMP_CMAX file on one core: the dumped kernel queries
and cache chunks through the same cmax2 calls (cached-chunk call, then the final call), against the dumped
accumulator and output and an fp32 reference. Run with LINGBOT_DIAG_NOGATHER=1 dumps (the final chunk is then the
cache slot the dump holds).
    NEURON_RT_VISIBLE_CORES=0 LINGBOT_CMAX=2 python3 kiro/cmax_replay.py /work/out/cmax_dump/r00.pt"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ATTENTION_BACKEND", "cte")
os.environ.setdefault("LINGBOT_HOT_ATTN", "cte")
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from wan.modules import attention as A  # noqa: E402

OFF = float(os.environ.get("LINGBOT_CMAX_OFFSET", "30"))


def stats(name, t):
    t = t.float()
    fin = torch.isfinite(t)
    return (f"{name}: shape {tuple(t.shape)} finite {bool(fin.all())} max|.| "
            f"{t[fin].abs().max().item() if fin.any() else float('nan'):.3e}")


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def main(path):
    z = torch.load(path)
    q_t, acc_d, o_d, ks, vs, cp = z["q_t"], z["acc"], z["o_n"], z["ks"], z["vs"], z["cp"]
    lq = z["L_loc"]
    print(stats("q_t", q_t), "|", stats("acc (dumped)", acc_d), "|", stats("o_n (dumped)", o_d), flush=True)
    dev = torch.device("neuron")
    qd = q_t.to(dev)
    kd = [t.to(dev) for t in ks]
    vd = [t.to(dev) for t in vs]

    def graph(q_, *kv):
        k_, v_ = list(kv[:len(ks)]), list(kv[len(ks):])
        acc = A.cmax_partial(q_, k_[:-1], v_[:-1], OFF, cp, True)
        return acc, A.cmax_final(q_, k_[-1], v_[-1], acc, OFF, cp, True)
    f = torch.compile(graph, backend="neuron", fullgraph=True, dynamic=False)
    with torch.no_grad():
        acc_r, o_r = (t.cpu() for t in f(qd, *kd, *vd))
    # fp32 reference from the logical layouts: K memory [cp, n, d, l] -> [n, d, L]; V memory [L, n, d] -> [n, L, d]
    n, d, Lk = ks[0].shape
    Kl = [k.reshape(cp, n, d, Lk // cp).permute(1, 2, 0, 3).reshape(n, d, Lk).float() for k in ks]
    Vl = [v.reshape(Lk, n, d).permute(1, 0, 2).float() for v in vs]
    sc = torch.matmul(q_t.float().permute(0, 2, 1), torch.cat(Kl, 2))                   # [n, Lq_p, S], pre-scaled q
    ref = torch.softmax(sc, -1) @ torch.cat(Vl, 1)                                      # [n, Lq_p, d]
    print(f"[cmax_replay] row max of scores: {sc.amax(-1).max().item():.2f}; "
          f"replay acc vs dumped rel {rel(acc_r[:, :lq], acc_d[:, :lq]):.3e}; "
          f"replay out vs fp32 rel {rel(o_r[:, :lq], ref[:, :lq]):.3e}; "
          f"dumped out vs fp32 rel {rel(o_d[:, :lq], ref[:, :lq]):.3e}", flush=True)
    bad = ~torch.isfinite(o_d.float()) | ((o_d.float() - ref).abs() > 1.0)
    if bad.any():
        idx = bad.nonzero()
        heads = sorted(set(idx[:, 0].tolist()))
        rows = idx[:, 1]
        print(f"[cmax_replay] dumped out bad elements {int(bad.sum())}: heads {heads}, query rows "
              f"{int(rows.min())}..{int(rows.max())} ({len(set(rows.tolist()))} rows)", flush=True)
    bad_a = ~torch.isfinite(acc_d.float())
    if bad_a.any():
        idx = bad_a.nonzero()
        print(f"[cmax_replay] dumped acc non-finite {int(bad_a.sum())}: heads {sorted(set(idx[:, 0].tolist()))}, "
              f"rows {int(idx[:, 1].min())}..{int(idx[:, 1].max())}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
