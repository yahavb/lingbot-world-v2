"""How attention_cte's cost scales with query tiles, key length and call count, at the DiT's
per-rank shapes (10 heads, d 128, KV chunks of 4320 tokens). One logical core, no collectives.
    NEURON_RT_VISIBLE_CORES=0 python3 kiro/attn_tile_bench.py
Each case is one compiled graph holding only the kernel call(s) of cte_partial (K pre-transposed
[n, d, L], V [n, L, d], q pre-scaled and N-major), timed over REPS launches after warmup."""
import os
import sys
import time

os.environ.setdefault("ATTENTION_BACKEND", "cte")
os.environ.setdefault("LINGBOT_HOT_ATTN", "cte")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from wan.modules import attention as A  # noqa: E402

REPS = int(os.environ.get("REPS", "50"))
n, d, L = 10, 128, 4320
dev = torch.device("neuron")
g = torch.Generator().manual_seed(0)


def rnd(*s):
    return torch.randn(*s, generator=g).to(torch.bfloat16).to(dev)


def k_(lk):
    return rnd(n, d, lk)


def v_(lk):
    return rnd(n, lk, d)


def one(q, k, v):
    return A.cte_partial(q, k, v)


def prior(q, k, v, kp, vp):
    return A.cte_partial(q, k, v, kp, vp)


def pair2(q, k0, v0, k1, v1, k2, v2, k3, v3):
    """the current steady-state block: (lead0 prior, lead1) + (lead2 prior, new)"""
    return A.cte_partial(q, k1, v1, k0, v0), A.cte_partial(q, k3, v3, k2, v2)


def single4(q, k0, v0, k1, v1, k2, v2, k3, v3):
    """the same 4 chunks as 4 calls without priors"""
    return tuple(A.cte_partial(q, kk, vv) for kk, vv in ((k0, v0), (k1, v1), (k2, v2), (k3, v3)))


def bench(name, fn, *args):
    cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    with torch.no_grad():
        t0 = time.perf_counter()
        cf(*args)
        torch.neuron.synchronize()
        tf = time.perf_counter() - t0
        for _ in range(3):
            cf(*args)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            cf(*args)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / REPS * 1e3
    print(f"[attn_tile] {name:44s} {ms:7.3f} ms   (first {tf:.0f} s)", flush=True)
    return ms


def check():
    """cte_partial (normalized out / sum, and the returned row max) against fp32 attention on the
    CPU over the same bf16 inputs, for the call shapes the DiT block uses."""
    import math as _m
    gq = torch.Generator().manual_seed(7)

    def r(*s):
        return torch.randn(*s, generator=gq).to(torch.bfloat16)
    for lq, lk_prior, lk in ((384, 4320, 4320), (384, 0, 4320), (256, 0, 1080), (256, 12960, 4320)):
        q = r(n, lq, d) * (1.0 / _m.sqrt(d))
        q = q.to(torch.bfloat16)
        k = r(n, d, lk)
        v = r(n, lk, d)
        kp = r(n, d, lk_prior) if lk_prior else None
        vp = r(n, lk_prior, d) if lk_prior else None
        args = [q.to(dev), k.to(dev), v.to(dev)] + ([kp.to(dev), vp.to(dev)] if lk_prior else [])
        fn = prior if lk_prior else one
        cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        with torch.no_grad():
            o, m_, s_ = cf(*args)
        o, m_, s_ = o.float().cpu(), m_.float().cpu(), s_.float().cpu()
        g_ = int(m_.shape[2])
        m_ = m_.transpose(1, 2).reshape(n, g_ * 128)[:, :lq]
        s_ = s_.transpose(1, 2).reshape(n, g_ * 128)[:, :lq]
        out = o[:, :lq] / s_.unsqueeze(-1)
        K = torch.cat(([kp.float()] if lk_prior else []) + [k.float()], dim=2)      # [n, d, L]
        V = torch.cat(([vp.float()] if lk_prior else []) + [v.float()], dim=1)      # [n, L, d]
        sc = torch.matmul(q.float(), K)                                            # [n, lq, L]
        ref = torch.softmax(sc, -1) @ V
        rmax = sc.amax(-1)
        rel = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        print(f"[attn_tile] check lq={lq} prior={lk_prior} keys={lk}: finite={bool(torch.isfinite(out).all())}"
              f" rel_rms={rel:.5f} max|d|={(out - ref).abs().max().item():.3e}"
              f" rowmax max|d|={(m_ - rmax).abs().max().item():.3e}", flush=True)


def main():
    if os.environ.get("CASES", "") == "check":
        check()
        return
    if os.environ.get("CASES", "") == "prof":
        # the steady block's two calls only (run under the runtime inspector)
        four = [t for _ in range(4) for t in (k_(L), v_(L))]
        global REPS
        REPS = 5
        bench("lq=384 pair2 (2 calls, 4 x 4320 keys)", pair2, rnd(n, 384, d), *four)
        return
    if os.environ.get("CASES", "") == "dge":
        # the steady block's two calls, at its query count, plus one cached chunk alone
        four = [t for _ in range(4) for t in (k_(L), v_(L))]
        q384 = rnd(n, 384, d)
        bench("lq=384 pair2 (2 calls, 4 x 4320 keys)", pair2, q384, *four)
        bench("lq=384 one call, prior 4320 + 4320", prior, q384, k_(L), v_(L), k_(L), v_(L))
        bench("lq=384 one call, 4320 keys", one, q384, k_(L), v_(L))
        return
    if os.environ.get("CASES", "") == "qm":
        # the steady block's two calls: 128-padded (384) vs 32-padded (288) queries, head-major vs
        # query-major outputs (LINGBOT_CTE_QMAJOR; vendored kernel), with and without the merge
        four = [t for _ in range(4) for t in (k_(L), v_(L))]
        def pair2_qm(q, k0, v0, k1, v1, k2, v2, k3, v3, qm=False):
            return (A.cte_partial(q, k1, v1, k0, v0, q_major=qm),
                    A.cte_partial(q, k3, v3, k2, v2, q_major=qm))
        def pair2_merge(q, k0, v0, k1, v1, k2, v2, k3, v3, qm=False):
            return A.cte_merge(list(pair2_qm(q, k0, v0, k1, v1, k2, v2, k3, v3, qm=qm)), 288,
                               q_major=qm)
        for lq in (384, 288):
            q = rnd(n, lq, d)
            for qm in (False, True):
                bench(f"lq={lq} q_major={int(qm)} pair2", lambda *a, qm=qm: pair2_qm(*a, qm=qm), q, *four)
                bench(f"lq={lq} q_major={int(qm)} pair2 + merge", lambda *a, qm=qm: pair2_merge(*a, qm=qm),
                      q, *four)
        return
    if os.environ.get("CASES", "") == "contig":
        # one call over a contiguous window vs the block's two calls, and the in-graph write of
        # the new chunk into a slot of a contiguous window buffer (what a single-call cache needs)
        four = [t for _ in range(4) for t in (k_(L), v_(L))]
        for lq in (384, 256):
            q = rnd(n, lq, d)
            bench(f"lq={lq} pair2 (2 calls, 4 x 4320 keys)", pair2, q, *four)
            bench(f"lq={lq} one call, 17280 keys", one, q, k_(4 * L), v_(4 * L))
            bench(f"lq={lq} one call, 8640 keys", one, q, k_(2 * L), v_(2 * L))
        kw, vw = k_(4 * L), v_(4 * L)
        for t in (kw, vw):
            torch._dynamo.mark_static_address(t)
        kn, vn = k_(L), v_(L)

        def write_then_one(q, kw, vw, kn, vn):
            kw[:, :, 2 * L:3 * L].copy_(kn)
            vw[:, 2 * L:3 * L].copy_(vn)
            return A.cte_partial(q, kw, vw)
        for lq in (384, 256):
            bench(f"lq={lq} write slot 2 + one call 17280", write_then_one, rnd(n, lq, d), kw, vw, kn, vn)
        # the write must land in the buffer the caller holds
        kw.zero_()
        torch.neuron.synchronize()
        cf = torch.compile(write_then_one, backend="neuron", fullgraph=True, dynamic=False)
        with torch.no_grad():
            cf(rnd(n, 256, d), kw, vw, kn, vn)
        torch.neuron.synchronize()
        ok = torch.equal(kw[:, :, 2 * L:3 * L].cpu(), kn.cpu()) and bool((kw[:, :, :2 * L].cpu() == 0).all())
        print(f"[attn_tile] in-place slot write visible to the caller: {ok}", flush=True)
        return
    four = [t for _ in range(4) for t in (k_(L), v_(L))]
    for lq in (384, 256, 128):
        q = rnd(n, lq, d)
        bench(f"lq={lq} pair2 (2 calls, 4 x 4320 keys)", pair2, q, *four)
    q256, q384 = rnd(n, 256, d), rnd(n, 384, d)
    bench("lq=384 single4 (4 calls, 4 x 4320 keys)", single4, q384, *four)
    bench("lq=256 single4 (4 calls, 4 x 4320 keys)", single4, q256, *four)
    for lq, q in ((384, q384), (256, q256)):
        bench(f"lq={lq} one call, prior 12960 + 4320", prior, q, k_(L), v_(L), k_(3 * L), v_(3 * L))
        bench(f"lq={lq} one call, 4320 keys", one, q, k_(L), v_(L))
        bench(f"lq={lq} one call, prior 4320 + 4320", prior, q, k_(L), v_(L), k_(L), v_(L))
    for lk in (1080, 270):
        bench(f"lq=256 one call, {lk} keys", one, q256, k_(lk), v_(lk))
    bench("lq=128 one call, 1080 keys", one, rnd(n, 128, d), k_(1080), v_(1080))


if __name__ == "__main__":
    main()
