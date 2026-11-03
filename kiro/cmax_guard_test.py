"""The guarded constant-offset attention (kernels/attention_cmax2_lb.py *_g*, LINGBOT_CMAX_GUARD) against the
unguarded kernel and an fp32 reference, on one logical core at the DiT's per-rank shapes (10 heads, d 128, 4320-key
chunks, rank-blocked K, token-major V), with the in-block call structure (cached chunks in one call, the new chunk
in the final call), for 1-4 chunks:
  normal   scores in the band of every clip measured: no block flagged, output bitwise equal to the unguarded kernel
  padoob   only padding query rows (LQ .. LQP - 1, LINGBOT_TOKEN_PAD) overflow: nothing redone, the real rows bitwise
           equal to the unguarded kernel
  mixed    rows whose maximum is far above 108 (inf without the guard), rows whose scores are all below -55 (0 / 0),
           a block whose overflow is in a cached chunk only, the last (partial) query tile, a NaN row, and the
           padding rows of padoob: exactly the real-row blocks and their partner slots on the other core are redone,
           the redone blocks match fp32 attention (NaN rows excepted), every other block is bitwise equal to the
           unguarded kernel
and the cost per call: unguarded, guarded with nothing out of band, guarded with every block redone.
    NEURON_RT_VISIBLE_CORES=<c> LQ=540 LQP=544 NB=8 python3 kiro/cmax_guard_test.py      (CP8; CP16: 270 288 16)"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402

from kernels import attention_cmax2_lb as C2  # noqa: E402

n, d, L = 10, 128, 4320
LQ, LQP, NB = int(os.environ.get("LQ", "540")), int(os.environ.get("LQP", "544")), int(os.environ.get("NB", "8"))
C = float(os.environ.get("CMAX", "30"))
REPS = int(os.environ.get("REPS", "30"))
CHUNKS = [int(s) for s in os.environ.get("CHUNKS", "1,2,3,4").split(",")]
PER, NO = n // 2, -(-LQP // 128)          # heads per physical core, query tiles
dev = torch.device("neuron")
g = torch.Generator().manual_seed(7)
K = {name: wrap_nki(getattr(C2, f"cmax2_{name}")) for name in
     ("full_1", "part_1", "part_2", "part_3", "final_1", "full_1g", "final_1g1", "final_1g2", "final_1g3")}


def blocked(k):
    """[n, d, L] K stored as rank blocks [NB, n, d, L / NB] under the same shape."""
    n_, d_, l_ = k.shape
    return k.view(n_, d_, NB, l_ // NB).permute(2, 0, 1, 3).contiguous().view(n_, d_, l_)


def token_major(v):
    """[n, L, d] V stored token-major [L, n, d] under the same shape."""
    return v.permute(1, 0, 2).contiguous().view(v.shape)


def attend(q_t, ks, vs, guard):
    """The block's structure: one call over the cached chunks (fp32 accumulator), the final call over the new one."""
    m = len(ks)
    if m == 1:
        if guard:
            return K["full_1g"](q_t, ks[0], vs[0], C, NB, True, LQ)
        return K["full_1"](q_t, ks[0], vs[0], C, NB, True)
    acc = K[f"part_{m - 1}"](q_t, *ks[:-1], *vs[:-1], C, NB, True)
    if guard:
        return K[f"final_1g{m - 1}"](q_t, ks[-1], vs[-1], acc, *ks[:-1], *vs[:-1], C, NB, True, LQ)
    return K["final_1"](q_t, ks[-1], vs[-1], acc, C, NB, True)


def reference(q, ks, vs):
    """fp32 softmax(q K^T) V, q [n, LQP, d] pre-scaled, K [n, d, L], V [n, L, d] (plain layouts) -> [n, LQP, d]."""
    kk = torch.cat([k.float() for k in ks], dim=2)
    vv = torch.cat([v.float() for v in vs], dim=1)
    return torch.softmax(torch.matmul(q.float(), kk), -1) @ vv


def rnd(*s, scale=1.0):
    return torch.randn(*s, generator=g) * scale


def make(case, m):
    """q [n, LQP, d] (rows past LQ zero), K chunks [n, d, L], V chunks [n, L, d] (bf16) and the expected flags
    [n, NO] (2 = out of band, 1 = redone as the partner slot, 0)."""
    q = rnd(n, LQP, d, scale=3.0 / math.sqrt(d))
    q[:, LQ:] = 0
    ks = [rnd(n, d, L) for _ in range(m)]
    vs = [rnd(n, L, d) for _ in range(m)]
    own = torch.zeros(n, NO)
    if case == "mixed":
        u = torch.nn.functional.normalize(rnd(d), dim=0)
        # head 1, tile 1: row maxima ~150 (scores std ~36): exp(s - 30) overflows
        q[1, 128:256] *= 12.0
        own[1, 1] = 2
        # head 3, tile 0: every score near -96 (keys share a component along u, the rows point against it)
        for k in ks:
            k[3] = 8.0 * u[:, None] + 0.5 * rnd(d, L)
        q[3, 0:128] = -12.0 * u + 0.05 * rnd(128, d)
        own[3, 0] = 2
        # head 9 (the other core), last tile (partial): overflow
        q[9, (NO - 1) * 128:LQ] *= 14.0
        own[9, NO - 1] = 2
        # head 4, tile 2: scores ~150 in the FIRST chunk only (a cached chunk when m > 1), in band elsewhere
        ks[0][4] = 10.0 * u[:, None] + 0.5 * rnd(d, L)
        r1 = min(384, LQ)
        q[4, 256:r1] = 15.0 * u + 0.05 * rnd(r1 - 256, d)
        own[4, 2] = 2
        # head 6, row 5: NaN
        q[6, 5, 0] = float("nan")
        own[6, 0] = 2
    if case in ("mixed", "padoob"):
        # head 2: its padding rows (LQ .. LQP - 1) overflow, its real rows do not: not redone
        q[2, LQ:LQP] = rnd(LQP - LQ, d, scale=40.0 / math.sqrt(d))
    q, ks, vs = q.to(torch.bfloat16), [k.to(torch.bfloat16) for k in ks], [v.to(torch.bfloat16) for v in vs]
    exp = own.clone()
    for h in range(n):
        p = (h + PER) % n
        for t in range(NO):
            if own[h, t] == 0 and own[p, t] == 2:
                exp[h, t] = 1
    return q, ks, vs, exp


def blocks_of(x):
    """[n, LQP, d] -> per (head, tile) views over the real rows."""
    for h in range(n):
        for t in range(NO):
            r1 = min(LQ, (t + 1) * 128)
            if t * 128 < r1:
                yield h, t, x[h, t * 128:r1]


def rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def timed(f, args):
    with torch.no_grad():
        for _ in range(3):
            f(*args)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            f(*args)
        torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3


def timing_only():
    """TIMING_ONLY=1: ms per call of the unguarded and the guarded kernel (nothing out of band), normal inputs."""
    for m in CHUNKS:
        q, ks, vs, _ = make("normal", m)
        args = ([q.permute(0, 2, 1).contiguous().to(dev)] + [blocked(k).to(dev) for k in ks]
                + [token_major(v).to(dev) for v in vs])
        res = []
        for guard in (False, True, False, True):
            f = torch.compile(lambda q_, *t, m=m, g_=guard: attend(q_, list(t[:m]), list(t[m:]), g_),
                              backend="neuron", fullgraph=True, dynamic=False)
            res.append(timed(f, args))
        print(f"[guard] chunks={m} ms per call (twice): unguarded {res[0]:.3f} / {res[2]:.3f} | guarded "
              f"{res[1]:.3f} / {res[3]:.3f} | diag={os.environ.get('LINGBOT_CMAX_GUARD_DIAG', '')} "
              f"pin={os.environ.get('LINGBOT_CMAX_PSUM_PIN', '')}", flush=True)


def profile_only():
    """PROFILE_ONLY=1 (with NEURON_RT_INSPECT_* set): a few calls of the guarded kernel only, normal inputs."""
    m = CHUNKS[-1]
    q, ks, vs, _ = make("normal", m)
    args = ([q.permute(0, 2, 1).contiguous().to(dev)] + [blocked(k).to(dev) for k in ks]
            + [token_major(v).to(dev) for v in vs])
    guard = os.environ.get("PROFILE_GUARD", "1") == "1"
    f = torch.compile(lambda q_, *t: attend(q_, list(t[:m]), list(t[m:]), guard), backend="neuron",
                      fullgraph=True, dynamic=False)
    print(f"[guard] profiled chunks={m} guard={guard}: {timed(f, args):.3f} ms per call", flush=True)


def main():
    if os.environ.get("TIMING_ONLY", "0") == "1":
        return timing_only()
    if os.environ.get("PROFILE_ONLY", "0") == "1":
        return profile_only()
    ok_all = True
    for m in CHUNKS:
        f_un = torch.compile(lambda q_, *t, m=m: attend(q_, list(t[:m]), list(t[m:]), False), backend="neuron",
                             fullgraph=True, dynamic=False)
        f_g = torch.compile(lambda q_, *t, m=m: attend(q_, list(t[:m]), list(t[m:]), True), backend="neuron",
                            fullgraph=True, dynamic=False)
        for case in ("normal", "padoob", "mixed"):
            q, ks, vs, exp = make(case, m)
            args = ([q.permute(0, 2, 1).contiguous().to(dev)] + [blocked(k).to(dev) for k in ks]
                    + [token_major(v).to(dev) for v in vs])
            with torch.no_grad():
                o_un = f_un(*args).float().cpu()
                o_g, fl = f_g(*args)
                o_g, fl = o_g.float().cpu(), fl.float().cpu()
            flags = torch.cat([fl[0].view(PER, NO), fl[1].view(PER, NO)], 0)        # [n, NO], head order
            ref = reference(q, ks, vs)
            same_flags = torch.equal(flags, exp)
            n_eq = n_cmp = 0
            worst_redo, redo_finite, n_redo = 0.0, True, 0
            bad_un = []
            for (h, t, a), (_, _, b), (_, _, r) in zip(blocks_of(o_g), blocks_of(o_un), blocks_of(ref)):
                if exp[h, t] == 0:
                    n_cmp += 1
                    n_eq += int(torch.equal(a, b))
                else:
                    n_redo += 1
                    keep = torch.isfinite(r).all(-1)
                    redo_finite &= bool(torch.isfinite(a[keep]).all())
                    worst_redo = max(worst_redo, rel(a[keep], r[keep]))
                    if exp[h, t] == 2:
                        bad_un.append(f"h{h}t{t}:{'nan' if torch.isnan(b).any() else 'inf' if torch.isinf(b).any() else f'{rel(b[keep], r[keep]):.2g}'}")
            ok = same_flags and n_eq == n_cmp and redo_finite and worst_redo < 5e-3
            ok_all &= ok
            print(f"[guard] chunks={m} {case}: flags as expected {same_flags} ({int((flags == 2).sum())} out of band, "
                  f"{int((flags == 1).sum())} partner slots) | blocks not redone bitwise equal to unguarded "
                  f"{n_eq}/{n_cmp} | redone {n_redo}: rel vs fp32 worst {worst_redo:.2e} finite {redo_finite} | "
                  f"unguarded on the out-of-band blocks: {' '.join(bad_un) or '-'} | {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not same_flags:
                print(f"[guard]   flags {flags.tolist()}\n[guard]   expected {exp.tolist()}", flush=True)
            if case == "normal":
                t_un, t_g = timed(f_un, args), timed(f_g, args)
                qa = q.float().clone()
                qa[:, :LQ] *= 14.0                                   # every real row overflows: every block redone
                args_a = [qa.to(torch.bfloat16).permute(0, 2, 1).contiguous().to(dev)] + args[1:]
                t_a = timed(f_g, args_a)
                with torch.no_grad():
                    o_a, fl_a = f_g(*args_a)
                ref_a = reference(qa.to(torch.bfloat16), ks, vs)
                print(f"[guard] chunks={m} ms per call: unguarded {t_un:.3f} | guarded, none redone {t_g:.3f} | "
                      f"guarded, all redone {t_a:.3f} (flags 2: {int((fl_a.cpu() == 2).sum())}, rel vs fp32 "
                      f"{rel(o_a.float().cpu()[:, :LQ], ref_a[:, :LQ]):.2e})", flush=True)
    print(f"[guard] ALL {'PASS' if ok_all else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
