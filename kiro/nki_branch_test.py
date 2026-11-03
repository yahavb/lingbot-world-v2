"""Runtime branches inside an LNC2 NKI kernel (nl.fori_loop on a register loaded from a computed flag), on one
logical core. Found: a branch taken on one physical core only hangs the logical core (NRT_TIMEOUT), so both cores
must take every branch the same number of times.
  local  each core branches on its own flags (only valid when they agree: symmetric cases)
  pair   the cores exchange their per-block flags (nisa.sendrecv, GpSimd SB-to-SB) and branch on the OR, so block slot
         b is redone on both cores when either core's block b is bad
Checks the flags (band [e^-60, e^78] on column 0 of every block: inf, NaN, 0 and 1e35 must fail, 1e-26 must pass),
the outputs, and the cost of the check when no branch is taken.
    NEURON_RT_VISIBLE_CORES=<c> NEURON_RT_EXEC_TIMEOUT=30 python3 kiro/nki_branch_test.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402

from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info  # noqa: E402

P, B, W = 128, 4, 128          # partitions, blocks per program, columns per block
LO, HI = 8.75651076269652e-27, 7.498221244986777e33   # e^-60, e^78


def _body(x, mode):
    """x [2, P, B, W] fp32, one slab per program. out = x, except redone blocks, which get 2 * x through per-block
    runtime branches inside a global one. flags [2, B]: this core's own band failures (1.0)."""
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    flags = nl.ndarray((2, B), dtype=nl.float32, buffer=nl.shared_hbm)
    _, n_prgs, prg = get_verified_program_sharding_info("nki_branch_test", max_sharding=2)
    t = nl.ndarray((P, B, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x[prg])
    nisa.dma_copy(dst=out[prg], src=t)
    bad = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
    if mode == "plain":
        nisa.memset(bad, 0.0)
        nisa.dma_copy(dst=flags[prg:prg + 1, :], src=bad)
        return out, flags
    # band check of column 0 of every block: ok = (v >= LO) * (v <= HI) (NaN: 0), bad = any partition not ok
    le = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=le, data=t[:, :, 0:1], op0=nl.less_equal, operand0=HI)
    ok = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=ok, data=t[:, :, 0:1], op0=nl.greater_equal, operand0=LO, op1=nl.multiply,
                              operand1=le)
    ones = nl.ndarray((P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    n_ok = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=n_ok, stationary=ones, moving=ok)
    nisa.tensor_scalar(dst=bad, data=n_ok, op0=nl.less, operand0=P - 0.5)
    nisa.dma_copy(dst=flags[prg:prg + 1, :], src=bad)
    sel = bad
    if mode == "pair":
        other = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=bad, dst=other, send_to_rank=1 - prg, recv_from_rank=1 - prg, pipe_id=0,
                      dma_engine=nisa.dma_engine.gpsimd_dma)
        sel = nl.ndarray((1, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sel, data1=bad, data2=other, op=nl.maximum)
    sel_i = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sel_i, src=sel)
    any_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=any_f, data=sel, op=nl.maximum, axis=(1,))
    any_i = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=any_i, src=any_f)
    reg = nisa.register_alloc()
    nisa.register_load(reg, any_i)

    def redo_all(i):
        for b in range(B):
            rb = nisa.register_alloc()
            nisa.register_load(rb, sel_i[0:1, b:b + 1])

            def redo_block(j, b=b):
                t2 = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=t2, data=t[:, b, :], op0=nl.multiply, operand0=2.0)
                nisa.dma_copy(dst=out[prg, :, b, :], src=t2)
            nl.fori_loop(0, rb, redo_block)
    nl.fori_loop(0, reg, redo_all)
    return out, flags


@nki.jit
def k_plain(x):
    return _body(x, "plain")


@nki.jit
def k_local(x):
    return _body(x, "local")


@nki.jit
def k_pair(x):
    return _body(x, "pair")


def main():
    dev = torch.device("neuron")
    fns = {}
    for name, k in (("plain", k_plain), ("local", k_local), ("pair", k_pair)):
        w = wrap_nki(k)
        fns[name] = torch.compile(lambda x, w=w: w(x), backend="neuron", fullgraph=True, dynamic=False)
    base = torch.rand(2, P, B, W) + 1.0                       # every value in band
    x = base.to(dev)
    for name in ("plain", "local", "pair"):
        f_ = fns[name]
        with torch.no_grad():
            for _ in range(5):
                f_(x)
            torch.neuron.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                f_(x)
            torch.neuron.synchronize()
        print(f"[branch] {name} (no branch taken): {(time.perf_counter() - t0) / 200 * 1e6:.1f} us per call",
              flush=True)
    cases = [("pair", "none", []), ("pair", "p0b1_inf", [(0, 1, float("inf"))]),
             ("pair", "p1b3_nan", [(1, 3, float("nan"))]),
             ("pair", "p0b0_zero_p1b2_big", [(0, 0, 0.0), (1, 2, 1e35)]),
             ("pair", "p0b2_lo_edge_ok", [(0, 2, 1e-26)]),
             ("local", "sym_b1_inf", [(0, 1, float("inf")), (1, 1, float("inf"))]),
             ("local", "sym_b0_zero_b3_nan", [(0, 0, 0.0), (1, 0, 0.0), (0, 3, float("nan")), (1, 3, float("nan"))])]
    ok_all = True
    for mode, name, edits in cases:
        x = base.clone()
        own = torch.zeros(2, B)
        for prg, b, val in edits:
            x[prg, 37, b, 0] = val
            own[prg, b] = 0.0 if name.endswith("_ok") else 1.0
        redo = own.amax(0, keepdim=True).expand(2, B) if mode == "pair" else own
        with torch.no_grad():
            o, f = fns[mode](x.to(dev))
        o, f = o.cpu(), f.cpu()
        exp = x.clone()
        for prg in range(2):
            for b in range(B):
                if redo[prg, b]:
                    exp[prg, :, b, :] *= 2.0
        same = torch.equal(torch.nan_to_num(o, nan=-1.0), torch.nan_to_num(exp, nan=-1.0))
        good = bool(same) and torch.equal(f, own)
        ok_all &= good
        print(f"[branch] {mode} {name}: flags {f.tolist()} expected {own.tolist()} output as expected {same} "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    print(f"[branch] ALL {'PASS' if ok_all else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
