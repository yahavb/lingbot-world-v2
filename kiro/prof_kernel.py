"""Fine timeline of one HLO op (e.g. an NKI kernel custom-call) on one pcore: busy % of every
engine in bins, plus the opcode mix per bin, to see whether loads and compute overlap.
    python3 kiro/prof_kernel.py <profile dir> <exec_index> <op> [pcore] [bin_us]"""
import sys

import pandas as pd


def merged(iv):
    iv = sorted(iv)
    if not iv:
        return 0
    t, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            t += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return t + ce - cs


def main():
    d, ex, op = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    pc = int(sys.argv[4]) if len(sys.argv) > 4 else None
    b_us = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    cols = ["execution_index", "start_ts", "end_ts", "engine", "pcore_idx", "hlo_name", "opcode"]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=cols)
    e = ins[ins.execution_index == ex].copy()
    t0 = e.start_ts.min()
    e["op"] = e.hlo_name.fillna("<none>").str.split(" = ").str[0]
    v = e[e.op == op]
    if pc is None:
        pc = int(v.pcore_idx.min())
    v = v[v.pcore_idx == pc]
    lo, hi = v.start_ts.min(), v.end_ts.max()
    print(f"{op} on pcore {pc}: {(lo - t0) / 1e3:.1f} .. {(hi - t0) / 1e3:.1f} us; "
          f"busy " + ", ".join(f"{k}={merged(list(zip(w.start_ts, w.end_ts))) / 1e3:.0f}us"
                               for k, w in v.groupby("engine")))
    engines = sorted(v.engine.unique())
    b = int(b_us * 1000)
    print("   t_us  " + " ".join(f"{x[:6]:>6s}" for x in engines) + "   opcodes (count)")
    for s in range(int(lo), int(hi), b):
        w = v[(v.end_ts > s) & (v.start_ts < s + b)]
        vals = []
        for x in engines:
            ww = w[w.engine == x]
            vals.append(100.0 * merged(list(zip(ww.start_ts.clip(lower=s), ww.end_ts.clip(upper=s + b)))) / b)
        oc = w.groupby(["engine", "opcode"]).size().sort_values(ascending=False).head(4)
        ocs = " ".join(f"{en[:3]}:{o[:14]}({n})" for (en, o), n in oc.items())
        print(f"  {(s - t0) / 1e3:6.0f}  " + " ".join(f"{x:6.0f}" for x in vals) + f"   {ocs}")


if __name__ == "__main__":
    main()
