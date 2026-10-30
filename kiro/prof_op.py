"""Instruction-level detail of selected HLO ops in one execution: per engine and pcore the count,
first start / last end (us from the execution start), busy time and a start-time histogram.
    python3 kiro/prof_op.py <profile dir> <exec_index> <op substring> [<op substring> ...]
Shows whether an op's span is real work or a few late instructions (e.g. DMA descriptor
generation on GpSimd, a queue on Sync) and which engine it waits on."""
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
    d, ex, pats = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    cols = ["execution_index", "start_ts", "end_ts", "engine", "pcore_idx", "hlo_name", "opcode",
            "hbm_read_bytes", "hbm_write_bytes"]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=cols)
    e = ins[ins.execution_index == ex].copy()
    t0 = e.start_ts.min()
    e["op"] = e.hlo_name.fillna("<none>").str.split(" = ").str[0]
    for p in pats:
        v = e[e.op.str.contains(p, regex=False)]
        if v.empty:
            print(f"== {p}: no instructions")
            continue
        full = v.hlo_name.dropna().iloc[0] if v.hlo_name.notna().any() else p
        print(f"== {p}: {len(v)} instructions  [{str(full)[:160]}]")
        for (eng, pc), w in v.groupby(["engine", "pcore_idx"]):
            s = (w.start_ts - t0) / 1e3
            hist = pd.cut(s, bins=range(0, 3100, 100)).value_counts(sort=False)
            hs = " ".join(f"{int(b.left)}:{c}" for b, c in hist.items() if c)
            ops = w.opcode.value_counts().head(3).to_dict()
            rd = w.hbm_read_bytes.fillna(0).sum() / 1e6
            wr = w.hbm_write_bytes.fillna(0).sum() / 1e6
            print(f"   {eng:7s} pc{int(pc)} n={len(w):5d} {s.min():7.1f}..{(w.end_ts.max() - t0) / 1e3:7.1f} us"
                  f" busy {merged(list(zip(w.start_ts, w.end_ts))) / 1e3:6.1f} us  rd {rd:5.1f} MB wr {wr:5.1f} MB"
                  f"  {ops}")
            print(f"      starts/100us: {hs}")


if __name__ == "__main__":
    main()
