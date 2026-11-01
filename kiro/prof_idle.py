"""Idle windows of pcore 0 in one execution of an ingested device profile (no engine running an instruction), and the
instructions around the longest ones on every engine.
    python3 kiro/prof_idle.py <session dir> [min_idle_us] [exec_index|auto]"""
import sys

import pandas as pd


def main():
    d = sys.argv[1]
    min_us = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    ins = pd.read_parquet(f"{d}/Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "pcore_idx", "engine", "opcode",
                                   "hlo_name", "operands", "nki_source_location"])
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    if len(sys.argv) > 3 and sys.argv[3] != "auto":
        ex = int(sys.argv[3])
    else:
        big = g[g.n > 0.9 * g.n.max()].sort_values("ms")
        ex = big.index[len(big) // 2]
    t0 = int(g.loc[ex, "t0"])
    x = ins[(ins.execution_index == ex) & (ins.pcore_idx == ins.pcore_idx.min())].copy()
    x["s"] = (x.start_ts - t0) / 1e3
    x["e"] = (x.end_ts - t0) / 1e3
    x = x.sort_values("s")
    print(f"exec {ex}: {(g.loc[ex, 't1'] - t0) / 1e3:.1f} us, {len(x)} instructions on pcore {ins.pcore_idx.min()}")
    # union of busy intervals
    gaps, cur = [], None
    for s, e in zip(x.s.values, x.e.values):
        if cur is None:
            cur = e
            continue
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    gaps = [(a, b) for a, b in gaps if b - a >= min_us]
    print(f"idle windows >= {min_us} us: {len(gaps)}, total {sum(b - a for a, b in gaps):.1f} us")
    for a, b in sorted(gaps, key=lambda t: t[0] - t[1])[:8]:
        print(f"\n--- idle {a:.1f} -> {b:.1f} us ({b - a:.1f} us)")
        for eng, v in x.groupby("engine"):
            before = v[v.e <= a + 0.01].tail(2)
            after = v[v.s >= b - 0.01].head(2)
            for tag, rows in (("before", before), ("after", after)):
                for _, r in rows.iterrows():
                    loc = str(r.nki_source_location) if r.nki_source_location is not None else ""
                    print(f"  {eng:7} {tag:6} {r.s:8.1f}->{r.e:8.1f} {str(r.opcode):18} {str(r.hlo_name)[:60]:60} "
                          f"{loc[-50:]} | {str(r.operands)[:90]}")


if __name__ == "__main__":
    main()
