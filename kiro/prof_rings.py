"""Collective spans of one block execution from a neuron-explorer device profile: every ALGO_MESH / ALGO_RING
begin-end pair on each CC core, in us from the execution's first instruction, plus the all-engine idle intervals of
the first physical core (no instruction on any engine for more than min_gap_us).
    python3 kiro/prof_rings.py <ingested session dir> [exec_index] [min_gap_us]
exec_index defaults to the median-duration execution among those with the most common (largest) instruction count.
"""
import sys

import pandas as pd


def main():
    d = sys.argv[1]
    ins = pd.read_parquet(f"{d}/Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "pcore_idx"])
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    if len(sys.argv) > 2 and sys.argv[2] != "auto":
        ex = int(sys.argv[2])
    else:
        big = g[g.n > 0.9 * g.n.max()].sort_values("ms")
        ex = big.index[len(big) // 2]
    min_gap = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    t0 = g.loc[ex, "t0"]
    print(f"exec {ex}: {g.loc[ex, 'ms']:.3f} ms, {int(g.loc[ex, 'n'])} instructions "
          f"(block executions: {', '.join(f'{i}:{r.ms:.2f}' for i, r in g[g.n > 0.9 * g.n.max()].iterrows())})")
    cc = pd.read_parquet(f"{d}/CcInstruction.parquet")
    c = cc[cc.execution_index == ex].copy()
    c["s"] = (c.start_ts - t0) / 1e3
    x = c[c.description.str.match(r"ALGO_(MESH|RING)_(BEGIN|END)$")].sort_values("s")
    for core, v in x.groupby("core_idx"):
        st, out = {}, []
        for _, r in v.iterrows():
            kind = r.description.split("_")[1]
            if r.description.endswith("BEGIN"):
                st.setdefault(kind, []).append(r.s)
            elif st.get(kind):
                b = st[kind].pop(0)
                out.append(f"{kind} {b:.0f}-{r.s:.0f} ({r.s - b:.0f})")
        print(f"  cc core {core}: " + ", ".join(out))
    e = ins[(ins.execution_index == ex)]
    p0 = e.pcore_idx.min()
    iv = sorted(((s - t0) / 1e3, (t - t0) / 1e3) for s, t in e[e.pcore_idx == p0][["start_ts", "end_ts"]].to_numpy())
    gaps, ce = [], iv[0][1]
    for s, t in iv[1:]:
        if s - ce > min_gap:
            gaps.append(f"{ce:.0f}-{s:.0f} ({s - ce:.0f})")
        ce = max(ce, t)
    print(f"  pcore {p0} idle > {min_gap:.0f} us: " + ", ".join(gaps))


if __name__ == "__main__":
    main()
