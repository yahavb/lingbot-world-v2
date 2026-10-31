"""Collective timeline of one execution: each collective operation's begin/end (CC core events)
against the compute of the same execution, so waits on the critical path can be attributed.
    python3 kiro/prof_cc.py <profile dir (..._session_0@latest)> <exec_index>
Prints the CcInstruction schema, then per collective algorithm (ALGO_*_BEGIN .. ALGO_*_END pairs)
its offset from the execution start and duration, and the busy fraction of every engine of pcore
0 in 50 us bins."""
import sys

import pandas as pd


def main():
    d, ex = sys.argv[1], int(sys.argv[2])
    ins = pd.read_parquet(f"{d}/Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "engine", "pcore_idx",
                                   "hlo_name"])
    e = ins[ins.execution_index == ex]
    t0, t1 = e.start_ts.min(), e.end_ts.max()
    print(f"exec {ex}: {(t1 - t0) / 1e3:.1f} us")
    cc = pd.read_parquet(f"{d}/CcInstruction.parquet")
    print("CcInstruction columns:", list(cc.columns))
    c = cc[cc.execution_index == ex].sort_values("start_ts")
    print(f"{len(c)} cc instructions")
    # pair ALGO_*_BEGIN / _END events in order, per (pcore if present) stream
    key = [k for k in ("pcore_idx", "cc_core_idx", "channel_id", "queue_name") if k in c.columns]
    print("grouping by", key)
    begins = {}
    rows = []
    for _, r in c.iterrows():
        desc = str(r.get("description", ""))
        if not desc.startswith("ALGO_") or "EVENT" in desc or "_CH_" in desc:
            continue
        base = desc.rsplit("_", 1)[0]
        k = tuple(r[k_] for k_ in key) + (base,)
        if desc.endswith("_BEGIN"):
            begins.setdefault(k, []).append(r.start_ts)
        elif desc.endswith("_END") and begins.get(k):
            b = begins[k].pop(0)
            rows.append((b - t0, r.end_ts - b, base, k[:-1]))
    print("\n  start_us   dur_us  algo  stream")
    for s, dur, base, k in sorted(rows):
        print(f"  {s / 1e3:8.1f} {dur / 1e3:8.1f}  {base:14s} {k}")
    # engine busy per 50 us bin on pcore 0
    p0 = e[e.pcore_idx == e.pcore_idx.min()]
    nb = int((t1 - t0) // 50_000) + 1
    print("\nbusy % per 50 us bin, pcore", int(p0.pcore_idx.min()))
    engines = sorted(p0.engine.dropna().unique())
    print("   bin_us " + " ".join(f"{x[:6]:>6s}" for x in engines) + "  top op")
    for b in range(nb):
        lo, hi = t0 + b * 50_000, t0 + (b + 1) * 50_000
        w = p0[(p0.end_ts > lo) & (p0.start_ts < hi)]
        vals = []
        for x in engines:
            ww = w[w.engine == x]
            iv = sorted(zip(ww.start_ts.clip(lower=lo), ww.end_ts.clip(upper=hi)))
            tot, cs, ce = 0, None, None
            for s_, e_ in iv:
                if cs is None:
                    cs, ce = s_, e_
                elif s_ > ce:
                    tot += ce - cs
                    cs, ce = s_, e_
                else:
                    ce = max(ce, e_)
            if cs is not None:
                tot += ce - cs
            vals.append(100.0 * tot / 50_000)
        top = w.hlo_name.fillna("?").str.split(" = ").str[0].value_counts()
        top_s = ",".join(f"{k}" for k in top.index[:3])
        print(f"  {b * 50:6d}  " + " ".join(f"{v:6.0f}" for v in vals) + f"  {top_s[:60]}")


if __name__ == "__main__":
    main()
