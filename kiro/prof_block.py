"""Breakdown of one compiled-block execution from a neuron-explorer session profile.

python3 kiro/prof_block.py <profile dir (…_session_0@latest)> [exec_index]

Lists every execution (wall time, instruction count), picks the median-duration execution of the
most frequent instruction count (the timed REPS of the steady-state block) unless one is given,
then prints per-engine busy time (merged intervals), the top HLO ops per engine, collectives,
and the top framework nodes for it.
"""
import sys

import numpy as np
import pandas as pd


def merged_ns(df):
    """Wall time covered by the union of [start_ts, end_ts) intervals."""
    if df.empty:
        return 0
    iv = df[["start_ts", "end_ts"]].sort_values("start_ts").to_numpy()
    tot, cs, ce = 0, iv[0, 0], iv[0, 1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def main():
    d = sys.argv[1]
    cols = ["execution_index", "start_ts", "end_ts", "duration_ns", "engine", "opcode", "hlo_name",
            "hbm_read_bytes", "hbm_write_bytes", "spill_save_bytes", "spill_reload_bytes"]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=cols)
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"),
                                           n=("opcode", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    print("executions (index: ms, instructions):")
    print(" ".join(f"{i}:{r.ms:.2f}/{r.n}" for i, r in g.iterrows()))
    if len(sys.argv) > 2:
        ex = int(sys.argv[2])
    else:
        n_mode = g.n.mode().iloc[0]
        cand = g[g.n == n_mode].sort_values("ms")
        ex = cand.index[len(cand) // 2]
    e = ins[ins.execution_index == ex]
    span = (e.end_ts.max() - e.start_ts.min()) / 1e6
    print(f"\n=== execution {ex}: {span:.3f} ms, {len(e)} instructions")
    print("engine busy (merged) ms:", ", ".join(
        f"{k}={merged_ns(v) / 1e6:.2f}" for k, v in e.groupby("engine")))
    for k in ["hbm_read_bytes", "hbm_write_bytes", "spill_save_bytes", "spill_reload_bytes"]:
        print(f"  {k}: {e[k].fillna(0).sum() / 1e6:.1f} MB")
    e = e.assign(op=e.hlo_name.fillna("?").str.replace(r"\.\d+$", "", regex=True))
    for eng, v in e.groupby("engine"):
        top = v.groupby("op").agg(ms=("duration_ns", lambda x: x.sum() / 1e6), n=("op", "size"),
                                  spill=("spill_save_bytes", lambda x: x.fillna(0).sum() / 1e6))
        top = top.sort_values("ms", ascending=False).head(8)
        print(f"\n[{eng}] top HLO ops by summed instruction time (ms, count, spill MB):")
        for op, r in top.iterrows():
            print(f"  {r.ms:7.2f}  {int(r.n):6d}  {r.spill:7.1f}  {op[:110]}")
    try:
        cc = pd.read_parquet(f"{d}/CcInstruction.parquet")
        c = cc[cc.execution_index == ex]
        print(f"\ncollectives in exec {ex}: {len(c)} instr, sum {c.duration_ns.sum() / 1e6:.2f} ms, "
              f"merged {merged_ns(c) / 1e6:.2f} ms")
        for desc, v in c.groupby("description"):
            print(f"  {v.duration_ns.sum() / 1e6:7.2f} ms  x{len(v):4d}  {str(desc)[:100]}")
    except Exception as ex_:  # noqa: BLE001
        print("no CcInstruction:", ex_)
    try:
        fw = pd.read_parquet(f"{d}/FrameworkNode.parquet",
                             columns=["execution_index", "node_name", "start_ts", "end_ts", "mfu"])
        f = fw[fw.execution_index == ex].copy()
        f["ms"] = (f.end_ts - f.start_ts) / 1e6
        print(f"\nframework nodes in exec {ex} (top 25 by span):")
        for _, r in f.sort_values("ms", ascending=False).head(25).iterrows():
            print(f"  {r.ms:7.2f} ms  mfu={r.mfu if pd.notna(r.mfu) else float('nan'):.3f}  {str(r.node_name)[:110]}")
    except Exception as ex_:  # noqa: BLE001
        print("no FrameworkNode:", ex_)


if __name__ == "__main__" and not (len(sys.argv) > 3 and sys.argv[3] == "timeline"):
    main()


def timeline(d, ex, min_ms=0.1):
    """Per-HLO-op wall span inside one execution, in start order: offset, span, merged busy per engine."""
    cols = ["execution_index", "start_ts", "end_ts", "duration_ns", "engine", "pcore_idx", "hlo_name"]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=cols)
    e = ins[ins.execution_index == ex].copy()
    t0 = e.start_ts.min()
    e["op"] = e.hlo_name.fillna("<none>").str.split(" = ").str[0]
    rows = []
    for op, v in e.groupby("op"):
        if v.duration_ns.sum() / 1e6 < min_ms:
            continue
        busy = {k: merged_ns(w) / 1e6 for k, w in v.groupby("engine")}
        rows.append((v.start_ts.min() - t0, v.end_ts.max() - v.start_ts.min(), op, busy,
                     sorted(v.pcore_idx.dropna().unique().tolist())))
    print(f"\n=== timeline of exec {ex} (ops with >= {min_ms} ms summed instruction time)")
    print("   start    span  op                      merged busy per engine (ms)          pcores")
    for s, span, op, busy, pc in sorted(rows):
        b = " ".join(f"{k[:3]}={v:.2f}" for k, v in sorted(busy.items()))
        print(f"  {s / 1e6:6.2f}  {span / 1e6:6.2f}  {op[:22]:22s}  {b:38s} {pc}")


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "timeline":
    timeline(sys.argv[1], int(sys.argv[2]))
