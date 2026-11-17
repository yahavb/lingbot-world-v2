"""HLO-level spans of one block execution in an ingested device profile: per HLO op (hlo_name), the first and last
instruction on pcore 0 and their engines, plus the DMA MB per 100 us bin of the execution.
    python3 kiro/prof_hlo_spans.py <session dir> [regex] [exec_index|auto]"""
import re
import sys

import pandas as pd
import pyarrow.dataset as ds


def main():
    d = sys.argv[1]
    pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"custom-call|all-gather|copy|dynamic-update|reduce-scatter")
    ins = pd.read_parquet(f"{d}/Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "pcore_idx", "engine", "opcode",
                                   "hlo_name", "hbm_read_bytes", "hbm_write_bytes", "dma_wait_time_ns",
                                   "evt_wait_time_ns"])
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    if len(sys.argv) > 3 and sys.argv[3] != "auto":
        ex = int(sys.argv[3])
    else:
        big = g[g.n > 0.9 * g.n.max()].sort_values("ms")
        ex = big.index[len(big) // 2]
    t0, t1 = int(g.loc[ex, "t0"]), int(g.loc[ex, "t1"])
    print(f"exec {ex}: {(t1 - t0) / 1e3:.1f} us")
    x = ins[(ins.execution_index == ex) & (ins.pcore_idx == ins.pcore_idx.min())].copy()
    x["s"] = (x.start_ts - t0) / 1e3
    x["e"] = (x.end_ts - t0) / 1e3
    x["h"] = x.hlo_name.astype(str)
    sel = x[x.h.str.contains(pat)]
    agg = sel.groupby("h").agg(s=("s", "min"), e=("e", "max"), n=("s", "size"),
                               engines=("engine", lambda v: ",".join(sorted(set(map(str, v))))),
                               rd=("hbm_read_bytes", "sum"), wr=("hbm_write_bytes", "sum"),
                               evw=("evt_wait_time_ns", "sum")).sort_values("s")
    for h, r in agg.iterrows():
        print(f"  {r.s:8.1f} -> {r.e:8.1f}  n={r.n:6d}  rd {r.rd / 1e6:7.2f} MB  wr {r.wr / 1e6:7.2f} MB  "
              f"wait {r.evw / 1e3:8.1f} us  [{r.engines}]  {h[:90]}")
    # the longest waits: semaphore waits > 20 us on pcore 0
    w = x[x.evt_wait_time_ns > 20000].sort_values("s")
    print(f"\n  waits > 20 us on pcore 0: {len(w)}")
    for _, r in w.head(30).iterrows():
        print(f"  {r.s:8.1f} -> {r.e:8.1f}  {r.engine:7} {str(r.opcode):18} wait {r.evt_wait_time_ns / 1e3:7.1f} us  "
              f"{r.h[:80]}")
    t = ds.dataset(f"{d}/DmaPacket.parquet").to_table(
        columns=["start_ts", "pcore_idx", "queue_name", "transfer_bytes"],
        filter=(ds.field("start_ts") >= t0) & (ds.field("start_ts") <= t1)).to_pandas()
    t["b"] = ((t.start_ts - t0) / 1e5).astype(int)
    t["q"] = t.queue_name.astype(str).str.replace(r"\d+", "#", regex=True)
    tab = t.pivot_table(index="b", columns="q", values="transfer_bytes", aggfunc="sum").fillna(0) / 1e6
    print("\n  DMA MB per 100 us bin (all pcores):")
    print(tab.round(1).to_string())


if __name__ == "__main__":
    main()
