"""Two block executions side by side (two ingested device profiles, e.g. the step and the commit block): the framework
nodes that differ in time, the DMA traffic by queue, and the busy time of each engine of pcore 0 in time bins.
    python3 kiro/prof_cmp_exec.py <session dir A> <session dir B> [bin_us]
Each side takes its median-duration execution among those with >= 90% of the largest instruction count."""
import re
import sys

import pandas as pd
import pyarrow.dataset as ds


def pick(d):
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=["execution_index", "start_ts", "end_ts"])
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    big = g[g.n > 0.9 * g.n.max()].sort_values("ms")
    ex = big.index[len(big) // 2]
    return ex, int(g.loc[ex, "t0"]), int(g.loc[ex, "t1"])


def nodes(d, ex, t0):
    fw = pd.read_parquet(f"{d}/FrameworkNode.parquet",
                         columns=["execution_index", "node_name", "start_ts", "end_ts", "hbm_read_bytes",
                                  "hbm_write_bytes"])
    f = fw[fw.execution_index == ex].copy()
    f["s"] = (f.start_ts - t0) / 1e3
    f["e"] = (f.end_ts - t0) / 1e3
    f["name"] = f.node_name.astype(str).str.replace(r"\.\d+", "", regex=True).str[:80]
    return f.sort_values("s")


def dma(d, t0, t1):
    t = ds.dataset(f"{d}/DmaPacket.parquet").to_table(
        columns=["start_ts", "end_ts", "pcore_idx", "queue_name", "transfer_bytes"],
        filter=(ds.field("start_ts") >= t0) & (ds.field("start_ts") <= t1))
    p = t.to_pandas()
    p["s"] = (p.start_ts - t0) / 1e3
    p["e"] = (p.end_ts - t0) / 1e3
    p["q"] = p.queue_name.astype(str).str.replace(r"\d+", "#", regex=True)
    return p


def engines(d, ex, t0, bin_us):
    ins = pd.read_parquet(f"{d}/Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "pcore_idx", "engine"])
    x = ins[(ins.execution_index == ex) & (ins.pcore_idx == ins.pcore_idx.min())].copy()
    x["s"] = (x.start_ts - t0) / 1e3
    x["e"] = (x.end_ts - t0) / 1e3
    span = x.e.max()
    rows = {}
    for eng, v in x.groupby("engine"):
        b = []
        for i in range(int(span // bin_us) + 1):
            lo, hi = i * bin_us, (i + 1) * bin_us
            ov = (v.e.clip(upper=hi) - v.s.clip(lower=lo)).clip(lower=0)
            b.append(min(100.0, 100.0 * ov.sum() / bin_us))
        rows[eng] = b
    return pd.DataFrame(rows), span


def main():
    a, b = sys.argv[1], sys.argv[2]
    bin_us = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
    side = {}
    for lab, d in (("A", a), ("B", b)):
        ex, t0, t1 = pick(d)
        side[lab] = (d, ex, t0, t1)
        print(f"{lab}: {d} exec {ex}: {(t1 - t0) / 1e3:.1f} us")
    print("\n=== framework nodes (A start/end -> B start/end, us), attention / collectives / copies ===")
    na, nb = nodes(side["A"][0], side["A"][1], side["A"][2]), nodes(side["B"][0], side["B"][1], side["B"][2])
    pat = re.compile(r"custom-call|all-gather|reduce-scatter|all-reduce|copy|dynamic-update|Linear|ffn|cross")
    for lab, n in (("A", na), ("B", nb)):
        sel = n[n.name.str.contains(pat)]
        print(f"--- {lab}: {len(n)} nodes, {len(sel)} selected")
        for _, r in sel.head(60).iterrows():
            print(f"  {r.s:8.1f} -> {r.e:8.1f}  rd {r.hbm_read_bytes / 1e6:7.2f} MB  wr {r.hbm_write_bytes / 1e6:7.2f} MB  "
                  f"{r['name']}")
    print("\n=== DMA by queue kind (MB, first -> last packet us) ===")
    out = {}
    for lab in ("A", "B"):
        d, ex, t0, t1 = side[lab]
        p = dma(d, t0, t1)
        g = p.groupby("q").agg(mb=("transfer_bytes", lambda s: s.sum() / 1e6), s=("s", "min"), e=("e", "max"),
                               n=("s", "size"))
        out[lab] = g
        print(f"--- {lab}: {p.transfer_bytes.sum() / 1e6:.1f} MB in {len(p)} packets")
    m = out["A"].join(out["B"], how="outer", lsuffix="_A", rsuffix="_B").fillna(0)
    m["dMB"] = m.mb_B - m.mb_A
    print(m.sort_values("dMB", key=abs, ascending=False).head(25).round(1).to_string())
    print(f"\n=== pcore-0 engine busy % per {bin_us:.0f} us bin ===")
    for lab in ("A", "B"):
        d, ex, t0, t1 = side[lab]
        e, span = engines(d, ex, t0, bin_us)
        print(f"--- {lab} (span {span:.0f} us)")
        print(e.round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
