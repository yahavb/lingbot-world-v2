"""When the new chunk's K / V are ready and when their CP all-gathers run, in one block execution of a device profile:
the framework nodes of the K / V projections, their norms / RoPE, the all-gathers and the attention custom calls
(first and last instruction, us from the execution start), and the CC cores' ring spans.
    python3 kiro/prof_vgather.py <ingested session dir> [exec_index|auto]"""
import re
import sys
import pandas as pd


def main():
    d = sys.argv[1]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=["execution_index", "start_ts", "end_ts"])
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    if len(sys.argv) > 2 and sys.argv[2] != "auto":
        ex = int(sys.argv[2])
    else:
        big = g[g.n > 0.9 * g.n.max()].sort_values("ms")
        ex = big.index[len(big) // 2]
    t0 = g.loc[ex, "t0"]
    print(f"exec {ex}: {g.loc[ex, 'ms']:.3f} ms")
    fw = pd.read_parquet(f"{d}/FrameworkNode.parquet")
    f = fw[fw.execution_index == ex].copy()
    f["s"] = (f.start_ts - t0) / 1e3
    f["e"] = (f.end_ts - t0) / 1e3
    pat = re.compile(r"Linear\[(q|k|v)\]|all-gather|custom-call|rsqrt|_all-reduce|reduce-scatter")
    f = f[f.node_name.astype(str).str.contains(pat)].sort_values("s")
    for _, r in f.head(40).iterrows():
        print(f"  {r.s:8.1f} -> {r.e:8.1f}  {str(r.node_name)[:90]}")
    cc = pd.read_parquet(f"{d}/CcInstruction.parquet")
    c = cc[cc.execution_index == ex].copy()
    c["s"] = (c.start_ts - t0) / 1e3
    x = c[c.description.str.match(r"ALGO_(MESH|RING)_(BEGIN|END)$")].sort_values("s")
    for core, v in x.groupby("core_idx"):
        print(f"  cc core {core}: " + ", ".join(f"{r.description.split('_')[1][0]}{r.description.split('_')[2][0]} {r.s:.0f}"
                                                 for _, r in v.iterrows())[:400])


if __name__ == "__main__":
    main()
