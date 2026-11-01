"""Executions in a neuron-explorer device profile: index, duration, instruction count, and the engine busy time of
the chosen one (by default every execution that is not the largest-count kind, i.e. not the DiT block).
    python3 kiro/prof_execs.py <ingested session dir> [exec_index]"""
import sys
import pandas as pd


def main():
    d = sys.argv[1]
    ins = pd.read_parquet(f"{d}/Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "pcore_idx", "engine", "opcode"])
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    print(g[["ms", "n"]].to_string())
    ex = int(sys.argv[2]) if len(sys.argv) > 2 else int(g[g.n < 0.9 * g.n.max()].sort_values("ms").index[-1])
    x = ins[(ins.execution_index == ex) & (ins.pcore_idx == ins.pcore_idx.min())].copy()
    x["dur"] = (x.end_ts - x.start_ts) / 1e3
    span = (x.end_ts.max() - x.start_ts.min()) / 1e3
    print(f"\nexec {ex}: {span:.1f} us on pcore {ins.pcore_idx.min()}")
    by = x.groupby("engine").dur.agg(["size", "sum"]).sort_values("sum", ascending=False)
    by["busy%"] = 100 * by["sum"] / span
    print(by.to_string())
    top = x.groupby(["engine", "opcode"]).dur.agg(["size", "sum"]).sort_values("sum", ascending=False).head(15)
    print(top.to_string())


if __name__ == "__main__":
    main()
