"""Instruction count of one block execution per physical core and engine, from a neuron-explorer session dir.
    python3 kiro/prof_icount.py <ingested session dir> [exec_index|auto]
auto: the median-duration execution among those with the most common (largest) instruction count, as prof_rings.py."""
import sys
import pandas as pd


def main():
    d = sys.argv[1]
    cols = pd.read_parquet(f"{d}/Instruction.parquet").columns
    eng = next((c for c in ("engine", "engine_name", "queue_name", "engine_idx") if c in cols), None)
    use = ["execution_index", "start_ts", "end_ts", "pcore_idx"] + ([eng] if eng else [])
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=use)
    g = ins.groupby("execution_index").agg(t0=("start_ts", "min"), t1=("end_ts", "max"), n=("start_ts", "size"))
    g["ms"] = (g.t1 - g.t0) / 1e6
    if len(sys.argv) > 2 and sys.argv[2] != "auto":
        ex = int(sys.argv[2])
    else:
        big = g[g.n > 0.9 * g.n.max()].sort_values("ms")
        ex = big.index[len(big) // 2]
    x = ins[ins.execution_index == ex]
    print(f"exec {ex}: {g.loc[ex, 'ms']:.3f} ms, {len(x)} instructions; columns: {list(cols)[:20]}")
    keys = ["pcore_idx"] + ([eng] if eng else [])
    print(x.groupby(keys).size().to_string())


if __name__ == "__main__":
    main()
