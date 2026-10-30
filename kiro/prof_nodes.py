"""Framework-node (module + HLO op) timeline of one execution, in start order.
    python3 kiro/prof_nodes.py <profile dir (..._session_0@latest)> <exec_index> [min_us]
Maps the optimized-HLO instruction names (dot.2, all-gather.237, ...) to the model modules that
produced them, so a wait in prof_cc.py / prof_crit.py can be attributed to a tensor."""
import sys

import pandas as pd


def main():
    d, ex = sys.argv[1], int(sys.argv[2])
    min_us = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    fw = pd.read_parquet(f"{d}/FrameworkNode.parquet")
    print("FrameworkNode columns:", list(fw.columns))
    f = fw[fw.execution_index == ex].copy()
    t0 = f.start_ts.min()
    f["s_us"] = (f.start_ts - t0) / 1e3
    f["span_us"] = (f.end_ts - f.start_ts) / 1e3
    f = f[f.span_us >= min_us].sort_values("start_ts")
    cols = [c for c in ("hlo_name", "hlo_op", "op_name", "opcode") if c in f.columns]
    for _, r in f.iterrows():
        extra = " ".join(str(r[c])[:40] for c in cols)
        print(f"  {r.s_us:8.1f} {r.span_us:8.1f}  {str(r.node_name)[:80]:80s} {extra}")


if __name__ == "__main__":
    main()
