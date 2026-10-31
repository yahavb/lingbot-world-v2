"""DMA traffic per HLO op in one execution (HBM bytes read/written by its instructions, and the
window in which they ran), plus the consumers of chosen operands.
    python3 kiro/prof_dma.py <profile dir> <exec_index> [operand ...]
e.g. operand %Arg_0.1 lists every HLO instruction whose text uses it."""
import sys

import pandas as pd


def main():
    d, ex, opers = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    cols = ["execution_index", "start_ts", "end_ts", "engine", "pcore_idx", "hlo_name",
            "hbm_read_bytes", "hbm_write_bytes"]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=cols)
    e = ins[ins.execution_index == ex].copy()
    t0 = e.start_ts.min()
    e["op"] = e.hlo_name.fillna("<none>").str.split(" = ").str[0]
    e["rd"] = e.hbm_read_bytes.fillna(0)
    e["wr"] = e.hbm_write_bytes.fillna(0)
    g = e.groupby("op").agg(rd=("rd", "sum"), wr=("wr", "sum"), s=("start_ts", "min"),
                            t=("end_ts", "max"), n=("op", "size"))
    g = g[(g.rd + g.wr) > 2e5].sort_values("s")
    print(f"exec {ex}: HBM read {e.rd.sum() / 1e6:.1f} MB, write {e.wr.sum() / 1e6:.1f} MB")
    print("  start_us    end_us   read_MB  write_MB      n  op")
    for op, r in g.iterrows():
        print(f"  {(r.s - t0) / 1e3:8.1f}  {(r.t - t0) / 1e3:8.1f}  {r.rd / 1e6:8.2f}  {r.wr / 1e6:8.2f}"
              f"  {int(r.n):5d}  {op[:70]}")
    names = e.hlo_name.dropna().unique()
    for o in opers:
        print(f"\nHLO instructions using {o}:")
        for h in names:
            if o + "," in h or o + ")" in h:
                print("   ", h[:200])


if __name__ == "__main__":
    main()
