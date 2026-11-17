"""Device memory of one rank by usage type and by NEFF, from a runtime system profile's HbmAllocation table
(kiro/sysprof_rank0.sh, ingested): the allocations alive at time t (default: the end of the profile).
    python3 kiro/sysprof_hbm.py <ingested dir> [t_ns]"""
import sys
import pandas as pd


def main():
    d = sys.argv[1]
    h = pd.read_parquet(f"{d}/HbmAllocation.parquet")
    h["bytes"] = (h.end_address - h.start_address).astype("int64")
    t = int(sys.argv[2]) if len(sys.argv) > 2 else int(h.start_ts.max())
    alive = h[(h.start_ts <= t) & ((h.end_ts.isna()) | (h.end_ts <= 0) | (h.end_ts > t))]
    pc = sorted(alive.pcore_idx.dropna().unique())
    if pc:
        alive = alive[alive.pcore_idx == pc[0]]
    print(f"allocations alive: {len(alive)} on pcore {pc[:1]}, {alive.bytes.sum() / 1e9:.2f} GB")
    by_type = alive.groupby("usage_type").bytes.agg(["size", "sum"]).sort_values("sum", ascending=False)
    by_type["GB"] = by_type["sum"] / 1e9
    print(by_type[["size", "GB"]].to_string())
    by_neff = alive.groupby(["neff_name", "usage_type"]).bytes.sum().unstack(fill_value=0) / 1e6
    by_neff["total_MB"] = by_neff.sum(axis=1)
    print(by_neff.sort_values("total_MB", ascending=False).head(30).round(1).to_string())


if __name__ == "__main__":
    main()
