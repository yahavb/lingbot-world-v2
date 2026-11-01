"""One NEFF's executions in a Neuron runtime system profile (kiro/sysprof_rank0.sh, ingested): per execution its
duration and the collective time (cc_running spans) inside it; plus the device HBM allocations by category.
    python3 kiro/sysprof_neff.py <ingested dir> <model name substring> [n_execs]"""
import json
import sys
import pandas as pd


def main():
    d, sub = sys.argv[1], sys.argv[2]
    n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ev = pd.read_parquet(f"{d}/SystemProfileEvents.parquet")
    print("names:", ", ".join(f"{k}={v}" for k, v in ev.name.value_counts().head(20).items()))

    def model(s):
        try:
            a = json.loads(s) if s else {}
        except ValueError:
            return "?"
        return str(a.get("model_name") or a.get("model_id") or "?")
    x = ev[ev.name == "nc_exec_running"].copy()
    x["model"] = x.extra_attributes_json.map(model)
    if x.pcore_idx.notna().any():
        x = x[x.pcore_idx == sorted(x.pcore_idx.dropna().unique())[0]]
    w = x[x.model.str.contains(sub)].sort_values("start_ts").tail(n_show)
    cc = ev[ev.name == "cc_running"].copy()
    if "pcore_idx" in cc and cc.pcore_idx.notna().any():
        print("cc pcores:", sorted(cc.pcore_idx.dropna().unique())[:8])
    for _, r in w.iterrows():
        c = cc[(cc.start_ts >= r.start_ts) & (cc.end_ts <= r.end_ts)]
        tot = (c.end_ts - c.start_ts).sum() / 1e6
        # union of the collective spans (they may overlap across CC cores)
        spans = sorted(zip(c.start_ts, c.end_ts))
        u, cur_s, cur_e = 0, None, None
        for s_, e_ in spans:
            if cur_e is None or s_ > cur_e:
                if cur_e is not None:
                    u += cur_e - cur_s
                cur_s, cur_e = s_, e_
            else:
                cur_e = max(cur_e, e_)
        if cur_e is not None:
            u += cur_e - cur_s
        print(f"  exec {r.model[:40]} {r.duration_ns / 1e6:7.2f} ms: {len(c)} cc spans, summed {tot:7.2f} ms, "
              f"union {u / 1e6:7.2f} ms")
    try:
        h = pd.read_parquet(f"{d}/HbmAllocation.parquet")
        print("HbmAllocation columns:", list(h.columns)[:16])
        num = [c for c in h.columns if "size" in c.lower()]
        cat = [c for c in h.columns if any(t in c.lower() for t in ("type", "category", "usage", "name"))]
        if num and cat:
            g = h.groupby(cat[0])[num[0]].agg(["size", "sum"]).sort_values("sum", ascending=False)
            g["sum_gb"] = g["sum"] / 1e9
            print(g.head(20).to_string())
    except Exception as e:  # noqa: BLE001
        print("no HbmAllocation:", e)


if __name__ == "__main__":
    main()
