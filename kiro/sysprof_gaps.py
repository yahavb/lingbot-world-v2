"""Device timeline of the last generation in a Neuron runtime system profile (one rank, kiro/sysprof_rank0.sh).
    python3 kiro/sysprof_gaps.py <dir with SystemProfileEvents.parquet> [gen_ms] [min_gap_us]
Takes the nc_exec_running spans (one per NEFF execution on the core) that start in the last gen_ms (default 3100)
before the last one ends, then prints, per NEFF: executions, summed and median device time; the idle time between
consecutive executions, by the pair of NEFFs around it; and the gaps longer than min_gap_us (default 200) in order.
NEFFs are labelled by their execution count and median duration, since their names are content hashes.
"""
import json
import sys

import pandas as pd


def main():
    d = sys.argv[1]
    gen_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 3100.0
    min_gap_us = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
    ev = pd.read_parquet(f"{d}/SystemProfileEvents.parquet")
    print("events:", len(ev), "| names:", ", ".join(f"{k}={v}" for k, v in ev.name.value_counts().head(12).items()))
    x = ev[ev.name == "nc_exec_running"].copy()

    def model(s):
        try:
            a = json.loads(s) if s else {}
        except ValueError:
            return "?"
        return str(a.get("model_name") or a.get("model_id") or "?")

    x["model"] = x.extra_attributes_json.map(model)
    if x.pcore_idx.notna().any():
        # one span per physical core of the logical core (LNC=2): keep the first core's
        pc = sorted(x.pcore_idx.dropna().unique())
        print("pcores:", pc, "-> using", pc[0])
        x = x[x.pcore_idx == pc[0]]
    x = x.sort_values("start_ts")
    # the generation ends where the executions stop being dense: drop the trailing ones (latent save, barriers)
    # that follow a gap of more than 300 ms
    while True:
        st_, en_ = x.start_ts.to_numpy(), x.end_ts.to_numpy()
        big = [i for i in range(max(1, len(x) - 20), len(x)) if st_[i] - en_[i - 1] > 300e6]
        if not big:
            break
        x = x.iloc[:big[0]]
    t_end = x.end_ts.max()
    w = x[x.start_ts >= t_end - gen_ms * 1e6].copy()
    span = (w.end_ts.max() - w.start_ts.min()) / 1e6
    busy = w.duration_ns.sum() / 1e6
    print(f"\nwindow: {len(w)} executions over {span:.1f} ms, device busy {busy:.1f} ms, idle {span - busy:.1f} ms")
    st = w.groupby("model").duration_ns.agg(["size", "sum", "median"]).sort_values("sum", ascending=False)
    lab = {m: f"n{int(r['size'])}_med{r['median'] / 1e3:.0f}us" for m, r in st.iterrows()}
    w["lab"] = w.model.map(lab)
    print("\nper NEFF (executions, summed ms, median us):")
    for m, r in st.iterrows():
        print(f"  {int(r['size']):5d}  {r['sum'] / 1e6:8.2f}  {r['median'] / 1e3:8.1f}  {m[:60]}")
    s = w.start_ts.to_numpy()
    e = w.end_ts.to_numpy()
    labs = w.lab.to_numpy()
    gaps = pd.DataFrame({"gap_us": (s[1:] - e[:-1]) / 1e3, "prev": labs[:-1], "next": labs[1:],
                         "at_ms": (s[1:] - s[0]) / 1e6})
    print(f"\nidle between executions: total {gaps.gap_us.clip(lower=0).sum() / 1e3:.1f} ms, "
          f"median {gaps.gap_us.median():.1f} us, p90 {gaps.gap_us.quantile(.9):.1f} us, "
          f"{int((gaps.gap_us < 0).sum())} overlaps")
    pair = gaps.groupby(["prev", "next"]).gap_us.agg(["size", "sum", "median"]).sort_values("sum", ascending=False)
    print("\nidle by (previous NEFF -> next NEFF): count, summed ms, median us")
    for (p, n), r in pair.head(25).iterrows():
        print(f"  {int(r['size']):5d}  {r['sum'] / 1e3:8.2f}  {r['median']:8.1f}  {p} -> {n}")
    big = gaps[gaps.gap_us > min_gap_us]
    print(f"\ngaps > {min_gap_us:.0f} us: {len(big)}, summed {big.gap_us.sum() / 1e3:.1f} ms")
    for _, r in big.head(60).iterrows():
        print(f"  at {r.at_ms:8.2f} ms  {r.gap_us:8.1f} us  {r.prev} -> {r.next}")
    other = ev[(ev.start_ts >= w.start_ts.min()) & (ev.name != "nc_exec_running")]
    print("\nother runtime events in the window (count, summed ms):")
    for n_, v in other.groupby("name").duration_ns.agg(["size", "sum"]).sort_values("sum", ascending=False).head(15).iterrows():
        print(f"  {int(v['size']):6d}  {v['sum'] / 1e6:9.2f}  {n_}")


if __name__ == "__main__":
    main()
