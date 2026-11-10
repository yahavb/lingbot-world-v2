#!/usr/bin/env python3
"""Device-busy share of wall time per component (DiT / T5 / VAE / ...) from one rank's system trace.

  component_split.py SystemProfileEvents.parquet --roles '{"<neff-hash8>":"dit", ...}'
                     --anchor <hash8-of-the-DiT-stack> [--last 60]

Window = the last --last executions of the --anchor graph (one per DiT forward), extended to the
first execution of a "vae" graph after it. Each graph's executions are clipped so queued intervals
do not overlap (a NeuronCore runs one graph at a time). Prints busy seconds and share per role, and
idle. Use it to check what the bench's own phase timers claim: with the per-forward sync off,
a "non-DiT" timer can absorb DiT device time (measured: 5.7 s "non-DiT" on o2 that was <3% on device).
Graph hashes come from neuron_attribution.py. Needs duckdb.
"""
import argparse, collections, json
import duckdb

def load(p):
    rows = duckdb.connect().execute(
        f"SELECT start_ts,end_ts,extra_attributes_json FROM read_parquet('{p}') "
        "WHERE name='nc_exec_running' ORDER BY start_ts").fetchall()
    out, pe = [], None
    for a, b, j in rows:
        a, b = int(a), int(b)
        if pe is not None:
            a = min(max(a, pe), b)
        out.append((a, b, json.loads(j).get("model_name", "?") if j else "?"))
        pe = b if pe is None else max(pe, b)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet"); ap.add_argument("--roles", required=True)
    ap.add_argument("--anchor", required=True); ap.add_argument("--last", type=int, default=60)
    a = ap.parse_args()
    roles = json.loads(a.roles); ev = load(a.parquet)
    key = lambda n: next((r for h, r in roles.items() if h in n), "other")
    S = [x for x in ev if a.anchor in x[2]][-a.last:]
    t0, t1 = S[0][0], S[-1][1]
    vae = [x for x in ev if key(x[2]) == "vae" and x[0] >= t1]
    if vae: t1 = vae[0][1]
    busy = collections.Counter()
    for s, e, n in ev:
        s, e = max(s, t0), min(e, t1)
        if e > s: busy[key(n)] += e - s
    wall = t1 - t0; tot = sum(busy.values())
    print("window %.2f s (last %d %s executions + next decode)" % (wall / 1e9, a.last, a.anchor))
    for k, v in busy.most_common(): print("  %-10s %7.2f s  %5.1f%%" % (k, v / 1e9, 100 * v / wall))
    print("  %-10s %7.2f s  %5.1f%%" % ("idle", (wall - tot) / 1e9, 100 * (wall - tot) / wall))

if __name__ == "__main__":
    main()
