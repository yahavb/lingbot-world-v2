#!/usr/bin/env python3
"""Per-rank Neuron system traces -> one Perfetto/Chrome trace JSON, one track per rank.

  ranks_to_perfetto.py r0.parquet r1.parquet ... --names 0,1,... --out trace.json
                       [--align-index K | --boundary-index -1 --min-gap-ms 300] [--pad-s 0.3] [--span-s 1.5]

Each input is a SystemProfileEvents.parquet (neuron_trace_to_parquet.sh). Only device
executions (nc_exec_running) are kept. Ranks have separate clocks, so each
rank is aligned on the end of its DiT-stack execution --align-index (a pipe() call boundary
that every rank reaches together), and the window runs --pad-s before to --span-s after it.

Graph names are NEFF hashes; roles are inferred and written into the event name:
  "DiT stack"  the most frequent graph whose mean execution exceeds 50 ms
  "decoder"    graphs that run on some ranks but not on the anchor rank's peers that never
               decode (present on fewer ranks than the DiT stack), mean > 5 ms
  everything else keeps "neff_<hash8>".
"""
import argparse
import collections
import json

import duckdb


def load(path):
    rows = duckdb.connect().execute(
        f"SELECT start_ts, end_ts, extra_attributes_json FROM read_parquet('{path}') "
        "WHERE name = 'nc_exec_running' ORDER BY start_ts").fetchall()
    out, prev_end = [], None
    for a, b, j in rows:
        name = json.loads(j).get("model_name", "?") if j else "?"
        a, b = int(a), int(b)
        # Executions are queued ahead (async), so an interval can start before the previous
        # one ended. The core runs one at a time: clip the start to the previous end.
        if prev_end is not None and a < prev_end:
            a = min(prev_end, b)
        out.append((a, b, name))
        prev_end = max(prev_end or b, b)
    return out


def short(name):
    h = name.split("neff_")[-1]
    return "neff_" + h[:8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--names", required=True, help="comma-separated rank labels, one per parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--align-index", type=int, default=None,
                    help="DiT-stack execution whose end is t=0 on every rank (default: the last "
                         "forward before the --boundary-index call boundary of the first rank)")
    ap.add_argument("--boundary-index", type=int, default=-1)
    ap.add_argument("--min-gap-ms", type=float, default=300.0)
    ap.add_argument("--pad-s", type=float, default=0.3)
    ap.add_argument("--span-s", type=float, default=1.5)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    names = a.names.split(",")
    assert len(names) == len(a.parquets), "one --names label per parquet"
    ex = {n: load(p) for n, p in zip(names, a.parquets)}

    # Roles.
    stats = collections.defaultdict(lambda: [0, 0, set()])  # name -> [count, total_ns, ranks]
    for r, evs in ex.items():
        for s, e, nm in evs:
            st = stats[nm]
            st[0] += 1
            st[1] += e - s
            st[2].add(r)
    big = [(st[0], nm) for nm, st in stats.items() if st[1] / st[0] > 50e6]
    stack = max(big)[1] if big else None
    n_stack_ranks = len(stats[stack][2]) if stack else len(ex)
    role = {}
    for nm, st in stats.items():
        mean_ms = st[1] / st[0] / 1e6
        if nm == stack:
            role[nm] = "DiT stack"
        elif len(st[2]) < n_stack_ranks and mean_ms > 5:
            role[nm] = "decoder"
        else:
            role[nm] = short(nm)

    # Window. Each rank's trace has its own clock, so align every rank on the same logical
    # event: the END of its DiT-stack execution number --align-index. All ranks run that forward together (its collectives couple them), so t=0 is the call
    # boundary on every track; the window is [-pad, +span].
    # The index is found on the reference rank (the first one given; a decoding rank) as
    # the DiT forward just before its --decode-index decoder execution.
    if a.align_index is None:
        # The boundary: on the reference rank, the --boundary-index-th DiT-stack gap longer
        # than --min-gap-ms (pipe() call boundaries; the run's end has no following forward,
        # so -1 is the last boundary INSIDE the run). Counted from the end: ranks may start
        # tracing at different points, they end together.
        ref = names[0]
        st_ref = [x for x in ex[ref] if x[2] == stack]
        big = [i for i in range(len(st_ref) - 1)
               if st_ref[i + 1][0] - st_ref[i][1] > a.min_gap_ms * 1e6]
        if not big:
            raise SystemExit(f"rank {ref}: no DiT-stack gap > {a.min_gap_ms} ms")
        a.align_index = big[a.boundary_index] - len(st_ref)
    offs = {}
    for r, evs in ex.items():
        st = [x for x in evs if x[2] == stack]
        offs[r] = st[a.align_index][1]
    w0, w1 = -int(a.pad_s * 1e9), int(a.span_s * 1e9)
    out = [{"name": "process_name", "ph": "M", "pid": 1, "args": {"name": a.title or "Neuron device executions"}}]
    for i, r in enumerate(names):
        out.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": i + 1, "args": {"name": f"rank {r}"}})
        out.append({"name": "thread_sort_index", "ph": "M", "pid": 1, "tid": i + 1, "args": {"sort_index": i}})
        for s, e, nm in ex[r]:
            s, e = s - offs[r], e - offs[r]
            if e < w0 or s > w1:
                continue
            out.append({"name": role[nm], "ph": "X", "pid": 1, "tid": i + 1,
                        "ts": (max(s, w0) - w0) / 1e3, "dur": (min(e, w1) - max(s, w0)) / 1e3,
                        "args": {"neff": nm, "exec_ms": round((e - s) / 1e6, 3)}})
    json.dump({"traceEvents": out, "displayTimeUnit": "ms"}, open(a.out, "w"))

    # Text summary of the window, per rank: busy ms by role.
    print(f"window {(w1 - w0) / 1e6:.0f} ms: {a.pad_s * 1e3:.0f} ms before to {a.span_s * 1e3:.0f} ms after "
          f"the end of DiT forward #{a.align_index} (a call boundary) on every rank")
    for r in names:
        by = collections.Counter()
        for s, e, nm in ex[r]:
            s, e = s - offs[r], e - offs[r]
            if e < w0 or s > w1:
                continue
            by[role[nm]] += min(e, w1) - max(s, w0)
        busy = sum(by.values())
        parts = ", ".join(f"{k} {v / 1e6:.0f}" for k, v in by.most_common(3))
        print(f"  rank {r:>3}: busy {busy / 1e6:7.1f} ms, idle {(w1 - w0 - busy) / 1e6:7.1f} ms | {parts}")
    print(f"wrote {a.out} ({len(out)} events)")


if __name__ == "__main__":
    main()
