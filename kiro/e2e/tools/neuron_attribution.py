#!/usr/bin/env python3
"""System trace -> per-graph latency contribution on every rank -> labeled capture manifest.

The first stage of the Neuron attribution workflow (references/profiling.md, "From system trace
to code"). Reads one SystemProfileEvents.parquet per rank (scripts/neuron_trace_to_parquet.sh on
each <instance>_pid_<pid> dir of a runtime system trace), keeps the LAST run of each rank
(everything after its last idle gap > --run-gap-s), and reports per compiled graph (NEFF):

  executions, time per execution, total device time in the run (its latency contribution) on
  every rank, the slowest rank (the one that sets the wall time when ranks synchronize), and
  the share of that rank's run wall time.

It writes a manifest (--json-out) with one entry per graph above --min-share: the label
  rank<r>-<ms>ms-<role>-<hash8>
(r = the slowest rank, ms = its contribution, role from --roles if a graph name matches,
else "unk" until the pair is captured and its code is known; scripts/neuron_pairs.sh relabels it
from the pair's dominant source file). Contributions must add up to the device-busy time of the
run; the idle remainder is printed as its own row so nothing is hidden.

Usage:
  neuron_attribution.py rank0.parquet rank1.parquet ... [--names r0,r1,...] [--run-gap-s 0.5]
        [--min-share 0.01] [--roles 'dit=block|dit;vae=vae;t5=t5'] [--json-out manifest.json]
"""
import argparse
import collections
import json
import os
import re
import sys


def load(path):
    import duckdb
    rows = duckdb.connect().execute(
        f"SELECT start_ts, end_ts, extra_attributes_json FROM read_parquet('{path}') "
        f"WHERE name = 'nc_exec_running' ORDER BY start_ts").fetchall()
    out = []
    for a, b, j in rows:
        try:
            nm = json.loads(j).get("model_name", "?") if j else "?"
        except Exception:
            nm = "?"
        out.append((int(a), int(b), nm))
    return out


def last_run(ev, gap_ns):
    start = 0
    for i in range(1, len(ev)):
        if ev[i][0] - max(e[1] for e in ev[max(0, i - 64):i]) > gap_ns:
            start = i
    return ev[start:]


def short_hash(name):
    m = re.search(r"([0-9a-f]{8,})", name)
    return (m.group(1) if m else re.sub(r"\W+", "", name))[:8]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--names", default=None, help="comma-separated rank labels (default: 0..N-1)")
    ap.add_argument("--run-gap-s", type=float, default=0.5)
    ap.add_argument("--min-share", type=float, default=0.01, help="manifest threshold, share of the run wall")
    ap.add_argument("--roles", default="", help="'role=regex;role=regex' matched against graph names")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    names = a.names.split(",") if a.names else [str(i) for i in range(len(a.parquets))]
    roles = [(r.split("=", 1)[0], re.compile(r.split("=", 1)[1])) for r in a.roles.split(";") if "=" in r]

    per = {}          # rank -> {graph: [n, sum_ns]}
    wall, busy = {}, {}
    for rk, p in zip(names, a.parquets):
        ev = last_run(load(p), int(a.run_gap_s * 1e9))
        if not ev:
            print(f"rank {rk}: no executions in {p}", file=sys.stderr)
            continue
        g = collections.defaultdict(lambda: [0, 0])
        for s, e, nm in ev:
            g[nm][0] += 1
            g[nm][1] += e - s
        per[rk] = g
        wall[rk] = max(e for _, e, _ in ev) - ev[0][0]
        busy[rk] = sum(v[1] for v in g.values())

    graphs = sorted({nm for g in per.values() for nm in g},
                    key=lambda nm: -max(per[r].get(nm, [0, 0])[1] for r in per))
    slow = max(wall, key=wall.get)
    # Buffer the report and write the manifest first: piping this script into `head` once killed it
    # before the manifest was written (SA3, 2026-11).
    import io
    real_stdout, sys.stdout = sys.stdout, io.StringIO()
    print(f"ranks {len(per)}; run wall per rank ms: " +
          " ".join(f"{r}:{wall[r] / 1e6:.1f}" for r in per) +
          f"; slowest rank {slow} ({wall[slow] / 1e6:.1f} ms), skew {(max(wall.values()) - min(wall.values())) / 1e6:.1f} ms")
    print(f"\n{'graph':44} {'n':>6} {'ms/exec':>9} {'total ms (slowest rank)':>24} {'share':>6} {'min..max ms over ranks':>24}")
    manifest = []
    for nm in graphs:
        tots = {r: per[r].get(nm, [0, 0]) for r in per}
        worst = max(tots, key=lambda r: tots[r][1])
        n, t = tots[worst]
        share = t / wall[worst] if wall[worst] else 0
        lo = min(v[1] for v in tots.values()) / 1e6
        hi = max(v[1] for v in tots.values()) / 1e6
        print(f"{nm[-44:]:44} {n:6} {t / max(n, 1) / 1e6:9.3f} {t / 1e6:24.1f} {100 * share:5.1f}% {lo:11.1f}..{hi:.1f}")
        if share >= a.min_share:
            role = next((r for r, rx in roles if rx.search(nm)), "unk")
            manifest.append({"graph": nm, "hash": short_hash(nm), "rank": worst, "executions": n,
                             "ms_per_exec": t / max(n, 1) / 1e6, "contribution_ms": t / 1e6,
                             "share": share, "role": role,
                             "label": f"rank{worst}-{t / 1e6:.0f}ms-{role}-{short_hash(nm)}",
                             "per_rank_ms": {r: v[1] / 1e6 for r, v in tots.items()}})
    for r in (slow,):
        print(f"{'(idle: no graph running)':44} {'':6} {'':9} {(wall[r] - busy[r]) / 1e6:24.1f} "
              f"{100 * (wall[r] - busy[r]) / wall[r]:5.1f}%   rank {r}")
    print("\nmanifest (capture these pairs; neuron_pairs.sh relabels unk roles from the code):")
    for m in manifest:
        print(f"  {m['label']:40} {m['executions']} x {m['ms_per_exec']:.3f} ms")
    report, sys.stdout = sys.stdout.getvalue(), real_stdout
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump({"wall_ms": {r: v / 1e6 for r, v in wall.items()}, "busy_ms": {r: v / 1e6 for r, v in busy.items()},
                       "slowest_rank": slow, "graphs": manifest}, fh, indent=1)
        report += f"wrote {a.json_out}\n"
    try:
        sys.stdout.write(report)
        sys.stdout.flush()
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
