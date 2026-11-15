#!/usr/bin/env python3
"""Bottleneck rows (JSON) -> two markdown tables: areas with floor and realistic headroom, and
every bottleneck ranked by time. Checks that the rows add up to the measured total.

Usage: bottleneck_tables.py rows.json [--out tables.md]
Input format (all times in seconds per run; see references/report-templates.md):
{
  "total_s": 8.81,                    measured wall time of the profiled run
  "tolerance": 0.03,                  optional; allowed mismatch of the row sum (default 3%)
  "unit": "per video",                optional; label for the time column
  "areas": [ {"name": "...", "running_at": "83% of peak", "floor_s": 2.05 | [lo, hi] | "text",
              "headroom_s": 0.2 | [lo, hi] | null, "how": "..."} ],
  "rows":  [ {"item": "...", "area": "<an area name>", "s": 2.47, "s_hi": 2.9 (optional range),
              "approx": true (optional, prints ~), "detail": "...", "source": "trace|ablation|bench|derived"} ]
}
Exits 1 if the rows do not add up to total_s within the tolerance, a row names an unknown area,
or a source is not one of trace, ablation, bench, derived (comma-separated combinations allowed).
"""
import argparse
import json
import sys

SOURCES = {"trace", "ablation", "bench", "derived"}


def num(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def rng(lo, hi, unit=" s", approx=False):
    t = num(lo) if hi is None or abs(hi - lo) < 1e-9 else f"{num(lo)}-{num(hi)}"
    return ("~" if approx else "") + t + unit


def share(lo, hi, total, approx=False):
    def p(v):
        v = 100 * v / total
        return "<1" if v < 0.995 else f"{v:.0f}"
    t = p(lo) if hi is None or abs(hi - lo) < 1e-9 else f"{p(lo)}-{p(hi)}"
    return ("~" if approx else "") + t + "%"


def pair(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]), float(v[1])
    return v  # free text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.rows))
    total, tol, unit = float(d["total_s"]), float(d.get("tolerance", 0.03)), d.get("unit", "per run")
    areas = {ar["name"]: ar for ar in d["areas"]}
    errors = []
    for r in d["rows"]:
        if r["area"] not in areas:
            errors.append(f"row {r['item']!r}: unknown area {r['area']!r}")
        bad = {s.strip() for s in str(r.get("source", "")).split(",")} - SOURCES
        if bad:
            errors.append(f"row {r['item']!r}: source {sorted(bad)} not in {sorted(SOURCES)}")
    lo_sum = sum(r["s"] for r in d["rows"])
    hi_sum = sum(r.get("s_hi") or r["s"] for r in d["rows"])
    if not (lo_sum * (1 - tol) <= total <= hi_sum * (1 + tol)):
        errors.append(f"rows add up to {num(lo_sum)}-{num(hi_sum)} s but the measured total is {num(total)} s "
                      f"(tolerance {100 * tol:.0f}%): something is double counted or missing")
    if errors:
        print("\n".join("ERROR: " + e for e in errors), file=sys.stderr)
        return 1

    agg = {n: [0.0, 0.0, False] for n in areas}
    for r in d["rows"]:
        g = agg[r["area"]]
        g[0] += r["s"]
        g[1] += r.get("s_hi") or r["s"]
        g[2] = g[2] or bool(r.get("approx"))
    out = []
    out.append(f"| Area | Time now | Share | Running at | Floor | Realistic headroom | How |")
    out.append("|---|---|---|---|---|---|---|")
    f_lo = f_hi = h_lo = h_hi = 0.0
    warnings = []
    for n in sorted(areas, key=lambda n: -agg[n][0]):
        ar, (lo, hi, approx) = areas[n], agg[n]
        fl, hr = pair(ar.get("floor_s")), pair(ar.get("headroom_s"))
        if isinstance(fl, tuple):
            f_lo, f_hi = f_lo + fl[0], f_hi + fl[1]
            floor_txt = rng(*fl)
            if fl[0] > hi + 1e-9:
                warnings.append(f"area {n!r}: floor {num(fl[0])} s is above its time {num(hi)} s")
        else:
            floor_txt = fl or ""
        if isinstance(hr, tuple):
            h_lo, h_hi = h_lo + hr[0], h_hi + hr[1]
            head_txt = rng(*hr, approx=hr[0] == hr[1] and hr[0] > 0)
            if hr[1] > hi + 1e-9:
                warnings.append(f"area {n!r}: headroom {num(hr[1])} s is above its time {num(hi)} s")
        else:
            head_txt = hr or ""
        out.append(f"| {n} | {rng(lo, hi, approx=approx)} | {share(lo, hi, total, approx)} | "
                   f"{ar.get('running_at', '')} | {floor_txt} | {head_txt} | {ar.get('how', '')} |")
    out.append(f"| **Total** | **{num(total)} s** | | | **{rng(f_lo, f_hi)}** | **{rng(h_lo, h_hi)}** | "
               f"Areas without a numeric floor count as 0; headroom ranges overlap and do not add exactly |")
    out.append("")
    out.append(f"| # | Bottleneck | Time {unit} | Share | Detail | Source |")
    out.append("|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(d["rows"], key=lambda r: -r["s"]), 1):
        out.append(f"| {i} | {r['item']} | {rng(r['s'], r.get('s_hi'), approx=r.get('approx', False))} | "
                   f"{share(r['s'], r.get('s_hi'), total, r.get('approx', False))} | {r.get('detail', '')} | "
                   f"{r.get('source', '')} |")
    out.append("")
    out.append(f"Rows add up to {rng(lo_sum, hi_sum)} against {num(total)} s measured.")
    text = "\n".join(out)
    if a.out:
        open(a.out, "w").write(text + "\n")
        print(f"wrote {a.out}")
    else:
        print(text)
    for w in warnings:
        print("WARNING: " + w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
