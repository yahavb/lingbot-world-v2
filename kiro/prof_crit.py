"""Per-physical-core op sequence of one execution: where each core is busy, and the idle gaps.
    python3 kiro/prof_crit.py <profile dir (..._session_0@latest)> <exec_index> [min_gap_us]
For every HLO op (instructions grouped by hlo_name), per pcore: first start, last end, and merged
busy time per engine. Then the idle gaps on each pcore (no instruction on any engine) longer than
min_gap_us (default 10). The ops are listed in start order, so the critical path reads top-down.
"""
import sys

import pandas as pd


def merged(iv):
    if len(iv) == 0:
        return 0
    iv = sorted(iv)
    t, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            t += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return t + ce - cs


def gaps(iv, min_ns):
    iv = sorted(iv)
    out, ce = [], iv[0][1]
    for s, e in iv[1:]:
        if s - ce > min_ns:
            out.append((ce, s))
        ce = max(ce, e)
    return out


def main():
    d, ex = sys.argv[1], int(sys.argv[2])
    min_gap = float(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 10_000
    cols = ["execution_index", "start_ts", "end_ts", "engine", "hlo_name", "pcore_idx"]
    ins = pd.read_parquet(f"{d}/Instruction.parquet", columns=cols)
    e = ins[ins.execution_index == ex].copy()
    t0 = e.start_ts.min()
    e["s"] = (e.start_ts - t0) / 1e3
    e["e"] = (e.end_ts - t0) / 1e3
    e["op"] = e.hlo_name.fillna("?").str.replace(r"^%", "", regex=True)
    print(f"exec {ex}: {(e.end_ts.max() - t0) / 1e6:.3f} ms, {len(e)} instructions")
    for pc, g in e.groupby("pcore_idx"):
        print(f"\n=== pcore {pc}: busy (any engine) "
              f"{merged(list(zip(g.s, g.e))) / 1e3:.3f} ms of {(g.e.max() - g.s.min()) / 1e3:.3f}")
        rows = []
        for op, v in g.groupby("op"):
            busy = {eng: merged(list(zip(vv.s, vv.e))) for eng, vv in v.groupby("engine")}
            rows.append((v.s.min(), v.e.max(), op, busy))
        rows.sort()
        for s, en, op, busy in rows:
            if en - s < 5 and sum(busy.values()) < 5:
                continue
            b = " ".join(f"{k[:3]}={x:.0f}" for k, x in sorted(busy.items()) if x >= 2)
            print(f"  {s:8.1f} {en:8.1f} {en - s:7.1f}us  {op[:34]:34s} {b}")
        gp = gaps(list(zip(g.s, g.e)), min_gap / 1e3 * 1e3 / 1e3 if False else min_gap / 1e3)
        tot = sum(b - a for a, b in gp)
        print(f"  idle gaps > {min_gap / 1e3:.0f} us: {len(gp)}, total {tot:.0f} us: "
              + ", ".join(f"[{a:.0f},{b:.0f}]" for a, b in gp[:20]))


if __name__ == "__main__":
    main()
