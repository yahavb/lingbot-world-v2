"""Device busy / idle and the largest idle gaps of a (windowed) system-trace parquet, plus the
collective (cc_running) time. python3 kiro/e2e/gaps.py <events.parquet>"""
import sys, duckdb
c = duckdb.connect()
f = sys.argv[1]
ex = c.execute(f"select start_ts, end_ts from read_parquet('{f}') where name = 'nc_exec_running' order by start_ts").fetchall()
t0, t1 = ex[0][0], max(b for a, b in ex)
busy, gaps, pe = 0, [], None
for a, b in ex:
    if pe is not None and a > pe:
        gaps.append((a - pe, (pe - t0) / 1e6))
    s = a if pe is None else max(a, pe)
    if b > s: busy += b - s
    pe = b if pe is None else max(pe, b)
span = t1 - t0
print(f"span {span / 1e6:.1f} ms, device busy {busy / 1e6:.1f} ms ({100 * busy / span:.0f}%), idle {(span - busy) / 1e6:.1f} ms, {len(ex)} executions")
gaps.sort(reverse=True)
print("largest gaps (ms @ ms):", " ".join(f"{g / 1e6:.2f}@{t:.0f}" for g, t in gaps[:15]))
print("gaps > 0.1 ms:", sum(1 for g, _ in gaps if g > 1e5), " total", f"{sum(g for g, _ in gaps if g > 1e5) / 1e6:.1f} ms;",
      "gaps <= 0.1 ms total", f"{sum(g for g, _ in gaps if g <= 1e5) / 1e6:.1f} ms")
cc = c.execute(f"select count(*), sum(end_ts - start_ts) from read_parquet('{f}') where name = 'cc_running'").fetchone()
print(f"cc_running: {cc[0]} events, {(cc[1] or 0) / 1e6:.1f} ms")
