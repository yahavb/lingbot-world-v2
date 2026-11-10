"""Clip a rank's system-trace parquet to the timed generation pass: the last `gen_ms` of the last
busy segment longer than 1 s (segments split at idle gaps > 0.5 s). The trace also holds compile,
warm-up and post-run work.
    python3 kiro/e2e/trace_window.py <SystemProfileEvents.parquet> <gen_ms> <out.parquet>"""
import sys
import duckdb
src, gen_ms, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
c = duckdb.connect()
rows = c.execute(f"select start_ts, end_ts from read_parquet('{src}') where name = 'nc_exec_running' "
                 "order by start_ts").fetchall()
segs, pe = [], None
for a, b in rows:
    if pe is None or a - pe > 0.5e9:
        segs.append([a, b])
    segs[-1][1] = max(segs[-1][1], b); pe = max(pe or 0, b)
seg = [s for s in segs if s[1] - s[0] > 1e9][-1]
lo, hi = int(seg[1] - gen_ms * 1e6), int(seg[1])
c.execute(f"copy (select * from read_parquet('{src}') where start_ts >= {lo} and start_ts <= {hi}) "
          f"to '{out}' (format parquet)")
print(f"window [{(lo - rows[0][0]) / 1e9:.2f}, {(hi - rows[0][0]) / 1e9:.2f}] s of the trace ({gen_ms:.0f} ms)")
