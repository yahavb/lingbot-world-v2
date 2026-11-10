"""Execution sequence of a windowed trace: graph hash8, start (ms), duration (ms), gap before (ms).
python3 kiro/e2e/seq.py <events.parquet> [n]"""
import json, sys, duckdb
c = duckdb.connect(); f = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ex = c.execute(f"select start_ts, end_ts, extra_attributes_json from read_parquet('{f}') where name = 'nc_exec_running' order by start_ts").fetchall()
t0, pe = ex[0][0], None
for a, b, j in ex[:n]:
    g = json.loads(j).get("model_name", "?")[-32:][:8]
    print(f"{g} start {(a - t0) / 1e6:8.2f} dur {(b - a) / 1e6:7.3f} gap {((a - pe) / 1e6 if pe else 0):7.2f}")
    pe = b if pe is None else max(pe, b)
