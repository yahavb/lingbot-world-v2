"""Wall time of one graph execution split by source group (20 us bins, each bin charged to the group
with the most instruction time in it), from a pair's instruction parquet (pcore 0).
    python3 kiro/e2e/block_split.py <pair dir>"""
import collections, re, sys
import duckdb
d = sys.argv[1]
c = duckdb.connect()
rows = c.execute(f"""select start_ts, end_ts, engine,
  coalesce(nki_source_location, bir_debug_info_source_location, '') src, opcode,
  coalesce(spill_save_bytes,0), coalesce(spill_reload_bytes,0)
  from read_parquet('{d}/pq/Instruction.parquet') where coalesce(pcore_idx, 0) = 0""").fetchall()
t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows)
print(f"pcore0 window {(t1 - t0) / 1e6:.3f} ms, {len(rows)} instructions")
def grp(src, op):
    if "attention_cte" in src: return "attention kernel"
    m = re.search(r"model_fast.py:(\d+)", src)
    if m: return f"model_fast.py:{m.group(1)}"
    if src: return src.split("/")[-1][:34]
    return "no source: " + (op or "")[:20]
B = 20000
bins = collections.defaultdict(collections.Counter)
for a, b, e, src, op, ss, sr in rows:
    g = grp(src, op); s = a
    while s < b:
        k = (s - t0) // B; e2 = min(b, t0 + (k + 1) * B); bins[k][g] += e2 - s; s = e2
own = collections.Counter()
for k, cn in bins.items(): own[cn.most_common(1)[0][0]] += B
idle = (t1 - t0) - sum(own.values())
tot = t1 - t0
for g, v in own.most_common(20): print(f"{g:40s} {v / 1e6:6.3f} ms {100 * v / tot:5.1f}%")
print(f"{'(no instruction)':40s} {idle / 1e6:6.3f} ms {100 * idle / tot:5.1f}%")
print("spill save / reload MB", sum(r[5] for r in rows) / 1e6, sum(r[6] for r in rows) / 1e6)
