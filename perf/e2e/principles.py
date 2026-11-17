"""Seven-principle metrics per DiT component (self-attn, cross-attn, FFN) from ONE pair's instruction
table (neuron_pairs.sh output dir), references/profiling.md "Seven principles".

Each instruction is assigned to a component by its custom-call id (kernel calls: given on the command
line, from the time-ordered custom-call inventory) or else by its source line (model_fast.py line
ranges of the component's code). Unassigned instructions are reported as 'other' / 'no source'.

  python3 perf/e2e/principles.py <pair dir> <ranges.json> 'self=161,189,...' 'cross=381,...' 'ffn=475,...'

Per component (ms are summed over every block in the graph):
  P1 DMA instructions, GpSimd time on DMA/descriptor instructions, bytes per DMA instruction, share < 2 KiB
  P2 transpose time on Tensor (MATMUL/LDWEIGHTS transpose), Vector (STREAM_TRANSPOSE), DMA transposes
  P3 Tensor busy / component wall, Tensor semaphore-wait time
  P4 per-engine busy (merged) -> busiest engine; copy/cast time per engine
  P5 duration-weighted SBUF partitions used by Vector/Scalar/GpSimd instructions (of 128)
  P6 mask/select/iota/memset time
  P7 indirect (gather/scatter) DMA instructions
"""
import glob
import json
import re
import sys

import duckdb

d, rng_path = sys.argv[1], sys.argv[2]
cc = {}
for a in [x for x in sys.argv[3:] if x != "ALL"]:
    comp, ids = a.split("=")
    for i in ids.split(","):
        cc[f"%custom-call.{i}"] = comp
rng = json.load(open(rng_path))
f = glob.glob(f"{d}/pq/**/Instruction.parquet", recursive=True)[0]
con = duckdb.connect()
df = con.execute(f"""select start_ts s, start_ts + duration_ns e, duration_ns dur, engine, opcode,
  coalesce(hlo_name, '') hlo, coalesce(nki_source_location, bir_debug_info_source_location, '') src,
  coalesce(hbm_read_bytes,0)+coalesce(hbm_write_bytes,0)+coalesce(sbuf_read_bytes,0)+coalesce(sbuf_write_bytes,0) bytes,
  sbuf_write_partition_start ps, sbuf_write_partition_end pe, coalesce(tensor_instruction_type,'') tit
  from '{f}'""").df()
T = (df.e.max() - df.s.min()) / 1e6


ALL = "ALL" in sys.argv[3:]


def comp_of(hlo, src):
    if ALL:
        return "all"
    k = hlo.split(" = ")[0]
    if k in cc:
        return cc[k]
    m = re.search(r"model_fast\.py:(\d+)", src)
    if m:
        ln = int(m.group(1))
        if 1895 <= ln <= 1917:
            return "self"
        if 1918 <= ln <= 1921:
            return "cross"
        if 1922 <= ln <= 1930:
            return "ffn"
        best = None
        for a, b, c, _n in rng:
            if a <= ln <= b and (best is None or b - a < best[1] - best[0]):
                best = (a, b, c)
        if best:
            return {"rowpar": "cross", "norm": "self", "mxhelper": "mx?", "blockfwd": "other"}.get(best[2], best[2])
        return "other"
    if "attention.py" in src or "attention_cte" in src:
        return "self?"
    return "nosource" if not src else "other"


df["comp"] = [comp_of(h, s) for h, s in zip(df.hlo, df.src)]


def merged(x):
    if x.empty:
        return 0.0
    iv = x[["s", "e"]].sort_values("s").to_numpy()
    tot, cs, ce = 0, iv[0, 0], iv[0, 1]
    for s_, e_ in iv[1:]:
        if s_ > ce:
            tot += ce - cs; cs, ce = s_, e_
        else:
            ce = max(ce, e_)
    return (tot + ce - cs) / 1e6


print(f"graph window {T:.3f} ms, {len(df)} instructions")
share = df.groupby("comp").dur.sum() / df.dur.sum()
print("instruction-time share by component:", ", ".join(f"{k} {v:.1%}" for k, v in share.sort_values(ascending=False).items()))
out = {}
for comp in (("all",) if ALL else ("self", "cross", "ffn")):
    x = df[df.comp == comp]
    wall = merged(x)
    dma = x[x.opcode.str.contains("DMA|DGE", regex=True)]
    busy = {eng: merged(v) for eng, v in x.groupby("engine")}
    top = max(busy, key=busy.get) if busy else "-"
    trans_t = x[(x.engine == "Tensor") & (x.opcode.str.contains("transpose", case=False) | x.tit.str.contains("transpose", case=False))].dur.sum() / 1e6
    trans_v = x[x.opcode.str.contains("TRANSPOSE")  & (x.engine != "Tensor")].dur.sum() / 1e6
    tsem = x[(x.engine == "Tensor") & x.opcode.str.contains("SEMAPHORE")].dur.sum() / 1e6
    cp = x[x.opcode.str.contains("COPY|CAST", regex=True)].groupby("engine").dur.sum() / 1e6
    lanes = x[x.engine.isin(["Vector", "Scalar", "GpSimd"]) & x.ps.notna() & x.pe.notna()]
    w = (lanes.pe - lanes.ps + 1).clip(1, 128)
    lane_avg = float((w * lanes.dur).sum() / max(lanes.dur.sum(), 1))
    lane_narrow = float(lanes[w < 64].dur.sum() / max(lanes.dur.sum(), 1))
    mask = x[x.opcode.str.contains("IOTA|SELECT|MEMSET|AFFINE", regex=True)].dur.sum() / 1e6
    gather = int(x.opcode.str.contains("GATHER|SCATTER", regex=True).sum())
    small = float((dma.bytes < 2048).mean()) if len(dma) else float("nan")
    out[comp] = dict(
        wall_ms=round(wall, 3), instr=len(x),
        P1=dict(dma_instr=len(dma), gpsimd_dma_ms=round(dma[dma.engine == "GpSimd"].dur.sum() / 1e6, 3),
                bytes_per_dma=int(dma.bytes.mean()) if len(dma) else 0, share_lt_2KiB=round(small, 3)),
        P2=dict(tensor_transpose_ms=round(trans_t, 3), other_engine_transpose_ms=round(trans_v, 3)),
        P3=dict(tensor_busy_ms=round(busy.get("Tensor", 0), 3), tensor_share_of_wall=round(busy.get("Tensor", 0) / wall, 3) if wall else 0,
                tensor_semaphore_wait_ms=round(tsem, 3)),
        P4=dict(busy_ms={k: round(v, 3) for k, v in busy.items()}, busiest=top,
                copy_cast_ms={k: round(v, 3) for k, v in cp.items()}),
        P5=dict(avg_partitions=round(lane_avg, 1), share_time_lt_64_partitions=round(lane_narrow, 3)),
        P6=dict(mask_select_iota_memset_ms=round(mask, 3)),
        P7=dict(gather_scatter_instr=gather),
    )
print(json.dumps(out, indent=1))
