#!/usr/bin/env python3
"""Attribute UNNAMED tensor-engine transposes by the instructions around them.

  transpose_neighbors.py <Instruction.parquet> [--top 12]

Compiler-inserted layout transposes (MATMUL/LDWEIGHTS with tensor_instruction_type TRANSPOSE and
no source line or HLO name) cannot be targeted by source. This walks the tensor engine in time
order and charges each run of them to the next named tensor instruction (the consumer) and to the
previous one (the producer). Measured on o2-14b at TP4: 214.7 ms of such transposes, ~34% after
the row-parallel matmul (parallel.py:241) and ~26% feeding the Q/K/V and FFN-up matmuls -- which
pointed at fusing the LayerNorm into those matmuls. Needs duckdb.
"""
import argparse, collections
import duckdb

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet"); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    src = "coalesce(nullif(bir_debug_info_source_location,''), nki_source_location, '')"
    rows = duckdb.connect().execute(
        f"SELECT start_ts, duration_ns, CAST(tensor_instruction_type AS VARCHAR), coalesce(hlo_name,''), {src} "
        f"FROM read_parquet('{a.parquet}') WHERE engine ILIKE '%tensor%' ORDER BY start_ts").fetchall()
    nxt, prv = collections.Counter(), collections.Counter(); tot = 0; last = None; pending = []
    for _, d, tt, h, s in rows:
        if tt == "TRANSPOSE" and not s and not h:
            pending.append(d); continue
        label = (s.split("/")[-1] if s else "") + (" " + h.split("=")[0].strip() if h else "")
        if pending:
            t = sum(pending); tot += t; nxt[label or "(none)"] += t; prv[last or "(none)"] += t; pending = []
        if label: last = label
    if not tot:
        print("no unattributed tensor transposes"); return
    print("unattributed transpose instruction time %.1f ms" % (tot / 1e6))
    print("consumer (next named tensor instruction):")
    for k, v in nxt.most_common(a.top): print("  %7.1f ms %5.1f%%  -> %s" % (v / 1e6, 100 * v / tot, k[:100]))
    print("producer (previous named tensor instruction):")
    for k, v in prv.most_common(a.top): print("  %7.1f ms %5.1f%%  <- %s" % (v / 1e6, 100 * v / tot, k[:100]))

if __name__ == "__main__":
    main()
