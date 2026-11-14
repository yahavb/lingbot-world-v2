import sys, duckdb
pq, pat = sys.argv[1], sys.argv[2]
con = duckdb.connect()
cols = [r[0] for r in con.execute(f"describe select * from '{pq}/Instruction.parquet'").fetchall()]
src = "coalesce(nki_source_location,'')||' '||coalesce(bir_debug_info_source_location,'')"
rows = con.execute(f"select start_ts, start_ts+duration_ns, engine, ({src}) like '%{pat}%' from '{pq}/Instruction.parquet'").fetchall()
t0 = min(r[0] for r in rows); t1 = max(r[1] for r in rows)
def union(iv):
    iv = sorted(iv); out = []; 
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
def tot(u): return sum(b - a for a, b in u)
def inter(u, v):
    i = j = 0; s = 0
    while i < len(u) and j < len(v):
        a = max(u[i][0], v[j][0]); b = min(u[i][1], v[j][1])
        if b > a: s += b - a
        if u[i][1] < v[j][1]: i += 1
        else: j += 1
    return s
eng = sorted(set(r[2] for r in rows))
att = [r for r in rows if r[3]]
print(f"block {(t1-t0)/1e6:.3f} ms; instructions matching '{pat}': {len(att)}")
if not att: sys.exit()
a0 = min(r[0] for r in att); a1 = max(r[1] for r in att)
print(f"attention span {(a0-t0)/1e6:.2f}..{(a1-t0)/1e6:.2f} ms = {(a1-a0)/1e6:.3f} ms")
for e in eng:
    u = union([[r[0], r[1]] for r in att if r[2] == e])
    if u: print(f"  attention on {e:8s} busy {tot(u)/1e6:.3f} ms")
tensor_all = union([[r[0], r[1]] for r in rows if 'ensor' in r[2]])
softmax = union([[r[0], r[1]] for r in att if ('ector' in r[2] or 'calar' in r[2])])
span = [[a0, a1]]
print(f"  Tensor idle inside attention span: {(a1-a0-inter(tensor_all, span))/1e6:.3f} ms")
print(f"  Vector/Scalar attention work with Tensor idle: {(tot(softmax)-inter(softmax, tensor_all))/1e6:.3f} ms")
# gaps: Tensor idle stretches > 50 us inside the span
tu = [[max(a, a0), min(b, a1)] for a, b in tensor_all if b > a0 and a < a1]
gaps = [(tu[k][1], tu[k+1][0]) for k in range(len(tu)-1) if tu[k+1][0] - tu[k][1] > 50000]
print(f"  Tensor gaps >50us in span: {len(gaps)}, total {sum(b-a for a,b in gaps)/1e6:.3f} ms; longest: " +
      ", ".join(f"{(a-t0)/1e6:.2f}-{(b-t0)/1e6:.2f}" for a, b in sorted(gaps, key=lambda g: g[0]-g[1])[:4]))
