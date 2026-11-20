"""summarize.py CSV CELL: fps over the steady part of a run (chunks after the first WARM), median DiT and VAE ms."""
import csv, statistics as st, sys
WARM = 5
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
s = [r for r in rows if int(r["chunk"]) >= WARM]
frames = sum(int(r["frames"]) for r in s[1:])
dt = float(s[-1]["t"]) - float(s[0]["t"])
med = lambda k: round(st.median(float(r[k]) for r in s if r.get(k)), 1) if any(r.get(k) for r in s) else None
print(f"{sys.argv[2]}: fps {frames / dt:.2f} over {len(s)} chunks, dit_ms {med('dit_ms')}, vae_ms {med('vae_ms')}")
