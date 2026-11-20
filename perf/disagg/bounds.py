"""bounds.py PAIRS_DIR [ROLE_RE...]: per hot graph, the floor of each resource and what binds the graph.

Reads every <pair>/summary.json (neuron-explorer summary-json of a NEFF + NTFF pair) and <pair>/engine_overlap.json
(perf/e2e/tools/engine_overlap.py), as captured by perf/disagg/devprof.sh. Per graph, all in ms per execution:

  t        measured device time
  compute  model FLOPs / peak (peak derived from the profiler: model_flops / (t * mfu))
  hbm      (HBM read + write bytes) / HBM bandwidth (peak / peak_flops_bandwidth_ratio)
  tensor   Tensor-engine busy time
  dma      DMA-queue busy time
  serial   time with exactly one engine busy + time with all engines idle

Classes (the binding resource is the one whose time is closest to t):
  COMPUTE       tensor busy ~ t and compute floor ~ tensor busy (the array is full)
  ARRAY         tensor busy ~ t but compute floor << tensor busy (the array is under-filled: shapes)
  HBM           hbm floor ~ t
  ONCHIP-MOVE   dma busy ~ t but hbm floor << dma busy (SBUF/PSUM moves, transposes, gathers, descriptors)
  SERIAL        single-engine + idle time is the largest share (engines do not overlap)
Also: array efficiency = compute floor / tensor busy; movement excess = dma busy / hbm floor.
The graph's ceiling is max(compute, hbm); the gap to it is t / ceiling.
"""
import glob
import json
import os
import re
import sys


def load(p):
    d = json.load(open(os.path.join(p, "summary.json")))
    s = d[list(d)[0]]
    eo = {}
    f = os.path.join(p, "engine_overlap.json")
    if os.path.exists(f):
        eo = json.load(open(f))
    return s, eo


def eng_busy(eo, name):
    for k, e in (eo.get("engines") or {}).items():
        if k.lower().startswith(name):
            return e["busy_ns"] / 1e6
    return None


def main():
    pairs = sorted(glob.glob(os.path.join(sys.argv[1], "*/")))
    print(f"{'graph':32s} {'role':4s} {'t':>7s} {'compute':>8s} {'hbm':>6s} {'tensor':>7s} {'dma':>6s} {'serial':>7s} "
          f"{'arrayEff':>8s} {'moveX':>6s}  class")
    rows = []
    for p in pairs:
        try:
            s, eo = load(p)
        except Exception:
            continue
        name = os.path.basename(p.rstrip("/"))
        t = s["total_time"] * 1e3
        mfu = s.get("mfu_estimated_percent") or 0
        peak = s["model_flops"] / (s["total_time"] * mfu) if mfu else 0
        bw = peak / s["peak_flops_bandwidth_ratio"] if peak else 0
        comp = s["model_flops"] / peak * 1e3 if peak else 0
        hbm = (s["hbm_read_bytes"] + s["hbm_write_bytes"]) / bw * 1e3 if bw else 0
        tensor = eng_busy(eo, "tensor") or s.get("tensor_engine_active_time_percent", 0) * t
        dma = s.get("dma_active_time_percent", 0) * t
        serial = ((eo.get("n_busy_ns") or {}).get("1", 0) + (eo.get("idle_ns") or 0)) / 1e6
        src = " ".join(x[1] for x in eo.get("serial_src", []))
        role = "vae" if re.search(r"vae", src + (open(os.path.join(p, "pair.json")).read() if os.path.exists(os.path.join(p, "pair.json")) else "")) else "dit"
        cand = {"COMPUTE/ARRAY": tensor, "HBM": hbm, "ONCHIP-MOVE" if dma > 2 * hbm else "HBM": dma, "SERIAL": serial}
        cls = max(cand, key=cand.get)
        if cls == "COMPUTE/ARRAY":
            cls = "COMPUTE" if comp > 0.7 * tensor else "ARRAY"
        rows.append((name, role, t, comp, hbm, tensor, dma, serial, comp / tensor if tensor else 0,
                     dma / hbm if hbm else 0, cls))
    for r in rows:
        print(f"{r[0][:32]:32s} {r[1]:4s} {r[2]:7.3f} {r[3]:8.3f} {r[4]:6.3f} {r[5]:7.3f} {r[6]:6.3f} {r[7]:7.3f} "
              f"{r[8]:8.2f} {r[9]:6.1f}  {r[10]}")
    print(json.dumps([dict(zip(["graph", "role", "t", "compute", "hbm", "tensor", "dma", "serial", "array_eff",
                                 "move_x", "class"], r)) for r in rows]))


if __name__ == "__main__":
    main()
