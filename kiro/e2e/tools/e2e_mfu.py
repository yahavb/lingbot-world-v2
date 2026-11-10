#!/usr/bin/env python3
"""End-to-end model-FLOPs utilization of a pipeline run (the DiT, or any repeated graph).

  mfu.py <wall_s> <devices> [--fwd 60] [--flops-per-fwd 3.548e14] [--peak 1.573e14] [--lnc 2]

flops-per-fwd: model FLOPs of ONE forward summed over all ranks. Take it from a device profile of
the forward graph: summary-json model_flops (per rank) x ranks; the parallel layout does not change
the model FLOPs, so measure it once. peak: bf16 FLOP/s per logical core, derived from the same
profile as model_flops / (mfu_estimated_percent x total_time). Defaults are o2-14b on trn3 at LNC2
(3.548e14 per forward, 1.573e14 per logical core). Ranks = devices x 8 physical cores / lnc.
MFU here counts replicated work (e.g. a text encoder every rank recomputes) once.
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("wall_s", type=float); ap.add_argument("devices", type=int)
ap.add_argument("--fwd", type=int, default=60); ap.add_argument("--flops-per-fwd", type=float, default=3.548e14)
ap.add_argument("--peak", type=float, default=1.573e14); ap.add_argument("--lnc", type=int, default=2)
a = ap.parse_args()
ranks = a.devices * 8 // a.lnc
print(f"{100 * a.fwd * a.flops_per_fwd / (ranks * a.peak * a.wall_s):.1f}%")
