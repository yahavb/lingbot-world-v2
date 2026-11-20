"""model_check.py: the stage-composition cost model against every measured LingBot 480 cell (docs/disagg/RESULTS.md).
Inputs: per-stage measurements only (DiT alone, VAE pod, inline VAE). Monolith = 12000 / (P_d(N) + T_v_inline(N));
split = 12000 / max(P_d(n_d), T_v_pod(n_v)); streams per VAE = P_d / T_v_pod(1); plus a held-out scaling fit."""
