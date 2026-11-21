# DiT/VAE disaggregation: every measured number and where it comes from

This file is the data index for paper_draft.md. All runs are on KEEP-trn3-dev1 (trn3-dev1, LNC2, Neuron SDK 2.33),
2026-11-19 to 11-21.

- LingBot rows are in perf/disagg/results/summary.txt, with per-chunk CSVs perf/disagg/results/<cell>.csv.
- Stable Audio rows are on github yahavb/stable-audio-3-medium, branch disagg, bench/disagg/*.log.
- Rows marked DISCARD, INVALID, FLAGLEAK or STALE in summary.txt are superseded; the reason is in the commit that
  marked them.

## LingBot-World v2, 480x768, fps per stream

| Measurement | Value | Cells |
|---|---|---|
| DiT alone, 2/4/8/16 chips | 10.07 / 17.49 / 25.84 / 31.97 fps (period 1192 / 686 / 464 / 375 ms) | p1_dit{2,4,8,16}_480 |
| Co-located, 4/8/16 chips, with async emit | 15.54 / 22.63 / 26.70 fps | p9_emit1_r*, p16_mono8_fps, p16_mono16e_fps* |
| Co-located, 4/8/16 chips, without async emit (superseded) | 14.33 / 20.03 / 23.37 fps | p6_monofix4, p6_monofix8, p6_monofix16u |
| Co-located, true per-phase times (synced) | DiT 552 / 371 / 300 ms; VAE 151 / 133 / 143 ms | p6_monofix{4,8,16} [PHASE] |
| VAE pod alone, 1/2/4/8 chips | 324 / 213 / 173 / 153 ms per chunk | vae-*c logs |
| Split 2+1 / 2+2 (common flags) | 10.07 / 10.06 fps | p3_split_d2v{1,2}_480 |
| Split 2+1 / 2+2 (tuned 2-chip flags) | 14.66 / 14.58 fps | p3_split_d2tv{1,2}_480 |
| Split 4+1 / 4+2 / 4+4 | 17.54 / 17.52 / 17.47 fps | p3_split_d4v* |
| Split 8+1 / 8+2 / 8+4 / 8+8 | 25.76 / 25.95 / 25.48 / 25.60 fps | p3_split_d8v* |
| Split 16+1 / 16+2 / 16+4 | 32.14 / 32.21 / 32.17 fps | p3_split_d16v* |
| Single pod, 4+1 / 8+1 | 17.51 / 25.74 fps | p11_combo4p1, p11_combo8p1 |
| Placement, 4+1, same node / other node | 17.58 / 17.60 fps | pl_same_fps, pl_far_fps |
| Async emit, co-located 4 | 14.21 → 15.54 fps, byte-identical | p9_emit{0,1}_r{1,2} |
| MX transpose on tensor engine vs DMA | 14.90 vs 15.54 fps (refuted) | p15_* |

## LingBot 720p (704x1280 grid)

| Measurement | Value | Cells |
|---|---|---|
| Co-located 16, with async emit | 11.38 fps | p16_mono16e_720 |
| Co-located 16, without async emit (superseded) | 9.96 fps | p7_mono16_720 |
| Co-located 8 | does not fit (device memory once the VAE loads) | h8-inline logs |
| Co-located 4 and split 4+x | do not fit | h4-inline, hs4-split logs |
| Split 8+1 / 8+2 / 8+4 | 8.46 / 8.28 / 8.38 fps | p3_split_d8v*_720 |
| Split 16+1 / 16+2 / 16+4 | 12.68 / 12.65 / 12.16 fps | p3_split_d16v*_720 |

## LingBot denoising steps (4-chip DiT, 480)

The model is distilled to 4 steps, so only fewer steps can be tested.

| Steps | Co-located | 4+1 | 4+2 | Cells |
|---|---|---|---|---|
| 2 | 23.46 fps | 28.22 fps | 33.79 fps | p13_*_st2_480 |

## LingBot N:M and scale-out

| Measurement | Value | Cells |
|---|---|---|
| One 1-chip VAE, 2-chip DiT streams, 1/2/3 streams | 10.07 / 10.06 / 10.08 fps each; frames byte-identical | nm2_N*, nm_alone/nm_pair |
| One 1-chip VAE, 4-chip DiT streams, 1/2 streams | 17.52 / 17.53 fps each | nm4_N* |
| Per-session decoder cache | 427 MiB per rank; 4 sessions 1.7 GiB | nvae-1c [SESSIONS] |
| 2 VAEs, 6 streams (3+3) | 10.01-10.05 fps | hs_6x2_* |
| 3 VAEs, 9 streams (3+3+3) | 9.99-10.09 fps, all 9 recorded | hs3_9x3_*, hs3_9x3_placement.txt |
| Elasticity, second VAE added under load | 18 min to Ready; new streams 10.06-10.07 fps | el_d/e/f |

## LingBot latency (press-to-photon, median / p90)

| Setup | Value | Cells |
|---|---|---|
| Co-located 4 | 1114 / 1416 ms | p12_lat_mono4_480 |
| 4+1 | 1221 / 1471 ms | p12_lat_split4p1_480 |
| 4+1, single pod | 1230 / 1475 ms | p12_lat_combo4p1_480 |
| 4+4 | 1114 / 1371 ms | p12_lat_split4p4_480 |
| 4+1, VAE on the same node / other node | 1216 / 1485 ms, 1231 / 1482 ms | pl_*_lat |

## Stable Audio 3, 1800 s of audio (branch disagg)

| Measurement | Value | Log |
|---|---|---|
| Co-located 1 / 2 chips | 9.53 / 6.15 s | dit-1c.log, dit-2c.run1.log |
| 1+1 / 1+2 / 1+4 / 1+8 split | 7.50 / 7.53 / 7.61 / 7.57 s | dit-1c.log |
| 2+1 / 2+4 split | 5.48 / 5.29 s | dit-2c-b.log |
| 190 s windows: co-located 2 chips, 2+1 split | 10.67 / 7.87 s | dit-2c.run1.log, dit-2c-b.log |
| Steps 2/4/8/25/50, co-located 2 chips | 3.06 / 4.07 / 6.15 / 14.88 / 27.81 s | dit-2c-fewsteps, dit-2c-steps |
| Steps 2/4/8/25/50, 2+1 | 3.75 / 3.89 / 5.48 / 14.39 / 27.02 s | same |
| Steps 2/4/8/25/50, 2+2 | 2.51 / 3.32 / 5.37 / 14.16 / 26.80 s | same |
| N:M at 25 steps, one VAE, 1/2/4/6 jobs | 14.13 / 14.16 / 14.44 / about 20.1 s per job | bench/disagg/nm/*.log |

## Bounds and profiles (4-chip co-located, 480)

| Measurement | Value | Source |
|---|---|---|
| Per-graph floors and class | DiT on-chip movement; VAE array utilization | perf/disagg/results/prof/p10_bounds.txt |
| DiT block DMA time | 37% MXFP8 activation transposes; about 25% small collective transfers; about 5% cross-core | p14_devprof_dma_480, pairs DmaPacketAggregated |

## Model

| Measurement | Value | Source |
|---|---|---|
| Composition model, 15 cells | within 1.4% | perf/disagg/model_check.py |
| Held-out 16-chip DiT | +2.5% | same |
| Held-out 8-chip VAE | +10% | same |
