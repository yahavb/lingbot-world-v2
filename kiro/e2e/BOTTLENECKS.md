| Area | Time now | Share | Running at | Floor | Realistic headroom | How |
|---|---|---|---|---|---|---|
| DiT projections | 2.02 s | 31% | ~67% when busy | 1.35 s | 0.2-0.4 s | fewer LDWEIGHTS/serial matmul tails; fused QKV refuted (+10%, SBUF) |
| DiT attention kernel | 1.95 s | 30% | ~36% of its FLOP floor; Vector-bound | 0.72 s | 0.3-0.6 s | engine balance (EXP_SPLIT, SCALAR_ROWMAX+PRIOR_FULL done: -2.2%); const-max (row max/rescale) small here |
| DiT compiler transposes | 0.96 s | 15% | pure overhead | 0 s | 0.2-0.5 s | feature-major activations (rewrite); per-projection input transposes are not CSE'd |
| VAE decode | 0.63 s | 10% | - | not profiled |  | VAE_T / tile grid retune refuted |
| DiT other in-block | 0.61 s | 9% | - | semaphores, DMA, copies | 0.1-0.3 s | cross-block overlap: BLOCK_GROUP=2 -1.2% |
| camera FiLM | 0.21 s | 3% | 45-54% MFU | 0.1 s | 0.03-0.1 s | batched precompute <= 3 chunks per all-to-all (larger is wrong at CP4) |
| host / idle | 0.2 s | 3% | - | 0 s | 0.05-0.13 s | already async (NO_SYNC, PREUPLOAD) |
| **Total** | **6.58 s** | | | **2.17 s** | **0.88-2.03 s** | Areas without a numeric floor count as 0; headroom ranges overlap and do not add exactly |

| # | Bottleneck | Time per video (4 devices, es2 config; GEN window 5.945 s + VAE 0.630 s) | Share | Detail | Source |
|---|---|---|---|---|---|
| 1 | projection matmuls (QKV, O, cross Q/O, FFN up/down) | 2.02 s | 31% | 36.4% of block wall | trace,derived |
| 2 | attention kernel (attention_cte_lb) | 1.95 s | 30% | 35.2% of block wall x 5.537 s of blocks | trace,derived |
| 3 | unnamed PE transposes + Vector STREAM_TRANSPOSE | 0.96 s | 15% | 17.4% of block wall; feed model_fast.py:686/1633/866/867/871 | trace,derived |
| 4 | VAE decode (16 ranks, 2-D tiles) | 0.63 s | 10% | decode_latents_width_sharded steady state | bench |
| 5 | semaphores, DMA, copies, idle inside the block | 0.61 s | 9% | 11% of block wall (6.5% all engines idle) | trace,derived |
| 6 | camera FiLM (per chunk, 7 x ~30 ms) | 0.21 s | 3% | graphs e6bf/517c/0e5e | trace |
| 7 | device idle in GEN window | 0.13 s | 2% | window 5.945 s - device busy 5.811 s | trace,derived |
| 8 | head + small graphs | 0.06 s | <1% | head 34 x 0.37 ms + ~50 ms small graphs | trace |

Rows add up to 6.58 s against 6.58 s measured.
