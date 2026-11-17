# MXFP8 on LNC1: LingBot TP4 x CP8 on 4 trn3 chips

Goal: run the TP4 x CP8 layout of the 8-chip LNC2 server (32 ranks x 2 cores = 64 cores) on
4 chips at LNC1 (32 ranks x 1 core = 32 cores, half the compute) at 480x768, output quality
unchanged. Every number below is in `docs/perf/perf_ledger.jsonl` (ids in brackets); the loop
follows the e2e-latency-optimization skill (baseline, CPU fp32 gate, profile, candidate, A/B,
end to end, record).

## Result

| Config | Chips | Ranks | TP x CP | LNC | GEN (s) | VAE (s) | fps | Gate |
|---|---|---|---|---|---|---|---|---|
| server, bf16 (ledger q20_kbnw) | 8 | 32 | 4 x 8 | 2 | 4.08 | 0.42 | 18.0 | - |
| bf16 | 4 | 32 | 4 x 8 | 1 | out of device memory | | | |
| MX baseline `[mx_lnc1_base]` | 4 | 32 | 4 x 8 | 1 | 6.765 | 0.558 | 11.06 | B |
| + kept win 1 `[mx_lnc1_c12_e2e]` | 4 | 32 | 4 x 8 | 1 | 6.188 / 6.146 | 0.620 | 11.9 / 12.06 | PASS |
| + kept win 2 `[mx_lnc1_dge_e2e]` | 4 | 32 | 4 x 8 | 1 | **5.925 / 5.918** | 0.562 | **12.49 / 12.51** | PASS |

fps = 81 / (GEN + VAE), GEN = the last of 3 warm passes, second number = repeat run (same pod).
Gate: `perf/e2e/gate.sh` (validate_gate.py, pixels) vs a CPU fp32 rollout of the same inputs
(`/node-cache/e2e/gate/r480_ex03_s42`); verdicts `docs/perf/e2e/gate/verdict_mx_lnc1_*.json`.

## What made LNC1 fit (memory)

1. **MXFP8 Linears** (`LINGBOT_FFN_MX=1 LINGBOT_LIN_MX=1`, `kernels/ffn_mx.py`, nkilib
   `matmul_mxfp8` + `quantize_block_mxfp8_kernel`): weights quantized once right after load, bf16
   copies (and `LINGBOT_WT_PRE` transposed copies) freed. Rank-0 device memory after load
   10.27 -> 6.16 GiB.
2. **Time-embedding fp32 cast** (`LINGBOT_TEMB_SLICE=1`, `_seq_fp32_sliced`, model_fast.py): under the fp32 autocast the
   whole `time_projection` weight (5120 x 30720) was cast to fp32 per uncached timestep, a 600 MiB
   allocation (plus 100 MiB for time_embedding). With async launches the freed block could not be
   reused in time (`alloc_deferred -> drain_retry_fail -> oom`, memory snapshot). Now cast in 64 MiB
   row slices, same fp32 math. Found with `LINGBOT_MEM_HISTORY=1` (generate.py snapshot).
3. Peak at LNC1 after both: 10.74 GiB allocated, 16.29 GiB reserved per rank (~18 GiB per core).

Accuracy of MX itself: one MX matmul 3.7% rel-RMS vs fp32 (e4m3, block 32), bf16 0.17%
`[mx_ffn_bench]`; end to end c0s0 rel-RMS vs fp32 0.0485 (all Linears MX) -> 0.0396 (kept config).

## Kept wins

| # | Change (flag) | Principle | Block (ms) | GEN (ms) | Gate |
|---|---|---|---|---|---|
| 1 | K=1280 Linears (self o, cross o, cam layer 2) stay bf16: `LINGBOT_LIN_MX_SKIP=self_attn.o:cross_attn.o:cam_injector_layer2` | 6 skip work (per-call K pad 1280 -> 1536 cost 0.67 ms Tensor) | 5.25 -> 5.16 | | |
| 1 | Tuned matmul_mxfp8 tiles per shape at LNC1: `LINGBOT_MX_TUNED=1` (`perf/mx_tile_sweep.py`; q/k/v without PE swizzle) | 1/3 fewer re-reads | -> 4.94 | 6765 -> 6188 | PASS |
| 2 | K/V tile DMA descriptors by hardware DGE on the Sync engine: `LINGBOT_CTE_KV_DGE=hwdge_sync` | 1 setup off the busy GpSimd | 4.94 -> 4.72 | 6139 -> 5925 | PASS |

## Refuted / crashed (do not re-run without new information)

| Idea | Result | Ledger id |
|---|---|---|
| Fused MX Q/K/V (one N=3840 matmul) | block 14.0 ms, 2.8x slower (Scalar splitting the output) | mx_cand2_ffn_tiles_fusedqkv |
| Tuned tiles for o-proj (MX) | <= 1.5% block, within noise | mx_tiles_in_block |
| q/k/v tiles with PE-swizzle loads | compile error NCC_ILMM003 in the block graph | mx_tiles_in_block |
| K grouping (KGROUP 2/4), V grouping 16, double-buffered V | no gain at LNC1 | lnc1_attn_flags_ab |
| Row-max on Vector (SCALAR_ROWMAX=0), exp unsplit (EXP_SPLIT=1) | +5% / +1% block | lnc1_attn_flags_ab |
| VAE 2-D halo in one in-place buffer (LINGBOT_VAE_HALO_BUF) | compile error NCC_IBIR229 (SBUF) | vae_halo_buf |
| TP8 x CP4 at LNC1 (32 ranks) | 11.16 fps, slower than TP4 x CP8 | mx_lnc1_layouts_32r |
| TP2 x CP16 at LNC1 | out of device memory | mx_lnc1_layouts_32r |
| TP4 x CP4 on 2 chips at LNC1 (16 ranks) | out of device memory (graph/collective load, NRT_RESOURCE) | mx_lnc1_2chip_tp4cp4 |
| MX at 4-chip LNC2 TP4 x CP4 (FFN or all Linears) | GEN 5.43-5.46 vs bf16 5.42 s: no gain; peak 11.3 vs 18.5 GiB | - |

## Profile of the kept config (SKILL.md Step 3, before win 2)

System trace of a warm 3-pass run on all 32 ranks, `neuron_attribution.py`, NEFF+NTFF pairs
(`neuron_pairs.sh`), `engine_overlap.py`, per-component metrics `perf/e2e/principles.py`.
DiT: pair `rankr26-2863ms-dit-c57964f9` (2-block graph, 8.98 ms, 46.6% of GEN, coherence +0%).
VAE: pair `rankr18-209ms-vae-d07bf83f` (23 ms x 9, coherence -1%). DiT values per 2-block graph.

| Principle | Self-attn | Cross-attn | FFN | VAE |
|---|---|---|---|---|
| 1 DMAs | 4,309 DMA instr; GpSimd 2.22 of its 2.27 ms on DMA descriptors (`attention_cte_lb.py:210`, 0.44 ms serial) -> fixed by win 2 | 244; GpSimd 0.24 ms | 630; 0.23 ms | 3,226; DMA queue 9.7 ms busy, 5.6 ms alone (`vae2_1_fused.py:167` halo concat) |
| 2 transposes | Tensor 0.004 ms; other 0.28 ms | 0.001 / 0.09 | Tensor 0.47 ms: MX kernel's PE transpose of the activation = MX matmul time | Tensor 1.96 ms |
| 3 matrix fed | Tensor 2.82 of 4.98 ms (57%), wait 0.25 | 0.40 of 0.66 (60%) | 0.44 of 0.79 (56%) | 9.62 of 18.8 (51%); 18% all idle |
| 4 busiest | Tensor 2.82, Scalar 2.37 (exp 1.54), GpSimd 2.27 | Tensor 0.40 | Tensor 0.44, Vector 0.38 | Sync 9.72 ~ Tensor 9.62; copy/cast 3.6 ms |
| 5 lanes | 124 / 128 partitions; two-core split n/a at LNC1 | 126 | 125 | 64 / 128, 35% of time < 64 |
| 6 skip | masks 0.11 ms; padded rows 0.7% | 0.04 | 0.04 | 0.37 |
| 7 regular | 120 gather DMAs; ~0% < 2 KiB | 40 | 0 | 0 |

MFU (neuron-explorer estimate, peak basis not verified): DiT graphs 19.6-26.0% (top graph 26.0%),
VAE 15.2% (top graph). T5 is outside the timed window (pre-encode) and not profiled.

## Open candidates

1. MX activation orientation (principle 2): feed K-major activations so the kernel skips its PE
   transpose (`perf/mx_orient_bench.py`, not run yet).
2. VAE lane use (principle 5): convs at 64 of 128 partitions.
3. Attention kernel internals (principles 3/4): Tensor idle 43% of the self-attn wall.

Bound: the busiest engine (Tensor) is 4.27 of 8.98 ms per 2-block graph; perfect overlap at this
instruction mix would give GEN ~2.9 s (~23 fps), an estimate.

## Reproduce

- Kept config, end to end + gate dumps: `lbw-e2e-mx-lnc1-dge-job.yaml` (perf/k8s_sweep.sh).
- Gate: `bash perf/e2e/gate.sh <R> <B cfg> <C cfg> [<C repeat>]`; R from `perf/e2e/gate-ref-job.yaml`
  (CPU fp32 rollout; decode R frames on a Neuron pod).
- Step 3: `lbw-e2e-step3-job.yaml` (`perf/e2e/profile.sh`, `perf/e2e/profile_vae.sh`).
- One-block A/B: `perf/mx_block_ab.sh` (`VARIANTS="name:K=V,..."`).

Environment: trn3-dev1.48xlarge, torch-neuronx 2.14.3, neuronx-cc 2.28.7631, nki 0.7.0, SDK 2.33,
image `public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04`.
