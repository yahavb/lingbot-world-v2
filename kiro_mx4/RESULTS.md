# 16 fps at 480x768 on 4 trn3 chips (MXFP8, LNC2)
One 81-frame video (`examples/03`, seed 42, 4 steps, chunk 3, window 12, sink 1), fps = 81 / (GEN + VAE): GEN = the
last warm pass in one process, VAE = the faster of 2 decodes (`perf/k8s_sweep.sh` definition). Same pod, trn3-dev1.48xlarge
node .137 devices 4-7, torch-neuronx 2.14.3 / SDK 2.33. Every number: `kiro_mx4/ledger.jsonl`.
| Config | Chips | GEN (ms) | VAE (ms) | fps |
|---|---|---|---|---|
| deployed MX config (`lbw-v2-lnc2-mx-deploy.yaml` flags), LNC2 TP4 x CP4 | 4 | 5222.6 | 630 | 13.84 |
| + MX loads by direct-pattern DMA transpose, MX on all 12 block Linears | 4 | 4736.2 | 630 | 15.09 |
| + chained CTE attention calls, token-major V, queries padded to 32 | 4 | 4407.3 | 631 | 16.08 |
| **+ cross-attention over the 128 distinct context rows (`CROSS_DEDUP=128`)** | 4 | **4270.9** | 630 | **16.53** |
| **+ trn2 start/pipeline flags (`POST_LEAN KV_REUSE START_OVERLAP=4 FILM_SHARD SCHED_FUSE GC_OFF`), 5 passes, gate PASS** | 4 | **4166.9** | 630 | **16.89** |
| same flags, TP4 x CP2 | 2 | 7362.4 | 1173 | 9.49 |
Decoding at LNC1 (32 ranks, same devices) takes 557-566 ms instead of 630 (+0.4 fps), not counted above: the serving
path decodes inline at the DiT's LNC.
## What changed
1. `LINGBOT_MX_FAST_DMA=1` (`kernels/ffn_mx.py`): nkilib `matmul_mxfp8` loads the activation with the direct 4-D DMA
   gather-transpose instead of vector offsets built on GpSimd. The deployed MX block spent 1.94 GB of spill reloads and
   1.5 ms of GpSimd per block there (`load_apis.py:464`); block 3.97 -> 3.68 ms. This is why MX gave nothing at LNC2.
2. `LINGBOT_LIN_MX_SKIP=` (empty): o, cross o and cam layer 2 in MX too; block 3.69 -> 3.60 ms.
3. Merge of `loop_engineering_8dev` (trn2): `LINGBOT_CTE_CHAIN=1` (each self-attention call continues the previous
   call's online softmax, no merge graph; the kernel's trn3 assert lifted, output checked against the unchained
   call), `LINGBOT_V_TOKEN_MAJOR=1`, `LINGBOT_CTE_QPAD=32`: block 3.60 -> 3.30 ms; `LINGBOT_CROSS_DEDUP=128`: 3.30 -> 3.25.
## Refuted at 4 chips (do not rerun without new information)
LNC2-tuned MX tiles (0), `NEURON_RT_IO_RING_CACHE_SIZE=1024` (0), `LINGBOT_CROSS_TPOUT` (+0.9%), `LINGBOT_RS_BF16` (0),
`LINGBOT_KV_COALESCE` (0), constant-offset attention `LINGBOT_CMAX=2` with 2 x 544-row calls (+6%) or with
`LINGBOT_CMAX_BANK_REUSE=1` in one call (+2.6%).
## Reproduce
`kubectl create -f lbw-mx-lnc2-4chip-job.yaml` (l-lnc2 claim), or in a dev pod `bash kiro_mx4/e2e.sh <name> <K=V ...>`.
## Accuracy
`perf/e2e/gate.sh` against the CPU fp32 rollout `/node-cache/e2e/gate/r480_ex03_s42`, B = Yahav's approved MX LNC1 config
(`mx-lnc1-dge-g`), on the 16.54 fps repeat of the 16.53 fps config (`g4_dedup`): **PASS** (`kiro_mx4/gate/verdict_g4_dedup.json`).
| Check | B (approved MX LNC1) | C (this, 4 chips LNC2) |
|---|---|---|
| G0 first 9 frames PSNR vs R | 28.75 dB | 31.54 dB |
| G0 chunk-0 noise prediction rel-RMS vs R | 0.0396 | 0.0391 |
| G1 whole-video PSNR vs R | 20.55 dB | 20.51 dB (allowed -0.5) |
| G2 worst frame vs R | 15.99 dB | 15.50 dB (#75, allowed -1.0) |
| G3 first-chunk x0 rel-RMS vs R | 0.0673 | 0.0555 |
Not covered: G4 (fp32 CPU decodes), D1 (repeat run), D2 (flags off == main). This is the MX gate (floor 20 dB); the bf16
4-device approved base scored 39.4 dB on the first 9 frames.

# 2 chips (TP4 x CP2, LNC2, decode at LNC1 on 16 ranks): 16.11 fps, gate PASS (t27)
Same video and fps definition; GEN from one process's last warm pass, gate = the MX gate above (G1 floor 20.05 dB).
All rows from t5 on ran on the same 2-device pod (kiro-ishaybee-dev2); the other two 2-device pods on the node measure the same GEN flags ~25 ms and ~90 ms slower.
Flags: `kiro_mx4/flags_2chip.txt` + `kiro_mx4/flags_2chip_best.txt` (the t27 set).
| Run | Change | GEN (ms) | VAE (ms) | fps | Gate |
|---|---|---|---|---|---|
| s0_base | `flags_2chip.txt` | 6976.3 | 889 | 10.30 | FAIL (G1 19.99) |
| rp1 | + `MX_RP_POST` | 6314.8 | 889 | 11.24 | PASS |
| h1 | + `FFN_EPI FFN_UP_PAD M2176_DEFAULT QK_POST` | 5707.2 | 889 | 12.28 | PASS |
| k1 | + `CMAX=3` (512-query P@V, transposes) | 5625.1 | 890 | 12.43 | PASS |
| p1 | key-major scores, `C3_SUMMODE=pef SUMGROUP=16 DEPTH=4` | 5035.2 | 889 | 13.67 | FAIL (G1 18.85; G0/G3 at or better than B) |
| **r1** | **+ `VBIAS_FOLD QK_POST_T`** | **4937.2** | 890 | **13.90** | **PASS** (G1 20.14) |
| r1 + `VAE_T=1` | decode 813 ms (same frames, 52.76 dB vs R either way) | 4937.2 | 813 | 14.09 | (decode-only change) |
| **r5** | **+ `VAE_T=1 FILM_ALL=1` (all chunks' camera FiLM before the loop: 222 ms vs 278)** | **4906.0** | 811 | **14.17** | **PASS** (G1 20.62) |
| r7 / r7b | r5 set + `BLOCK_GROUP=4` (not kept) | 4866.6 / 4865.6 | 816 / 812 | 14.25 / 14.27 | FAIL (G2 14.18, deterministic) |
| r8 | r5 set + `VAE_HEAD_BF16=1`, VAE `-O1 --model-type=unet-inference` (decode 715 ms) | 4898.2 | 715 | 14.43 | PASS (G1 20.62) |
| r9 | + `BLOCK_GROUP=8` | 4885.5 | 715 | 14.46 | PASS (G1 20.94, G2 15.72) |
| t1 | + `FILM_TPFREE=1` (camera FiLM MLP with full weights on the TP token shard: no all-reduce / all-to-alls) | 4766.2 | 715 | 14.78 | PASS (G1 21.02) |
| t2 | + `CROSS_TPFREE=1` (cross-attention with full weights on the token shard: no AG / q-norm AR / RS) | 4721.4 | 716 | 14.90 | PASS (G1 20.97, c0s0 0.0439) |
| t3 | + `HOT_HEADS=default` (exact layers: hot heads exact, the rest constant-offset), `MX_QKV_FUSED=1 MX_NSHARD=5120x3840` (fused q/k/v, default tiles, N split: bit-identical), `FILM_ALL_CHUNKS=7` | 4594.0 | 715 | 15.26 | PASS (G1 20.22, G2 15.41, c0s0 0.0412) |
| t5 | + `MX_TUNED_M544=1` (tuned tiles for the full-weight cross q / o at M=544) | 4561.0 | 715 (-O1) | 15.35 | PASS (G1 20.65) |
| t8 | + `C3_DEPTH=3`, `VAE_NKI_HALO=1` (decode: norm + SiLU + halo + cache of every C <= 128 conv input as one NKI kernel) | 4555.1 | 646 | 15.57 | PASS (G1 20.65, G2 16.81) |
| t11 | + `VAE_NKI_CONV=all VAE_NKI_CONV_SQ=bf16 VAE_NKI_CONV_LA=1 VAE_NKI_CONV_CMAX=192` (decode: norm + halo + cache + causal 3x3x3 conv as one NKI kernel, C <= 192) | 4558.4 | 582 | 15.76 | PASS (G1 20.65, G2 16.81) |
| t14 | + `RESIDUAL_BF16=1` (bf16 residual stream between the block's steps) | 4521.6 | 587 | 15.86 | PASS (G1 20.33, G2 15.41, c0s0 0.0396) |
| t17 | + `VAE_NKI_UP=2d VAE_NKI_UP_FOLD=1` (decode: nearest x2 upsample + 3x3 conv as one NKI kernel on the low-res tile) | 4511.8 | 573 | 15.93 | PASS (G1 20.33, G2 15.41, c0s0 0.0396) |
| t19 | + `VAE_LOW_T=20` (decode: conv_in, mid block and stage 0 on all 20 latent frames in one step, the rest frame by frame) | 4516.2 | 552 | 15.98 | PASS (G1 20.33, G2 15.41, c0s0 0.0396) |
| t22 | + `VAE_NKI_UP=all` (the temporal upsamples too), `COMMIT_KV_ONLY=1` (the commit forward's last block stops after its K / V; GEN-neutral) | 4517.3 | 548 | 15.99 | PASS (G1 20.48, G2 16.10, c0s0 0.0396) |
| **t27** | **+ `VAE_T=2` (decode 2 latent frames per step), `VAE_NKI_CONV_LA=0` (the fused conv kernel's look-ahead off: at 8 frames per call it corrupted bursts of frames in about half the decodes)** | **4519.4** | **508** | **16.11** | **PASS** (G1 20.48, G2 16.10, c0s0 0.0396) |
| t27b | t27 on the second 2-device pod (GEN ~25 ms slower there) | 4552.8 | 508 | 16.01 | PASS (same frames) |
Refuted on the way (ledger): DVE exp offload (any share), paired exps, GpSimd adds, fused QKV (accuracy), TSPLIT,
token-major attention output, glue kernels, bf16 residual (accuracy), FiLM bf16 all-reduce (accuracy), collective
algorithm / DMA knobs, DMA priority for the attention loads, TP-free FFN (M=544 matmuls slower than the collectives
they save). Where the 2-chip block (3.57 ms in the pipeline, 3 cached chunks) goes: attention ~1.5 ms (exp-bound on
the Scalar engine at ~0.5 us per [128, 512] tile), MX matmuls ~1.07 ms (42-61% of MX peak), ~0.35 ms fully idle at
the TP collectives, the rest elementwise glue; the camera FiLM graphs are 222 ms of each 4.9 s generation. 
At t3, 16 fps needs GEN + VAE <= 5062 ms, i.e. GEN <= ~4347 ms at a 715 ms decode (-247 ms). The block (bench, 3 cached chunks) is 3.63 ms: pre-attention 0.63 (fused QKV 338 us, qk_post ~190 us on the critical path), attention ~1.53 (Scalar ~100% busy after a ~90 us start), cross 0.56, FFN 0.6. Also refuted since r5: N split for the separate q/k/v (DMA-bound in-graph), PAIR=2 key-tile pairs (Tensor saturates), bf16 cross dedup, ATTN_EARLY=1; the M2176 tuned-tile table predates the NKI trace-cache fix (every candidate after the first re-timed one kernel), re-swept with the cache off: only q/k/v (now fused, default tiles) and the M=544 cross matmuls changed.
