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

# 2 chips (TP4 x CP2, LNC2, decode at LNC1 on 16 ranks), work in progress toward 16 fps
Same video and fps definition; GEN from one process's last warm pass, gate = the MX gate above (G1 floor 20.05 dB).
Flags: `kiro_mx4/flags_2chip.txt` + `kiro_mx4/flags_2chip_best.txt` (the r5 set).
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
Refuted on the way (ledger): DVE exp offload (any share), paired exps, GpSimd adds, fused QKV (accuracy), TSPLIT,
token-major attention output, glue kernels, bf16 residual (accuracy), FiLM bf16 all-reduce (accuracy), collective
algorithm / DMA knobs, DMA priority for the attention loads, TP-free FFN (M=544 matmuls slower than the collectives
they save). Where the 2-chip block (3.57 ms in the pipeline, 3 cached chunks) goes: attention ~1.5 ms (exp-bound on
the Scalar engine at ~0.5 us per [128, 512] tile), MX matmuls ~1.07 ms (42-61% of MX peak), ~0.35 ms fully idle at
the TP collectives, the rest elementwise glue; the camera FiLM graphs are 222 ms of each 4.9 s generation. 16 fps needs GEN <= ~4.25 s (-0.66 s).
