# lingbot-world-v2 on 4 trn3 devices: e2e latency and MFU push (November 2026)

One 81-frame 480x768 video (`examples/03`, chunk 3, window 12, sink 1, 4 denoise steps, seed 42).
e2e = GEN (last of 3 passes in one process) + VAE (faster of 2 decodes), same claim, as
`kiro/k8s_sweep.sh` measures. MFU = 5.45e15 useful FLOPs per video / (ranks x 1.573e14 x e2e)
(`tools/e2e_mfu.py --fwd 1 --flops-per-fwd 5.45e15`; FLOPs analytic: 34 DiT forwards with the growing
attention window, camera FiLM once per chunk; the block profile's `model_flops` 2.675e11 per rank per
full-window block agrees). Every number: `perf_ledger.jsonl` (read it with `tools/perf_ledger.py report`).

## Result

| Config | Devices | GEN (s) | VAE (s) | e2e (s) | fps | e2e MFU | device-s / video |
|---|---|---|---|---|---|---|---|
| deployed q20_kbnw, TP4 x CP8 (`xl-lnc2`) | 8 | 4.08 | 0.42 | 4.50 | 18.0 | 24.1% | 36.0 |
| q20 at TP4 x CP4 (`l-lnc2`, FILM_ALL=0: see below) | 4 | 6.10 | 0.63 | 6.73 | 12.0 | 32.1% | 26.9 |
| **this push, best (`lbw-v2-i2v-4dev-job.yaml`)** | 4 | **5.77** | 0.63 | **6.40** | **12.7** | **33.9%** | **25.6** |

-4.9% e2e at 4 devices against the same-layout baseline; -29% device time per video against the
8-device deployment. Accuracy (CPU fp32 reference that also covers T5): first 9 frames 36.4 -> 39.4 dB,
chunk-0 noise prediction rel-RMS 0.0176 -> 0.0136; gate PASS, all new flags off == `main` bit for bit (`gate/verdict_stack.json` on the node).
TP2 x CP8 on 4 devices: 6.74 s GEN (+10%), rejected.

## Wins (all opt-in flags; same-pod A/B/A/B, noise 0.1-0.3%)

| Flag | What | GEN effect | Numerics |
|---|---|---|---|
| `LINGBOT_CTE_EXP_SPLIT=2` (52be233) | every 2nd Q group's softmax exp on the Scalar Engine; the attention kernel is Vector-bound | -1.6% | not bit-exact; more accurate |
| `LINGBOT_CTE_PRIOR_FULL=1` + `LINGBOT_CTE_SCALAR_ROWMAX=1` | prior KV tiles skip the runtime-mask range_select (the prior is always fully used); row-max evicts split over Scalar/Vector (upstream gated off). Each alone: 0 | -2.2% | more accurate (c0s0 0.0175 -> 0.0123) |
| `LINGBOT_BLOCK_GROUP=2` | 2 DiT blocks per compiled graph (cross-block overlap) | -1.2% | within the approved base's error |
| `LINGBOT_FILM_ALL_BLOCKS=10 LINGBOT_FILM_ALL_CHUNKS=3` | all-chunk camera FiLM in graphs of 10 blocks x 3 chunks | -0.6% | within the approved base's error |
| `LINGBOT_T5_FP32=1` | fp32 T5 encode (once per prompt, before GEN) | 0 | context 3.5% -> fp32; first-9-frame PSNR +2.6 dB |
| `LINGBOT_VAE_GATHER1=1` | one all-gather + one host copy of the decoded tiles (was per frame) | 0 (outside the VAE timer): frames on host 0.32 s sooner (651 -> 330 ms) | bit-identical frames |

Stack of the last four on top of EXP_SPLIT: 5.765 / 5.765 s vs 5.863 / 5.860 s.

## Refuted (do not rerun without new information)

| Tried | Result |
|---|---|
| q20 flag set unchanged at 4 devices | the 40-block x 7-chunk FiLM_ALL NEFF (86 MB, 152 MB PE instructions/core) fails to load: `Failed to load model with collectives, Failed to allocate resource` |
| `NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE` 32 MB | same load failure |
| FILM_ALL with >= 4 chunks per all-to-all at CP4 | **wrong output** (c0s0 0.45-0.68, black/garbled video): all-to-all of >= 11.1 MB per tensor returns wrong data; 8.4 MB (3 chunks, CP4) and 9.7 MB (7 chunks, CP8) are right. Not root-caused; report to the Neuron runtime/compiler team |
| split that all-to-all inside one graph (`LINGBOT_FILM_A2A_CHUNKS`) | neuronx-cc NCC_IDDT901 DramToDramTranspose assert |
| disaggregate T5 from the DiT | T5 runs before GEN and is freed before the DiT loads: 0 ms of the timed window; DiT tensors use 11-16 of 36 GB/rank |
| VAE frames per step 3/10/20, 2 x 8 tile grid | 630-650 ms / 821-839 ms vs 630 ms |
| fused Q/K/V matmul (`LINGBOT_QKV_FUSED`) | +10% (fused TP4 weight > SBUF; o2 saw the same) |
| one all-reduce for Q and K RMSNorm sums (`LINGBOT_QK_AR1`) | 0 |
| `--model-type=transformer`, `-O3` | bit-identical output, no change (torch-neuronx already passes `--model-type transformer`); `-O1` +2% |
| `LINGBOT_ATTN_SPLIT`, `LINGBOT_ATTN_REM` | assert: incompatible with `LINGBOT_K_BLOCKED` |
| EXP_SPLIT 3 / 4 | 3: -1.0%, 4: -0.8% (2 is best) |
| BLOCK_GROUP 4 | same speed as 2 |
| one explicit transpose of the QKV input (`LINGBOT_QKV_XT`) | +0.5% |
| `LINGBOT_TOKEN_PAD` 128 / 512 | +2.5% / +9% (32 stays) |
| "VAE is host-bound" | wrong window: the steady decode is 97% device-busy; the host-bound part was the frame gather after the timer (fixed by VAE_GATHER1) |

## Where the time goes now (`BOTTLENECKS.md`)

DiT blocks are ~93% of GEN (4.3 ms per block execution, 38% MFU). Inside a block: attention kernel
35% (Vector-bound), projection matmuls 36%, compiler transposes feeding the projections 17%
(the token-major all-gathered input is transposed once per consumer matmul and not CSE'd), other 11%.
The remaining large lever is a feature-major activation layout (removes the transposes); the attention
kernel has ~0.3-0.6 s of headroom left from engine balance.

Untested, ranked: (1) feature-major activations in the block (removes most of the 17% transposes; a
rewrite of the SP block, norms and RoPE); (2) attention kernel: exp read straight from PSUM and fewer
Vector copies (~0.3 s); (3) skip the last block's attention/FFN on the 6 cache-commit forwards (exact,
~0.35%); (4) fp32 T5 in `serve.py` (not wired; the streaming server builds its own T5).

## Accuracy gate

R = the whole rollout on CPU in fp32 with the upstream (CUDA) `WanModelFast`, replayed from a run's
dumped inputs (`LINGBOT_DUMP_ROLLOUT`, `kiro/ref_fp32_rollout.py`), T5 context from an fp32 CPU T5
(`t5_ref.py` -> `CONTEXT_PT`), decoded by the fp32 CPU VAE. `gate.sh` runs `tools/validate_gate.py`.

Calibration (`calib.py`). **No user-labelled lingbot videos exist yet**, so the labels are provisional:

| Output | first 9 frames vs R | sharpness / R | c0s0 rel-RMS | label |
|---|---|---|---|---|
| deployed 8-device config | 40.1-41.4 dB | 1.03 | 0.011-0.013 | approved (deployed) |
| 4-device base | 36.4-38.4 dB | 1.03 | 0.015-0.018 | approved (same code) |
| base blurred, sigma 1 / 2 px (synthetic) | 31.1 / 28.2 dB | 0.06 / 0.01 | - | rejected (synthetic) |
| FILM_ALL at CP4, 7 or 4 chunks | 2.6 / 11.1 dB | 0.00 / 0.28 | 0.45 / 0.68 | rejected |

G0: `--floor-early-db 35 --min-sharpness 0.9 --max-early-rel 0.03`. G3 (x1.3) is taken against the
approved base, not the best-so-far: equivalent exact-attention configs span chunk-0 latent rel-RMS
0.0144-0.0244, so a ratio against the luckiest one ratchets on compile noise.

## Rerun

1. `scripts/k8s_mkjob.py kiro/e2e/bench-job.yaml <name> <node> BRANCH=e2e-4dev LAYOUTS="4x4_a 4x4_b" CFG_lnc2_4x4_a="K=V ..." ...`
   then `sed s/xl-lnc2/l-lnc2/`; outputs land on the node in `/opt/lingbot-world-v2/e2e/<job>/`.
2. Profile: add `PROFILE_CFG=<entry>` (system trace clipped to the timed pass, NEFF/NTFF pairs).
3. Reference: `kiro/e2e/gate-ref-job.yaml` (CPU) with `SRC=<job>/<cfg>`, then a bench pod with
   `R_DIRS=<R dir>` decodes it; gate in `kiro/e2e/util-pod.yaml` with `kiro/e2e/gate.sh`.
