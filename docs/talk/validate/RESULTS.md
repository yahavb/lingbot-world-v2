# Talk deck vs LingBot-World v2: measured results, slide by slide

Deck: `~/Downloads/talk-deck_v2.html`. Each section below is one slide: what the deck says,
what LingBot measured, and where the number comes from.

Setup for every number here, unless a row says otherwise:
- repo `~/lingbot-world-v2`, commit e18d60f (`main`)
- cluster KEEP-trn3-dev1, node ip-192-168-192-229, one `xl-lnc2` claim (8 trn3 chips, 32 ranks, TP4 x CP8)
- 81 frames, 480x768, chunk 3, window 12, sink 1, 4 denoise steps, `examples/03`, seed 42
- image torch-neuronx 2.14.0 / SDK 2.33
- job `talk-pipeline-job.yaml` (perf/k8s_sweep.sh); raw logs on the node in `/opt/lingbot-world-v2/e2e/lbw-talk-pipeline/<config>/`

How the numbers are taken:
- GEN is the last pass in one process. Pass 1 includes compiling and is not a latency number.
- VAE is the faster of two decodes.
- fps = 81 / (GEN + VAE).
- DiT per forward = the `[BENCH] DiT forwards` line: 35 forwards per video (7 chunks x 4 steps, plus 7 cache commits).

Status: **measured** = on this setup; **pending** = job still running; **wrong model** = the deck
number is Wan2.2-TI2V-5B or trn2, not LingBot.

---

## S3a: Eager first, move the model to the device

Deck: 76.6 s baseline (DiT 44.3 s, VAE 32.3 s). **Wrong model**: Wan2.2-TI2V-5B, one trn2 device.

LingBot, eager (nothing compiled, native SDPA, every loop flag off, eager fp32 VAE). Config `4x8_eager`:

| Part | Measured | Status |
|---|---|---|
| DiT | 19.2 s (35 forwards, median 0.54 s per forward) | measured |
| VAE | 2.03 s (32 ranks; 2.03 / 2.04 s) | measured |
| GEN (DiT + host) | 19.59 s | measured |
| **Total** | **21.6 s, 3.75 fps** | measured |
| T5 encode | ~19 s on first call (includes per-op compile); cached afterwards | steady value pending |
| First pass (compile + warmup) | 184.8 s | measured |
| Accuracy: chunk-0 step-0 vs fp32 CPU | rel-RMS 0.0129 | measured |
| Accuracy: VAE frames vs fp32 CPU decode | 56.4 dB | measured |

Eager with attention_cte (`4x8_eagercte`): DiT 24.3 s (median 0.71 s per forward), GEN 24.70 s, VAE 2.04 s,
total 26.7 s, 3.03 fps, c0s0 vs fp32 0.034. Measured. In eager, attention_cte is slower than native SDPA
and less accurate.

## S4c: What one graph per block buys

Deck: DiT 4.4 -> 1.91 s/step, 76.6 -> 36.8 s. **Wrong model** (Wan2.2, 30 blocks, CFG batched).
LingBot has no CFG and 40 blocks.

LingBot, one compiled graph per DiT block, native SDPA, everything else as eager (`4x8_compile`):

| | Eager | Compiled blocks | Status |
|---|---|---|---|
| DiT per forward (median) | 0.54 s | 0.173 s (3.1x) | measured |
| DiT per video | 19.2 s | 5.7 s | measured |
| GEN | 19.59 s | 6.07 s (last of 3 passes; pass 2 6.04 s) | measured |
| VAE (unchanged, eager) | 2.03 s | 2.02 s | measured |
| **Total** | **21.6 s, 3.75 fps** | **8.09 s, 10.0 fps** (2.7x) | measured |
| First pass (compile + warmup) | 184.8 s | 23.9 s | measured |

## S4e: Compile is not always the answer, the VAE

Deck: Wan2.2 VAE, compiled decoder 11.5-12.9 s vs eager + NKI 1.7 s. **Wrong model.**
Backup B4i already says LingBot is the opposite case (compiling whole VAE blocks won).

LingBot, decode-only on the same latents:

| VAE | Measured | Status |
|---|---|---|
| eager fp32 (`4x8_eager`) | 2.03 s | measured |
| compiled blocks (fused) only (`vae_fused_only`) | 1.39 s | measured |
| final: fused, bf16, -O1, 2-D tiles (`vae_final`) | 0.41 s (56.4 -> 52.7 dB vs fp32 decode) | measured |

**Finding: on LingBot, compiling the VAE wins** (2.03 -> 1.39 s from compiling blocks alone, 0.41 s
with the rest). The deck's "compile is not always the answer" example is Wan2.2's VAE and does not
hold for LingBot; for LingBot it is the B4i story.

## S5a / S5b / S5c / S5d: Profile of one compiled block, native SDPA

Deck: one block 56.1 ms; Tensor busy 39%, one engine busy 36%, all idle 16%. Three heads, 7,854
tokens. **Wrong model** (Wan2.2 `models/fast_dit.py`, trn2).

**Finding: on SDK 2.33 the deck's "native SDPA" case does not exist by default.** When a block is
compiled, torch-neuronx replaces `F.scaled_dot_product_attention` with nkilib attention_cte by itself
(`neuron_dynamo_backend/decompositions.py`, `_can_use_nki_flash_attention`, on unless
`TORCH_NEURONX_ENABLE_NKI_SDPA=0`). The profile below shows `attention_cte.py` instructions even with
`ATTENTION_BACKEND=sdpa`.

LingBot, `4x8_compile` (native SDPA in the code, so attention_cte from the compiler), rank 0, the
heaviest DiT block graph (`prof/pairs/rankr0-6758ms-dit-f22a6c8d`, 1600 calls per pass,
`engine_overlap.py`):

| | Measured | Status |
|---|---|---|
| One block | 4.37 ms (22.4% MFU) | measured |
| Tensor busy | 56.0% | measured |
| Exactly one engine busy | 23.4% | measured |
| Every engine idle | 17.1% | measured |
| 3+ engines busy | 28.1% | measured |
| Tensor time spent on transposes | ~0.86 of 2.44 ms busy (`transpose.py:714` + unattributed) | measured |
| Biggest serial item | GpSimd semaphore wait 0.24 ms | measured |

The real "before the kernel" block: job `talk-sdpa-math-job.yaml`, config `4x8_compilemath`, the same
compiled block with `TORCH_NEURONX_ENABLE_NKI_SDPA=0`, so SDPA becomes matmul -> softmax -> matmul.
Pair `lbw-talk-sdpa-math/.../prof/pairs/rankr1-14107ms-dit-67edba54` (rank 1: +2% vs the system trace;
rank 0's capture is +16%). Tools: `engine_overlap.py`, `attn_span.py` (this folder).

| LingBot block, matmul -> softmax -> matmul | Measured | Deck (Wan2.2) |
|---|---|---|
| One block | **8.98 ms** (8.82 ms per call in the system trace) | 56.1 ms |
| Tensor busy | **45.3%** | 39% |
| Exactly one engine busy | **28.5%** | 36% |
| Every engine idle | **16.6%** | 16% |
| Tensor gaps > 50 us during attention | **6 gaps, 1.54 ms**; 5 of them are the softmax (~0.3 ms each, 1.46 ms: 1.37-1.67, 2.52-2.82, 3.67-3.97, 4.84-5.12, 6.34-6.61 ms), the 6th (85 us at 7.83 ms) is not | 3 gaps, ~22 ms |
| Softmax work (Vector + Scalar) while Tensor idle | **2.32 ms** | |
| Attention on Vector (max, sum) / Scalar (exp) | **2.93 / 1.87 ms** | 12.8 / 7.7 ms |
| Source line of every attention instruction (S5b) | `wan/modules/attention.py:802`, `F.scaled_dot_product_attention` | |
| Whole pipeline in this mode | GEN 11.37 s, VAE 2.03 s, 13.4 s, 6.05 fps | 21.9 s |

So the deck's S5a-S5d story (Tensor waits for the softmax) **does reproduce on LingBot**, once the
compiler's automatic kernel is switched off: Vector and Scalar run the softmax in 6 stretches where
Tensor sits idle.

## S5f / S5g: attention_cte

Per call, attention alone, one LingBot rank's shape on one core (`demo/`, `demo/README.md`):
SDPA (matmul -> softmax -> matmul) **2.69 ms** -> nkilib attention_cte **1.57 ms** device time (1.7x).
Deck: 18.2 -> 5.3 ms (Wan2.2, 7,854 tokens).

Deck: self-attention 18.2 -> 5.3 ms per call; 16.2 -> 13.2 s per video. **Wrong model.**

LingBot, both compiled one graph per block, everything else off:

| Attention | GEN | VAE | Total | fps | Status |
|---|---|---|---|---|---|
| native SDPA in the code; the compiler swaps in attention_cte (`4x8_compile`) | 6.07 s | 2.02 s | 8.09 s | 10.0 | measured |
| attention_cte called by our code, `ATTENTION_BACKEND=cte` (`4x8_compilecte`) | 8.98 s | 2.02 s | 11.00 s | 7.37 | measured |

**Finding: on LingBot the deck's S5g claim is reversed.** Calling attention_cte ourselves is slower than
leaving native SDPA in the code (8.98 vs 6.07 s GEN), because on SDK 2.33 the compiler already uses
attention_cte for SDPA. Our `cte` path also copies every KV chunk into a zero-filled, 512-aligned
buffer before each call (`docs/perf/LOOP_ENGINEERING.md`, "Merged attention"). The LingBot win from the
kernel came later, from changing how the kernel reads the cache (`LINGBOT_CTE_MERGE` and the q20 flags; S7d row).
The per-call times (deck: 18.2 -> 5.3 ms) come from the block profiles of jobs 2 and 3: pending.

## S5h / S5i: The same block with the kernel

Deck: 56.1 -> 17.3 ms; idle 16% -> 1.1%; Tensor busy 39% -> 79%. **Wrong model.**

LingBot, the same block both ways, same shard (pairs as in S5a):

| | matmul -> softmax -> matmul | attention_cte (the compiler's) | Deck (Wan2.2) |
|---|---|---|---|
| Block | **8.98 ms** | **4.37 ms** (2.1x) | 56.1 -> 17.3 ms |
| Attention span in the block | 7.16 ms | 2.52 ms | ~32 -> ~5 ms |
| Softmax work while Tensor idle | 2.32 ms | 0.21 ms | |
| Tensor gaps > 50 us during attention | 6, 1.54 ms | 3, 0.33 ms | |
| Every engine idle | 16.6% | 17.1% | 16% -> 1.1% |
| Tensor busy | 45.3% | 56.0% | 39% -> 79% |
| 3+ engines busy | 22.1% | 28.1% | 16% -> 34% |
| Pipeline GEN | 11.37 s | 6.07 s | |

Note for the slide: on LingBot "every engine idle" does **not** drop (16.6% -> 17.1%). The kernel
removes the softmax gaps, but the rest of the block (collective waits on GpSimd, transposes) keeps
the idle time. The deck's 1.1% is Wan2.2 with no collectives in its profile.
Why our own attention_cte path is slower and less accurate (S5g): job 3 (`talk-cte-prof-job.yaml`,
`lbw-talk-cte-prof/.../prof/pairs/`, rank 1, `engine_overlap.py`):

| Block graph (explicit cte) | Per call | Calls | Tensor busy | What dominates |
|---|---|---|---|---|
| steady block (`rankr1-6946ms-dit-c172464e`) | 4.92 ms | 1600 | 53% | same as the compiler's cte block (4.37 ms) plus the copy |
| one chunk position (`rankr1-6999ms-dit-b34b4834`) | **17.89 ms** | 400 | **10.6%** | Sync + GpSimd DMAs 83% / 79% busy: `attention.py:449`, the scatter that copies each KV chunk into a zero-filled, 512-aligned buffer |

That one graph costs 7.0 s of the run's DiT time. The zero-padded keys also enter every softmax
(no bounds are passed on this path), which is why its c0s0 is 0.034 instead of 0.012 (the B6a story,
now on trn3). The fix in the q20 flags reads the cache in place (`LINGBOT_CTE_MERGE=1`).


## S5j: The same approach for the VAE convolutions

Deck: Wan2.2 VAE 6.0 -> 2.6 s with nkilib conv3d. **Wrong model.** LingBot's VAE path is compiled
blocks, not nkilib conv3d (see S4e): 2.03 -> 1.39 -> 0.41 s.

## S7a: The frame rate decides how many chips

Deck: 8 chips 18.64 fps, 16 chips 25.02 fps. **trn2**, not trn3. 16 chips is over the 8-chip budget
and is not re-run.

LingBot, 8 trn3 chips, 480x768:

| Flags | GEN | VAE | Total | fps | Status |
|---|---|---|---|---|---|
| today's `main` (`lbw-bench-job.yaml` flags, `4x8_final`) | 4.27 s | 0.41 s | 4.68 s | **17.3** | measured |
| q20 (`4x8_q20`) | 4.07 s | 0.41 s | 4.48 s | **18.08** | measured |

The 8-chip number for this slide on trn3 is **19.9 fps** (`loop_engineering_8dev` flags, no chaining; see Backups, B6d) and 18.1 fps on `main` (q20). Today's `main` flags were tuned for 4 chips
and are 5% slower at 8. 16 chips not re-run (budget).

## S7d: Sequence parallelism, two functional collectives

Deck: VAE 9.0 -> 6.0 s (one spatial tile per core). **Wrong model** (Wan2.2 `models/vae_tiled.py`).
Deck: 17.88 fps on half a trn3, TP4 x CP8. **LingBot, trn3**: from `docs/perf/LOOP_ENGINEERING.md` (q20 flags).

| | Measured | Status |
|---|---|---|
| VAE on one core, eager (`vae_1rank_eager`) | 18.2 s (18.22 / 18.21) | measured |
| VAE split over 32 cores, eager | 2.03 s (9.0x) | measured |

LingBot version of the slide's VAE line: **18.2 -> 2.03 s** (one core -> one width tile per core, 32 cores).
| 17.88 fps re-run, q20 flags (`4x8_q20`) | **18.08 fps** (GEN 4.07 s, VAE 0.41 s); 17.88 confirmed within 1% | measured |

## S9a: Validate every step against the eager reference

Deck: step-0 within the bf16 floor; VAE 44.6 dB vs an fp32 decode. The 44.6 dB is Wan2.2.

LingBot (job 1): chunk-0 step-0 noise prediction vs an fp32 CPU forward (`perf/ref_fp32_c0s0.py`),
VAE frames vs an fp32 CPU decode of the same latents.

| Config | c0s0 rel-RMS vs fp32 | VAE vs fp32 decode |
|---|---|---|
| eager, native SDPA | 0.0129 | 56.4 dB |
| eager, attention_cte | 0.0340 | 56.3 dB |
| compiled blocks, native SDPA | 0.0123 | 56.4 dB |
| compiled blocks, attention_cte | 0.0343 | 56.3 dB |
| q20 | 0.0108 | 52.7 dB |
| today's `main` | 0.0137 | 52.7 dB |

For the slide: step-0 within ~1.1-1.4% of fp32; VAE **52.7 dB** vs an fp32 decode (deck: 44.6 dB, Wan2.2).
Note: an older memo said native SDPA was not a valid reference (0.33); on this image it is 0.0129.

## S9b: The journey, 76.6 -> 8.7 s

Deck: whole slide is Wan2.2-TI2V-5B. **Wrong model.** LingBot's version of this journey is the
rows of S3a, S4c, S5g and S7a above.

## S9c: LingBot-World v2, 25 fps on 16 devices, 18.4 on 8

Deck: 16 and 8 **trn2** devices. 16 devices is over budget and is not re-run.

LingBot, 8 trn3 chips: the journey, every row measured in job 1 (480x768, 81 frames):

| Step | GEN | VAE | Total | fps |
|---|---|---|---|---|
| eager, native SDPA | 19.59 s | 2.03 s | 21.6 s | 3.75 |
| + one compiled graph per DiT block | 6.07 s | 2.02 s | 8.09 s | 10.0 |
| + q20 flags (merged attention, SP, fused bf16 VAE, tuned kernel ...) | 4.07 s | 0.41 s | 4.48 s | 18.08 |
| today's `main` flags | 4.27 s | 0.41 s | 4.68 s | 17.3 |

Eager to q20: 4.8x. The deck's 4.40 s / 18.40 fps is trn2 with the softmax check; the trn3 check-on
number is not measured yet.

## S9d: A live stream

Deck: 17.3 fps, ~1.07 s key press to new frames, ~0.58 s per chunk; DiT + T5 on half a trn3 and the
fused VAE on a **separate** deployment (lbw-v2-vae).

LingBot, `serve.py` on 8 trn3 chips (`talk-live-deploy.yaml`: today's `main` flags, TP4 x CP8, 480x768),
VAE **inline on the same 8 chips** (a separate VAE deployment needs more than 8 chips). Client
`live_latency.py` on this Mac through `kubectl port-forward` (ClusterIP only), scene 03, presses
alternating left / right after 15 s. Two sessions:

| | Session 1 (8 presses) | Session 2 (6 presses) |
|---|---|---|
| Stream fps at the client | 11.12 | 11.82 |
| Key press -> first new frame, median | 2.026 s (2.008-2.630) | 2.046 s (1.961-2.081) |
| Server, per chunk: denoise start -> x0 (DiT) | | **0.554 s** |
| Server, per chunk: x0 -> decoded (VAE, inline) | | 0.276 s |
| Server, per chunk: decoded -> client | | 0.053 s |
| Server, chunk period | | 1.009 s |

What this validates: the DiT per chunk, **0.554 s vs the deck's ~0.58 s**. What it does not: 17.3 fps and
1.07 s need the VAE overlapped on other chips. Inline, DiT and VAE run one after the other (0.83 s of the
1.01 s period), so a press waits about two chunk periods (2.0 s). With the VAE off the DiT's chips the
period would be about the DiT's 0.55 s, and two periods ~1.1 s, which is consistent with the deck, but
that was not measured here.

---

# Backups re-measured on trn3 (8 chips)

Job 4, `talk-ladder-job.yaml`, branch `main`, same setup as above. Raw logs on the node in
`/opt/lingbot-world-v2/e2e/lbw-talk-ladder/`.

## B4h / B4i: LingBot VAE, step by step

Deck: 64 ranks on trn2: 3.01 / 1.36 / 0.80 / 0.55 / 0.33 / 0.31 s; 43.5 -> 56.6 dB. **TRN2.**

LingBot, 32 ranks on 8 trn3 chips, decode-only on the same latents (q20), each step adds to the row above.
dB = frames vs an fp32 CPU decode of the same latents.

| Step | VAE | vs fp32 decode |
|---|---|---|
| upstream decoder, per-conv eager (`v0_eager`) | 2.03 s | 56.34 dB |
| whole blocks compiled (`LINGBOT_VAE_FUSED=1`, `v1_fused`) | 1.38 s | 56.34 dB |
| bf16 convolutions (`v2_bf16`) | 0.86 s | 52.71 dB |
| -O1, norm as a matmul (`v3_o1mm`) | 0.65 s | 52.71 dB |
| 2-D tiles (`v4_2d`) | 0.46 s | 52.70 dB |
| 6 frames per step (`final`) | 0.41 s | 52.71 dB |

Differences from the deck: on trn3 the upstream decoder is already 56.3 dB (no 43.5 dB padding bug at
this tiling), and bf16 is what costs 3.6 dB (56.3 -> 52.7), not a gain. B4i's "~3 s -> 1.36 s" is
2.03 -> 1.38 s here.

## B6a / B6b / B6c / B6e: attention flags, one at a time

Each row is q20 with ONE flag turned off. GEN = last of 3 passes; c0s0 = chunk-0 step-0 vs fp32 CPU.

| Config | Flag off | GEN | fps | c0s0 vs fp32 | GEN cost of turning it off |
|---|---|---|---|---|---|
| `4x8_q20` (all on) | | 4.08 s | 18.03 | 0.0108 | |
| `4x8_nomerge` | `LINGBOT_CTE_MERGE` (+ ATTN_PAIR, ATTN_EARLY that need it) | 7.37 s | 10.42 | **0.0315** | +3.28 s |
| `4x8_nopair` | `LINGBOT_ATTN_PAIR` (two chunks per call) | 4.34 s | 17.06 | 0.0108 | +0.26 s |
| `4x8_novgroup` | `LINGBOT_CTE_VGROUP=8` (8 V tiles per DMA) | 4.20 s | 17.56 | 0.0131 | +0.12 s |
| `4x8_nokblk` | `LINGBOT_K_BLOCKED` (K in the all-gather's layout) | 4.32 s | 17.12 | 0.0109 | +0.24 s |

For the slides:
- **B6a** (copy before every call, zero keys in the softmax): reading the cache in place is worth
  3.28 s of GEN on 8 trn3 chips and cuts the error vs fp32 from 0.0315 to 0.0108. Deck: 8.95 -> 5.73 s (trn2, 16 devices).
- **B6b** (two chunks per call): 4.34 -> 4.08 s. Deck: 4.88 -> 4.48 s.
- **B6e** (V tiles per DMA, K layout): 4.20 -> 4.08 s and 4.32 -> 4.08 s. Deck: 3.94 -> 3.74 -> 3.66 s.
- q20 again in this job: 18.03 fps (18.08 in job 1): run-to-run spread 0.3%.

## B6d / B6g / S7a 18.64 / S9c 18.40: the softmax shortcut and its check

These come from branch `loop_engineering_8dev` (ad34d57), measured on trn2. **They do not run on trn3
as written**: every run from q21 on uses chained kernel calls (`LINGBOT_CTE_CHAIN=1`), and the kernel
asserts that path is not handled on trn3 (`kernels/attention_cte_lb.py:767`, `is_trn3_b1()`:
"chained calls: the Vector-engine exp path is not handled"). The shortcut itself is only switched on
together with chaining (`wan/modules/model_fast.py:1389` on that branch: `_c_off = ... if (_CTE_CHAIN and use_cmax)`),
and `LINGBOT_V_TOKEN_MAJOR` asserts without chaining (`model_fast.py:222`). Job 5 tried; every
shortcut config failed (logs on the node, `lbw-talk-8dev/`).

**So on trn3 today: B6d, B6g, S7a's 18.64 / 25.02 fps and S9c's 18.40 / 25.02 fps cannot be measured.**
They need a trn3 version of the chained-call path in `kernels/attention_cte_lb.py` first.
**The best trn3 number for S7a / S9c (8 chips):** that branch's 8-device flags with chaining off (so the
softmax shortcut is off too), `4x8_d8nochain` in `talk-8dev-job.yaml`:

| | GEN | VAE | Total | fps | c0s0 vs fp32 | VAE vs fp32 decode |
|---|---|---|---|---|---|---|
| `4x8_d8nochain` (passes 3-5: 3.684 / 3.681 / 3.699 s) | 3.70 s | 0.39 s | 4.09 s | **19.81** | 0.0148 | 52.69 dB |
| repeat, new process (passes 2-5: 3.685-3.676 s) | 3.68 s | 0.39 s | 4.07 s | **19.92** | | |

Repeat bitwise identical (deterministic: True). Pass 2 of the first process still compiled (7.9 s).
This beats q20 on `main` (18.08 fps) by 10% and is the number to put on S7a / S9c for 8 trn3 chips,
labelled "without the softmax shortcut".

## B6h: the check's runtime branch in the kernel

Code only. **Matches the source**: `kernels/attention_cmax2_lb.py:389-403` on `loop_engineering_8dev`
(`nisa.register_alloc` / `register_load` of `any_redo`, `_redo_any`, `_redo_one`, `_redo_block`, two
`nl.fori_loop` with the register as the trip count). Not on `main`. Because the guarded kernel only runs
on the chained path, this code cannot execute on trn3 today (see B6d).

## B6g: why one constant works in 34 of 40 layers

The row-max measurement (`LINGBOT_ROWMAX_GRAPH=1`) is also on the chained path only
(`wan/modules/model_fast.py:223` on `loop_engineering_8dev`), so it cannot run on trn3 either.
The slide's ranges (-8.3..71.5, 126-864) and timings (0.70 -> 0.36 ms, 3.30 -> 3.06 s) stay trn2 numbers.

## B7c: four layouts on one device -> LingBot layouts on 8 trn3 chips

Deck: Wan2.2 on one trn2 device, 29.4 -> 20.6 ms per block. **WAN.** LingBot's layout comparison, q20 flags,
`talk-b6-layouts-job.yaml` (32 ranks):

| Layout | GEN | VAE | fps | Status |
|---|---|---|---|---|
| **TP4 x CP8** (TP inside a chip) | 4.08 s | 0.42 s | **18.04** | measured |
| TP8 x CP4 (TP across two chips) | 4.40 s | 0.41 s | 16.84 | measured |
| TP2 x CP16 | 5.05 s | 0.41 s | 14.85 | measured |

Same order as the 2026-10-31 sweep (17.88 / 16.70 / 14.74). Keeping TP inside a chip wins (S7b's point).

## B6e: three changes to a copy of the kernel (rest)

| q20 with one flag off | GEN | fps | GEN cost of turning it off | Status |
|---|---|---|---|---|
| `LINGBOT_CROSS_NMAJOR` (cross-attention cache in the kernel's layout) | 4.12 s | 17.88 | +0.04 s (deck: -250 us/block) | measured |
| `LINGBOT_CTE_KV_DGE=none` (runtime-built DMA descriptors) | | | | running |

## B6f: one block now

Deck: 2.53-2.60 ms at TP4 x CP16 on trn2 (270 tokens per rank). On 8 trn3 chips each rank holds twice
the tokens (540), so the block is not comparable 1:1.

LingBot q20, TP4 x CP8, the steady block (`lbw-talk-b6-layouts/lnc2_4x8_q20/prof/pairs/rankr0-3645ms-dit-c535e56b`,
1280 calls, 2.85 ms per call in the system trace):

| | Measured |
|---|---|
| Block | **2.92 ms** (30.4% MFU) |
| Tensor busy | 64.5% |
| Every engine idle | 0.236 ms (8.1%) |
| Attention kernel (self + cross, `attention_cte_lb.py`) | 1.22 ms |
| Projections / FFN / norms (`model_fast.py`) | 1.09 ms |
| Attention merge / prep (`attention.py`) | 0.11 ms |
| Unattributed (collectives, DMA queues) | 1.23 ms (overlaps the above) |

The 8dev-flags block (19.9 fps config) is profiled in `talk-b6f-8dev-job.yaml`: running.

## B6a / B6b / B6e / B6f: per-block numbers (jobs talk-b6-nopair, talk-b6-nomerge, talk-b6f-8dev, talk-b6-layouts)

| Slide | Deck (trn2) | LingBot, 8 trn3 chips | Source |
|---|---|---|---|
| B6a: the copy inside the block | 1.7 of 10.4 ms | **14.4 of 17.1 ms** | `lbw-talk-b6-nomerge/.../rankr0-6681ms-dit-18ab0e4b` (+2% vs trace), `attention.py:440-459` |
| B6b: two chunks per call | 3.28 -> 2.88 ms per block | **3.11 -> 2.85 ms** (steady block, system trace) | nopair `rankr0-3987ms` vs q20 `rankr0-3645ms`, 1280 calls each |
| B6e: runtime-built DMA descriptors | 4.26 vs 3.94 s | **4.67 vs 4.08 s** GEN (slower) | `4x8_kvdge` (`LINGBOT_CTE_KV_DGE=none`), c0s0 0.0125 |
| B6f: one block now | 2.5 ms (TP4 x CP16) | **2.74 ms** (TP4 x CP8, 19.9 fps flags); Tensor 63.8%, all idle 8.4%; attention kernel 1.08, projections/FFN 1.02, merge 0.13 ms | `lbw-talk-b6f-8dev/.../rankr0-3470ms-dit-4b1f2b11` |

The 19.9 fps config re-ran in this job at 19.87 fps (GEN 3.686 s): third process, same number.

## S2c: one node, how the chips are linked

Deck: 16 chips in a 4 x 4 2-D torus, 4 neighbours per chip, chip 0 -> 1, 3, 4, 12. Its own note says this was
read off neuron-ls on a **trn2.48xlarge**.

trn3-dev1 (`neuron-ls`, CONNECTED DEVICES, printed by the talk jobs; rows 0-6, the log keeps 20 lines):

| Chip | Linked to |
|---|---|
| 0 | 1, 2, 4, 6, 8, 10, 12, 14 |
| 1 | 0, 3, 5, 7, 9, 11, 13, 15 |
| 2 | 3, 0, 4, 6, 8, 10, 12, 14 |
| 3 | 2, 1, 5, 7, 9, 11, 13, 15 |
| 4 | 5, 0, 2, 6, 8, 10, 12, 14 |
| 5 | 4, 1, 3, 7, 9, 11, 13, 15 |
| 6 | 7, 0, 2, 4, 8, 10, 12, 14 |

**8 linked chips per chip, not 4**: its partner (d xor 1) and every other chip of the same parity. 144 GB HBM
per chip and 4 logical cores per chip at LNC2 (8 physical) confirmed. Rows 7-15 were not printed; check them on
the box. The S2c AFTER slide draws this.
