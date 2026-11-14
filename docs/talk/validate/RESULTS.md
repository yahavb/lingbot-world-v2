# Talk deck vs LingBot-World v2: measured results, slide by slide

Deck: `~/Downloads/talk-deck_v2.html`. Each section below is one slide: what the deck says,
what LingBot measured, and where the number comes from.

Setup for every number here, unless a row says otherwise:
- repo `~/lingbot-world-v2`, commit e18d60f (`main`)
- cluster KEEP-trn3-dev1, node ip-192-168-192-229, one `xl-lnc2` claim (8 trn3 chips, 32 ranks, TP4 x CP8)
- 81 frames, 480x768, chunk 3, window 12, sink 1, 4 denoise steps, `examples/03`, seed 42
- image torch-neuronx 2.14.0 / SDK 2.33
- job `talk-pipeline-job.yaml` (kiro/k8s_sweep.sh); raw logs on the node in `/opt/lingbot-world-v2/e2e/lbw-talk-pipeline/<config>/`

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
| Tensor gaps > 50 us during attention | **6 gaps, 1.54 ms** (each ~0.3 ms) | 3 gaps, ~22 ms |
| Softmax work (Vector + Scalar) while Tensor idle | **2.32 ms** | |
| Attention on Vector (max, sum) / Scalar (exp) | **2.93 / 1.87 ms** | 12.8 / 7.7 ms |
| Source line of every attention instruction (S5b) | `wan/modules/attention.py:802`, `F.scaled_dot_product_attention` | |
| Whole pipeline in this mode | GEN 11.37 s, VAE 2.03 s, 13.4 s, 6.05 fps | 21.9 s |

So the deck's S5a-S5d story (Tensor waits for the softmax) **does reproduce on LingBot**, once the
compiler's automatic kernel is switched off: Vector and Scalar run the softmax in 6 stretches where
Tensor sits idle.

## S5f / S5g: attention_cte

Deck: self-attention 18.2 -> 5.3 ms per call; 16.2 -> 13.2 s per video. **Wrong model.**

LingBot, both compiled one graph per block, everything else off:

| Attention | GEN | VAE | Total | fps | Status |
|---|---|---|---|---|---|
| native SDPA in the code; the compiler swaps in attention_cte (`4x8_compile`) | 6.07 s | 2.02 s | 8.09 s | 10.0 | measured |
| attention_cte called by our code, `ATTENTION_BACKEND=cte` (`4x8_compilecte`) | 8.98 s | 2.02 s | 11.00 s | 7.37 | measured |

**Finding: on LingBot the deck's S5g claim is reversed.** Calling attention_cte ourselves is slower than
leaving native SDPA in the code (8.98 vs 6.07 s GEN), because on SDK 2.33 the compiler already uses
attention_cte for SDPA. Our `cte` path also copies every KV chunk into a zero-filled, 512-aligned
buffer before each call (`kiro/LOOP_ENGINEERING.md`, "Merged attention"). The LingBot win from the
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
The explicit-cte block profile (job 3, `talk-cte-prof-job.yaml`), to explain S5g's slower path: pending.

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

The 8-chip number for this slide on trn3 is 18.1 fps (q20). Today's `main` flags were tuned for 4 chips
and are 5% slower at 8. 16 chips not re-run (budget).

## S7d: Sequence parallelism, two functional collectives

Deck: VAE 9.0 -> 6.0 s (one spatial tile per core). **Wrong model** (Wan2.2 `models/vae_tiled.py`).
Deck: 17.88 fps on half a trn3, TP4 x CP8. **LingBot, trn3**: from `kiro/LOOP_ENGINEERING.md` (q20 flags).

| | Measured | Status |
|---|---|---|
| VAE on one core, eager (`vae_1rank_eager`) | 18.2 s (18.22 / 18.21) | measured |
| VAE split over 32 cores, eager | 2.03 s (9.0x) | measured |

LingBot version of the slide's VAE line: **18.2 -> 2.03 s** (one core -> one width tile per core, 32 cores).
| 17.88 fps re-run, q20 flags (`4x8_q20`) | **18.08 fps** (GEN 4.07 s, VAE 0.41 s); 17.88 confirmed within 1% | measured |

## S9a: Validate every step against the eager reference

Deck: step-0 within the bf16 floor; VAE 44.6 dB vs an fp32 decode. The 44.6 dB is Wan2.2.

LingBot (job 1): chunk-0 step-0 noise prediction vs an fp32 CPU forward (`kiro/ref_fp32_c0s0.py`),
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

Deck: 17.3 fps, ~1.07 s key press to new frames, ~0.58 s per chunk.

Needs the serving deployment. Measured last, within 8 chips. Pending.
