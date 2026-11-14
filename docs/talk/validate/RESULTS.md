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
| Accuracy: chunk-0 step-0 vs fp32 CPU | | pending |

Eager with attention_cte (`4x8_eagercte`): DiT 24.3 s (median 0.71 s per forward), GEN 24.70 s, VAE 2.04 s,
total 26.7 s, 3.03 fps. Measured. In eager, attention_cte is slower than native SDPA.

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
| compiled blocks (fused) only (`vae_fused_only`) | | pending |
| final: fused, bf16, -O1, 2-D tiles (`vae_final`) | | pending |

## S5a / S5b / S5c / S5d: Profile of one compiled block, native SDPA

Deck: one block 56.1 ms; Tensor busy 39%, one engine busy 36%, all idle 16%. Three heads, 7,854
tokens. **Wrong model** (Wan2.2 `models/fast_dit.py`, trn2).

LingBot: profile of the `4x8_compile` config (`kiro/e2e/profile.sh`, pairs + engine overlap). Pending.

| | Measured | Status |
|---|---|---|
| One block, native SDPA | | pending |
| Tensor busy | | pending |
| Exactly one engine busy | | pending |
| Every engine idle | | pending |
| Is there a Tensor gap during the softmax? | | pending |

## S5f / S5g: attention_cte

Deck: self-attention 18.2 -> 5.3 ms per call; 16.2 -> 13.2 s per video. **Wrong model.**

LingBot: `4x8_compile` vs `4x8_compilecte`. Pending.

## S5h / S5i: The same block with the kernel

Deck: 56.1 -> 17.3 ms; idle 16% -> 1.1%; Tensor busy 39% -> 79%. **Wrong model.**

LingBot: needs a profile of `4x8_compilecte` (job 2, after job 1). Pending.

## S7a: The frame rate decides how many chips

Deck: 8 chips 18.64 fps, 16 chips 25.02 fps. **trn2**, not trn3. 16 chips is over the 8-chip budget
and is not re-run.

LingBot, today's `main` flags, 8 trn3 chips (`4x8_final`): pending.

## S7d: Sequence parallelism, two functional collectives

Deck: VAE 9.0 -> 6.0 s (one spatial tile per core). **Wrong model** (Wan2.2 `models/vae_tiled.py`).
Deck: 17.88 fps on half a trn3, TP4 x CP8. **LingBot, trn3**: from `kiro/LOOP_ENGINEERING.md` (q20 flags).

| | Measured | Status |
|---|---|---|
| VAE on one core, eager (`vae_1rank_eager`) | | pending |
| VAE split over 32 cores, eager | 2.03 s | measured |
| 17.88 fps re-run, q20 flags (`4x8_q20`) | | pending |

## S9a: Validate every step against the eager reference

Deck: step-0 within the bf16 floor; VAE 44.6 dB vs an fp32 decode. The 44.6 dB is Wan2.2.

LingBot, every config: chunk-0 step-0 rel-RMS vs an fp32 CPU forward, VAE frames vs an fp32 CPU
decode (end of job 1). Pending.

## S9b: The journey, 76.6 -> 8.7 s

Deck: whole slide is Wan2.2-TI2V-5B. **Wrong model.** LingBot's version of this journey is the
rows of S3a, S4c, S5g and S7a above.

## S9c: LingBot-World v2, 25 fps on 16 devices, 18.4 on 8

Deck: 16 and 8 **trn2** devices. 16 devices is over budget and is not re-run.

LingBot, 8 trn3 chips, today's `main` flags (`4x8_final`): pending.

## S9d: A live stream

Deck: 17.3 fps, ~1.07 s key press to new frames, ~0.58 s per chunk.

Needs the serving deployment. Measured last, within 8 chips. Pending.
