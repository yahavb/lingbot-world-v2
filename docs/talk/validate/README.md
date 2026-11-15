# Talk deck numbers, re-measured on LingBot-World v2

Deck: `~/Downloads/talk-deck_v2.html`. Only this repo, only cluster KEEP-trn3-dev1 (trn3),
at most 8 chips (one `xl-lnc2` claim: 32 ranks, TP4 x CP8). Spec: 81 frames, 480x768,
chunk 3, window 12, sink 1, 4 denoise steps, `examples/03`, seed 42.

## Why

The deck's S3a, S4c, S4e, S5a-S5j, the 9.0 -> 6.0 s on S7d, and S9b come from
Wan2.2-TI2V-5B on one trn2 device (`models/fast_dit.py`, 30 blocks). That is not LingBot.
The 16-device rows on S9c, and the 8-device 18.64 and 18.40 fps on S7a and S9c, are trn2.
Every number below is re-measured on LingBot's own T5 encoder, 40-block DiT and VAE.

## Job 1: `talk-pipeline-job.yaml` (perf/k8s_sweep.sh, one pod)

| Config | What it is | Slides |
|---|---|---|
| `4x8_eager` | nothing compiled, native SDPA, every loop flag off, eager fp32 VAE | S3a |
| `4x8_eagercte` | same, with nkilib attention_cte in eager | S3a (reference accuracy), S9a |
| `4x8_compile` | one compiled graph per DiT block, native SDPA; profiled | S4c, S5a, S5d |
| `4x8_compilecte` | the same, with attention_cte | S5f, S5g |
| `4x8_q20` | the q20 flags that measured 17.88 fps | S7d |
| `4x8_final` | today's `main` flags (`lbw-bench-job.yaml`) | S7a, S9c |
| VAE `eager` / `fused_only` / `final` | decode-only A/B on the same latents | S4e |
| `vae_1rank_eager` | eager VAE on one core | S7d (VAE split over cores) |

Results land on the node in `/opt/lingbot-world-v2/e2e/lbw-talk-pipeline/`.

## Results

Slide by slide in `RESULTS.md`.
