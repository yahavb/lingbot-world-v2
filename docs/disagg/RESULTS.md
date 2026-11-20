# DiT/VAE disaggregation: measured results (KEEP-trn3-dev1, 2026-11-19/20)

One live stream, scene 03, the scene's own camera path, 60 chunks (12 frames each), fps at the client
(perf/disagg/run_cell.sh). trn3-dev1, LNC2 (4 ranks per chip), TP4 x CP(chips), MXFP8 DiT, bf16 VAE. The same
DiT and VAE flags in the monolith and the split (perf/disagg/mk.py "mono" set; at 720p three attention
options that only handle 480 shapes are off on both sides). Split: VAE on the OTHER node, latents over EFA
with NIXL. Raw rows: perf/disagg/results/summary.txt and the per-cell CSVs.

## The core comparison (monolith N chips vs split N/2 DiT + VAE)

480 x 768:

| Monolith | fps | Split (DiT + VAE chips) | fps |
|---|---|---|---|
| 4 chips | 14.3 | 2+1, 2+2 (common flags) | 10.1, 10.1 |
| | | 2+1, 2+2 (tuned 2-chip flags) | 14.7, 14.6 |
| 8 chips | 20.0 | 4+1, 4+2, 4+4 | 17.5, 17.5, 17.5 |
| 16 chips | 23.4 | 8+1, 8+2, 8+4, 8+8 | 25.8, 26.0, 25.5, 25.6 |
| (none) | | 16+1, 16+2, 16+4 | 32.1, 32.2, 32.2 |

720p (runs on the 704 x 1280 grid):

| Monolith | fps | Split | fps |
|---|---|---|---|
| 4 chips | does not fit (device memory) | 4+x | does not fit (device memory) |
| 8 chips | does not fit (device memory once the inline VAE loads) | 8+1, 8+2, 8+4 | 8.5, 8.3, 8.4 |
| 16 chips | 10.0 | 16+1, 16+2, 16+4 | 12.7, 12.7, 12.2 |

## Per-phase times (true device time, LINGBOT_PHASE_SYNC=1, 480 x 768)

| Chips | DiT alone (ms/chunk) | DiT in the monolith | VAE inline (ms/chunk) | VAE share of DiT time |
|---|---|---|---|---|
| 2 | 957 | - | - | - |
| 4 | 552 | 552 | 151 | 27% |
| 8 | 372 | 371 | 133 | 36% |
| 16 | 299 | 300 | 143 | 48% |

VAE pod alone, decode per chunk (480): 1 chip 324 ms, 2 chips 213, 4 chips 173, 8 chips 153.

## What it shows
1. The monolith pays DiT + VAE for every chunk; the split pays max(DiT, VAE) because the VAE decodes chunk n
   on its own chips while the DiT does chunk n+1. In every split cell the fps equals the DiT's own pace, and
   the VAE size does not change it: a 1-chip VAE keeps up at both resolutions. Extra VAE chips buy latency
   (frames ready sooner after the DiT), not fps.
2. The DiT gets faster with chips; the VAE does not (flat at ~140 ms inline from 4 to 16 chips, 2.1x from 1 to
   8 chips as a pod). So the VAE's share of each chunk grows with the chip count (27% -> 48% at 480), and the
   monolith's returns diminish; the split's do not. That is where the split wins: at 16 chips (480: 8+1 on 9
   chips beats the 16-chip monolith, 25.8 vs 23.4 fps; 16+1 gives 1.37x) and at 720p (16+1 1.27x).
3. Below that, splitting the chip budget in half costs the DiT more than the VAE costs the monolith: at 8 chips
   (480) the monolith wins (20.0 vs 17.5).
4. Memory: at 720p the 8-chip DiT fits alone (split) but not with the VAE beside it (monolith).
5. At 720p the DiT grows faster than the VAE (attention over 10560 tokens a chunk vs 4320), so the VAE's
   share is smaller than at 480 (~33% at 16 chips); the win at 720p comes at the larger chip counts.

## Corrections made along the way (each invalidated earlier numbers)
- The probe computed image metrics on every frame (45 ms each), capping it at ~22 fps: now one frame a chunk.
- A VAE label from the previous round stayed in a DiT's pool: rounds now clear every label first.
- The monolith's first decode appended the VAE's compiler options (-O1 --model-type=unet-inference) to every
  later compile in the process, so the 16-chip monolith's DiT ran 2.1x slower (619 vs 299 ms). That looked
  like DiT/VAE interference; it was this bug. Fixed by vae_cc_scope (serve.py, vae2_1_fused.py). The earlier
  2-chip "co-located 676 ms vs split 519 ms" result probably had the same cause and must not be cited.

## Not yet measured
DiT-alone at 720p; 16+8 and 8+8 at 720p; several streams per VAE (fan-in); press-to-frame latency; device
profiles of each phase (engine utilization, memory traffic, collectives); same-node (NeuronLink) vs cross-node.
