# Long abstract (for collaborators; draft)

Every number below is measured on KEEP-trn3-dev1. The OSDI
submission text must be the authors' own; this draft is a pitch and an outline.

---

**[SYSTEM]: Role-Disaggregated Serving for Two-Stage Generative Pipelines (Operational Systems)**

Generative models for interactive video, world simulation for robotics, and long-form audio share one pipeline shape:
a diffusion transformer (DiT) denoises latents, and a learned decoder turns them into pixels or samples. Serving stacks
inherit the LLM playbook and run both stages co-located on the same accelerators, one after the other. We show that
this coupling, rather than either stage alone, limits these pipelines, and that it has a different cause than the one
behind prefill/decode disaggregation in LLM serving. There, a compute-bound phase interferes with a memory-bound one.
Here, device profiles on AWS Trainium3 show two stages with different binding resources: the DiT is bound by on-chip
data movement (DMA queues busy 82% of the time, moving 8.8x the data its HBM traffic needs), and the decoder by
systolic-array utilization (Tensor engine busy ~90% at 44-50% efficiency). They therefore scale differently with
chips, and a co-located replica charges each stage at the other's chip count and makes each wait for the other.

We formulate the cost of this coupling: from per-stage scaling curves, a co-located replica takes T_d(N) + T_v(N) per
unit of work, a disaggregated one max(T_d(n_d), T_v(n_v)), and one decoder replica can serve T_d / T_v DiT replicas.
This composition model predicts all 15 measured configurations of a 14B streaming world model within 1.4%, the
16-chip DiT from a fit on 2, 4 and 8 chips within 2.5%, and the number of streams a decoder sustains.

We build [SYSTEM] on Amazon EKS: roles (denoiser, decoder) are sized, replicated, placed and failed over
independently. Kubernetes DRA claims pair each Trainium device with its own EFA; NIXL moves each latent one-sided
from device memory to device memory in 0.17 ms; decoders keep per-session causal state so that one decoder serves many
streams; sessions are placed on the least-loaded decoder; a failed decoder costs a ~3 s switch with no lost chunks.

On LingBot-World v2 (14B, streaming video) and Stable Audio 3 Medium (long-form audio):
- At 16 chips, a 16-chip DiT plus a 1-chip decoder runs 1.20x the frame rate of the best co-located configuration
  (32.1 vs 26.7 fps); an 8-chip DiT plus a 1-chip decoder reaches 96% of the 16-chip co-located replica's frame rate on 9 chips.
- At 720p the co-located 8-chip replica does not fit in device memory; disaggregated, the same DiT runs; at 16 chips
  the split gains 1.27x.
- One 1-chip decoder serves 3 streams with no per-stream slowdown (427 MiB of state per session), and decoder replicas
  scale out: 3 decoders serve 9 streams on 21 chips at full speed.
- Rate matching works both ways: with a slow DiT one decoder feeds several, and with a fast, few-step DiT one DiT
  needs more decoder capacity (Stable Audio at 2 steps: 1.22x with a 2-chip decoder, a loss with a 1-chip one). Since
  steps, resolution and model are fixed per service, the sizing is a deploy-time decision.
- At real time (at least 16 fps per stream) disaggregation needs 4.5 chips per stream (two 4-chip DiTs sharing one
  1-chip decoder) against 8 co-located: 44% fewer chip-hours per stream-hour. With a 4-chip decoder, press-to-photon
  latency matches co-located at the median (1114 ms) and beats it at p90 (1371 vs 1416 ms).

We also report where disaggregation does not pay (splitting an 8-chip budget starves the DiT) and the operational
lessons of running it on Trainium, EKS, DRA, EFA and NIXL.
