# Why split the DiT from the VAE? The research problem, distilled

Companion to RESEARCH_PLAN.md (design space, experiment list). This file states the one question, the
mechanisms that could answer it, the evidence we already have for each, and the experiment that isolates
each one. A number marked (measured) comes from a run on KEEP-trn3-dev1; everything else is a hypothesis.

## The question

A streaming video world model makes every chunk in two steps: a DiT denoises a latent chunk, then a VAE
decodes it to pixels. **Is it worth running the two on different chips, and if so, why?** "Why" means which
mechanisms produce the gain, how big each one is, and when the gain disappears.

The LLM analogue (prefill/decode disaggregation: DistServe, Splitwise, the NVIDIA study arXiv 2506.05508)
separates two phases of a *request*. Here the two phases alternate in *every chunk of a stream*, both keep
state across chunks, and the hand-off is 1.1 MB. So the LLM reasons do not carry over as they are; each one
has to be re-tested.

## One thing must be ruled out first

Disaggregation has a simpler rival: **co-located pipelining**. Keep one pod but run the VAE of chunk n on
some of its own cores while the DiT does chunk n+1. If that gets the same result, splitting into pods
adds nothing but network and operations. So every mechanism below is tested against three setups with the
same number of chips:

| Setup | What it is |
|---|---|
| A. Co-located, serial | today's monolith: DiT, then VAE, on the same cores |
| B. Co-located, pipelined | one pod; the VAE on dedicated cores of the same chips, overlapped with the DiT |
| C. Disaggregated | DiT pod + VAE pod(s), latent over NeuronLink (same node) or EFA (cross node), NIXL |

A paper that only compares A and C overstates the case. B is the baseline that makes it honest.

## The model (agreed with the user 2026-11-20): two compute-bound stages, coupled
Unlike LLM prefill (compute-bound) vs decode (memory/DMA-bound), the DiT and the VAE decoder are both expected
to be compute-bound (to verify with device profiles). The bottleneck of co-location is the coupling: every chip
runs the DiT, then every chip runs the VAE, so each stage is charged at the other's chip count.

T_i(n): time of stage i per unit of work (LingBot: a 12-frame chunk; SA3: a window) on n chips;
e_i(n) = T_i(1) / (n T_i(n)) its parallel efficiency.

- Co-located on N chips: period P_co = T_d(N) + T_v(N); chip-seconds per unit C_co = N T_d(N) + N T_v(N).
- Disaggregated, n_d DiT chips + n_v VAE chips, replicas rate-matched across roles:
  P_dis = max(T_d(n_d), T_v(n_v)) (+ hand-off, measured 0.17 ms); C_dis = n_d T_d(n_d) + n_v T_v(n_v).
- Throughput-per-chip gain = C_co / C_dis, from two terms: (1) parallelism mismatch, each stage at its own
  efficient n instead of the shared N; (2) serialization, neither stage's compute waits on the other's.
- Rate matching: if T_v < T_d, one VAE replica serves k = floor(T_d / T_v) DiT replicas; if T_v > T_d, several VAE
  replicas serve one DiT (SA3: decode chunks are independent; LingBot: per-stream causal VAE state, so across
  streams or by handing the state over).

Measured so far (480 / 380 s windows): LingBot DiT 2/4/8/16 chips 957/552/372/299 ms (e vs 2 chips 0.87, 0.64,
0.40); LingBot VAE pod 1/2/4/8 chips 324/213/173/153 ms (e 0.76, 0.47, 0.26), inline on 16 chips 143 ms (e 0.14);
SA3 DiT 1/2 chips 5.8/4.05 s (e 0.72; TP16 does not shard); SA3 decode 1/2 chips 3.07/1.61 s (e 0.95).
Example, LingBot at 16 chips: the VAE costs 16 x 143 = 2.29 chip-s per chunk co-located vs 0.32 on its own chip;
per chunk 7.1 chip-s co-located vs 3.3 for the 8 + 1 split. The two models fail differently: LingBot's VAE does
not scale (co-location wastes chips on it); SA3's DiT does not scale past 2 chips (co-location cannot add chips).
VAE busy fraction with one VAE chip: LingBot ~70% of the 8-chip DiT period (324 of 465 ms), ~87% at 16 chips;
SA3 ~70% (0.56 of 0.8 s a window): at these settings one VAE cannot serve two DiTs.

### Experiments to confirm it
1. Device profiles of a DiT block and a VAE conv NEFF: Tensor-engine busy, achieved vs peak TFLOPs, DMA time
   (is each stage compute-bound?).
2. Efficiency curves: VAE pods 1-16 chips, DiT at every chip count that runs, both sizes; fit serial fraction +
   communication term.
3. Device timelines of one co-located and one split period (engine idle time per stage).
4. Move T_d / T_v (SA3 steps 8 / 25 / 50, LingBot steps 4 / 8, CFG) and check C_co / C_dis against measured
   throughput per chip, including rate-matched N:M replicas.
5. Co-located with devices partitioned inside one process group (separates term 1 from per-role scaling).
6. Extrapolation: a roofline cost model per stage (FLOPs, bytes, collective volume per unit) fitted to 1-4,
   validated on held-out cells (fit 480 -> predict 720p; fit 380 s -> predict 190 s), then projected to larger
   DiTs and chip counts.

## The mechanisms (candidate reasons)

### M1. Overlap: the VAE leaves the critical path
Serial: time per chunk = T_dit + T_vae. Overlapped: max(T_dit, T_vae). Both B and C get this.
- Evidence (measured): 2 chips co-located 11.53 fps; 2 + 1 chips split 14.65-14.75 fps.
- Open gap (measured): in the split, the DiT takes 519 ms per chunk, which allows 23 fps, but the stream runs
  at about 14.7 fps (816 ms per 12 frames). Something other than the DiT and the VAE sets the pace. It has to
  be explained before any claim about overlap: host scheduling, the KV-cache commit step, broadcasts.
- Test: per-chunk timeline (DiT forward, commit, hand-off, VAE, frames out) for A, B, C.

### M2. Interference: the DiT runs faster when the VAE is not beside it
- Evidence (measured): the DiT's own time per chunk is 676 ms co-located and 519 ms split, on the same 2 chips
  and the same graphs. That 23% is not overlap; it is the DiT itself slowing down when the VAE shares its
  chips. Candidates: switching between the two programs (NEFF loads), HBM and allocator pressure, cache
  thrash, the VAE's collectives on the DiT's rank group.
- Why it matters: B (same chips, other cores) may keep this interference; C removes it. This is the
  mechanism most likely to separate B from C, and the closest analogue to DistServe's prefill/decode
  interference.
- Test: DiT-only time per chunk in A, B, C; neuron-profile of the DiT forward with and without the VAE
  resident; HBM in use; NEFF switch counts.

### M3. Memory headroom: the DiT fills the chips
- Evidence (measured): with the DiT loaded, the VAE-encode of a new start image (needed to re-anchor a long
  stream) failed with out-of-memory on the DiT chips (2026-11-19, re-anchor test). Scenes only work today
  because their images are encoded at startup, before the DiT loads.
- Why it matters: features that need the VAE at run time (re-anchoring, new start images, a larger decode
  window, higher resolution) do not fit co-located. A VAE pod with its own memory allows them. This is a
  capability reason, not a speed reason.
- Test: largest decode resolution and longest decode window that fit in A/B vs C; whether re-anchoring
  works (encode on the VAE pod).

### M4. Each side gets its own parallelism
The DiT wants tensor x sequence parallel over many ranks; the VAE wants a few spatial tiles.
- Evidence (measured): co-located, the VAE has to run on the DiT's 8 ranks; one tiling (4 x 2) decoded black
  frames, and 2 x 4 needed a special layout. Split, the VAE uses 4 ranks with 2 x 2 tiles.
- Test: VAE time per chunk vs number of ranks (1, 2, 4, 8) and tiling; how much of it is halo exchange.

### M5. Rate matching: one VAE can feed several streams
- Estimate: in the split, the VAE is busy about 202 ms of every 816 ms chunk (~25%), so one VAE chip could
  serve 3-4 DiT streams. Co-located, each stream carries a VAE that sits idle most of the time.
- Evidence (measured, cost side): 4-chip monolith 14.1 fps vs 2 + 1 split 14.7 fps: the same speed on 25%
  fewer chips (4.9 vs 3.5 fps per chip).
- Test: k DiT streams : 1 VAE for k = 1..4: fps per stream, VAE utilization, chips per stream, cost per
  stream-hour. This is the Pareto chart (fps per stream vs streams per chip).

### M6. Heterogeneity: the VAE does not need the newest chip
- Test: VAE on trn2 (or a part of a trn3) with the DiT on trn3: fps and cost per stream-hour.

### M7. Failure isolation and elasticity
- Evidence (measured): a VAE pod deleted or its device tainted mid-stream: switch in about 3 s, 140/140
  chunks, no pause over 1.5 s. Co-located, any fault restarts the whole pod (about 15 minutes with warmup).
- Test: failure timeline (fps over time with fault events); scaling the VAE pool 0 -> N under load.

### M8. Latency for control (robotics, games)
Splitting does not make one chunk faster: it is still DiT, then VAE. What can change is the wait: a key press
waits for the current DiT step, not the current DiT + VAE step. But the hand-off adds a hop (0.17 ms over EFA,
measured), and a deeper pipeline can add a chunk of delay.
- Test: scripted key presses, press-to-first-changed-frame, split into wait / DiT / hand-off / VAE / network
  / client buffer, as a CDF over 100 presses, for A, B, C.
- This could come out against splitting. Either result goes in the paper.

## Where splitting should not pay (to map, not to hide)
- One stream with chips to spare: the 4-chip monolith is about as fast as the split.
- A VAE that is a small share of the chunk time (low resolution, many denoise steps).
- The operating cost of two services (discovery, failover, a transport).

## What the paper would claim, if the data holds
1. For streaming diffusion, the gain from splitting comes from interference and memory (M2, M3) and rate
   matching (M5), not only from overlap (M1), which co-located pipelining also gets.
2. The hand-off is so small that placement is nearly free: same node over NeuronLink or cross node over EFA.
   That makes fan-in and mixed chip types practical in a way they are not for LLM KV-cache transfer.
3. The Trainium stack to do it (EKS, DRA claims with paired EFA, NIXL on Neuron memory) and what broke along
   the way.

## Use cases each metric speaks to
- Games and interactive worlds: fps per stream, press-to-frame latency.
- Robotics (world model as a closed-loop simulator; many parallel rollouts for policy training): latency
  bound, streams per chip, cost per stream-hour.
- Synthetic data (long offline rollouts): throughput per chip, quality over minutes.

## Order of work (each step can change the next)
1. Explain the 519 ms vs 816 ms gap (M1). Without it, the overlap numbers mean little.
2. Build setup B, then measure M2 (DiT time per chunk in A, B, C).
3. M5 fan-in and the Pareto chart; M3 memory limits.
4. M8 latency; M6 trn2; M7 failure timeline; placement (NeuronLink vs EFA).

## The core experiment (agreed with the user 2026-11-19): is disaggregation advantageous?
Each monolith size is compared with a split whose DiT has HALF the monolith's chips, plus a VAE of 1, 2, 4 (or 8)
chips. The split wins if it gets the same fps on fewer chips (cheaper) or more fps on the same chips (faster).

| Monolith (DiT + VAE inline) | Split: DiT chips + VAE chips (total) |
|---|---|
| 4 chips | 2 + 1 (3), 2 + 2 (4) |
| 8 chips | 4 + 1 (5), 4 + 2 (6), 4 + 4 (8) |
| 16 chips | 8 + 1 (9), 8 + 2 (10), 8 + 4 (12), 8 + 8 (16) |
| none (the monolith stops scaling) | 16 + 1, 2, 4, 8 (VAE on the second node, over EFA) |

Then the same sweep at 720p (agreed 2026-11-19): ~2.5x the pixels per chunk pushes the VAE decode, so the
scaling gap between the phases should grow; it is the second proof of the mechanism.

Rules: one stream, 480x768 first (then 720p), the same DiT and VAE flags in the monolith and the split (the
common set, plus the tuned 2-chip flags as a second data point at 4 chips), 60 chunks, the scene's own camera
path, fps measured at the client (perf/disagg/run_cell.sh, results in perf/disagg/results/summary.txt).

## Scaling matrix (user request 2026-11-19): does the split keep winning as chips grow?
Equal chip budgets, monolith (DiT + VAE inline, TP4 x CP n) vs split (DiT pod + VAE capacity):

| Budget | Monolith | Split options |
|---|---|---|
| 3-4 chips | 4 (TP4xCP4) | 2 DiT + 1 VAE, 2 + 2 |
| 8 chips | 8 (TP4xCP8) | 4 + 1, 4 + 2, 6 + 2 |
| 16 chips | 16 (TP4xCP16) | 8 + 1, 8 + 2, 8 + 4, 12 + 4 |
| 17-24 chips (2 nodes) | not possible (one node max; cross-node inline VAE) | 16 + 1, 2, 4, 8 (VAE on the 2nd node over EFA) |

Each at 480x768 and 720p (if it compiles), single stream and N streams (xDiT : yVAE, the xPyD analogue).
Outputs per cell: fps per stream, press-to-frame latency, fps per chip, streams per chip at >= 16 fps.

Design point found while planning: the VAE is causal and keeps a temporal cache per stream, so one stream's
chunks cannot be spread round-robin over several VAE pods. "More VAE" for one stream means a bigger VAE
pod (more chips, more tiles) or handing the causal cache between pods; several VAE pods help across
streams (fan-in/fan-out). Both are measured; the cache hand-off is a candidate new contribution.
