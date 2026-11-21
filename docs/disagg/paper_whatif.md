# WHAT-IF DRAFT, not for submission

This is a structural mock-up of the paper, written as if every hypothesis in PROBLEM.md holds. Its purpose is to
show the shape of the argument and where each experiment lands.

- **[M]** marks a number measured on KEEP-trn3-dev1 (docs/disagg/RESULTS.md).
- **[H]** marks a hypothesis or projection that must still be measured. Replace it with data or delete it.
- The OSDI '27 rules require the submitted text to be written by the authors (AI-generated papers are treated as
  fraud), and require disclosing the use of coding agents in the Implementation section. Use this draft as an
  outline and a list of claims to prove, not as text to submit.
- The system needs a name not used anywhere outside peer review; it is written [SYSTEM] below.

---

# [SYSTEM]: Role-Disaggregated Serving for Two-Stage Generative Pipelines (Operational Systems)

## Abstract

Diffusion transformers (DiTs) for video, world models and audio share a two-stage shape: a DiT denoises latents,
and a learned decoder (a VAE) turns them into pixels or samples. Today both stages run co-located on the same
accelerators, one after the other. We show that this coupling, not a lack of compute, is what limits throughput
per chip. Unlike prefill and decode in LLM serving, both stages are compute-bound [H, profiles], and they scale
differently with the number of chips. On shared chips, each stage is charged at the other's chip count, and each
stage's engines wait on the other's.

We formulate the cost of the coupling and show when disaggregating the pipeline into independently sized and
replicated roles pays, and by how much. We build [SYSTEM] on Amazon EKS with Trainium: DRA claims that pair each
device with its own EFA, NIXL one-sided writes from device memory to device memory, per-role pod discovery and
failover. On a 14B streaming world model (LingBot-World v2) and a text-to-audio model (Stable Audio 3 Medium):
- [SYSTEM] reaches 1.37x [M] the frame rate of the best 16-chip co-located configuration at 480x768, on 17 chips.
- It matches it on 9 chips [M].
- It serves 720p where the co-located 8-chip configuration does not fit in device memory [M].
- With rate-matched role replicas it raises streams per chip by [H: x] and cuts the cost per stream-hour by [H: y].

A cost model fitted to per-stage scaling curves predicts the measured gains within [H: z]% and projects them to
larger DiTs.

## 1 Introduction

Generative pipelines that produce media in real time are becoming systems workloads:
- interactive world models for games and simulation;
- robot policies trained against video world models;
- long-form audio.

Their serving stacks inherit the LLM playbook: tensor and sequence parallelism for the big model, everything
co-located in one replica. That playbook was built for one dominant stage.

These pipelines have two, and both matter:
- At 480x768 on 16 Trainium3 chips, the VAE decoder takes 143 ms per chunk next to the DiT's 300 ms [M].
- In Stable Audio 3, decode is 40–53% of the DiT's time [M].

**Prefill/decode disaggregation does not explain these pipelines.** Its premise is a compute-bound phase next to a
memory-bound one that interfere. Here both stages are compute-bound (Section 3) [H]. The problem is coupling: a
co-located replica runs each stage on all of its chips, in sequence. The two stages scale differently:
- LingBot's DiT keeps gaining from more chips, while its VAE stays nearly flat (efficiency 0.14 at 16 chips [M]).
- SA3's DiT stops sharding at two chips, while its decode scales almost linearly (efficiency 0.95 [M]).

Any shared chip count is wrong for one of the two stages.

**Contributions:**
1. **A cost model of stage coupling.** It defines the co-located and disaggregated chip-time per unit of work
   (C_co, C_dis) from per-stage scaling curves. It splits the gain into a parallelism-mismatch term and a
   serialization term, and gives the rate-matching ratio between role replicas (Section 3).
2. **Role disaggregation as an operating-system abstraction.** A pipeline is a set of roles. Each role is sized,
   replicated, placed and failed over on its own. State that crosses roles is explicit (Section 4).
3. **[SYSTEM], an implementation on EKS and Trainium.** It provides:
   - DRA claims with paired EFA;
   - NIXL one-sided transfers between device memories, with per-transfer failure isolation;
   - per-session role discovery and failover;
   - a sizing planner driven by the cost model (Section 5).
4. **An evaluation on two pipelines** across chip budgets, resolutions or window lengths, denoising steps, and N:M
   replica ratios. It includes the cases where disaggregation loses (Section 6).

## 2 Background and motivation

**2.1 Two-stage generative pipelines.**
- **Streaming video world model.** It works chunk by chunk: 3 latent frames become 12 video frames. A causal DiT
  with a rolling KV cache does 4 denoising steps per chunk. A causal 3-D convolutional VAE keeps its own temporal
  cache.
- **Text-to-audio.** It generates 380 s windows. Each window is inpainted from the previous window's last 60 s,
  which must be decoded and re-encoded before the next window starts.
- **The hand-off between stages is small.** A 1.1 MB latent per video chunk [M], and a few MB per audio window.

**2.2 Why the LLM argument does not carry over.**
- Prefill is compute-bound; decode is bound by HBM bandwidth (on Trainium, by its DMA engines).
- Disaggregating them isolates two different resource profiles.
- Our two stages have the same profile: Tensor-engine-bound [H, profiles]. They do not interfere through memory
  bandwidth. They interfere through time and chip count.
- A correction we made along the way: an apparent 2x DiT slowdown in the co-located 16-chip configuration was a
  compiler-option leak from the VAE into the DiT's graphs. Fixed, the DiT runs at the same speed co-located and
  alone [M]. There is no interference in the LLM sense.

**2.3 Motivation, measured.** Figure 1 plots fps against chips for co-located and disaggregated serving.
- Co-located: 14.3, 20.0 and 23.4 fps at 4, 8 and 16 chips [M].
- Disaggregated: 25.8 fps at 8+1 chips and 32.2 fps at 16+1 [M].
- The co-located curve flattens because the VAE's share of each chunk grows from 27% to 48% of the DiT step [M].

## 3 The cost of coupling

T_i(n) is the time of stage i on one unit of work (a chunk or a window) on n chips. e_i(n) = T_i(1) / (n T_i(n))
is its parallel efficiency.

- **Co-located on N chips.** P_co = T_d(N) + T_v(N), and C_co = N (T_d(N) + T_v(N)) chip-seconds per unit.
- **Disaggregated, n_d + n_v chips, rate-matched.** P_dis = max(T_d(n_d), T_v(n_v)) + ε, where ε is the hand-off
  (0.17 ms over EFA [M]). C_dis = n_d T_d(n_d) + n_v T_v(n_v).

**Gain.** Throughput per chip improves by G = C_co / C_dis. It factors into:
- (i) **mismatch:** N T_v(N) / (n_v T_v(n_v)) for the decoder, and the same for the DiT. Each stage runs at its own
  efficient chip count.
- (ii) **serialization:** the latency of one unit drops from T_d + T_v to max(T_d, T_v). The per-stream speedup is
  bounded by 2x, which it reaches when T_d = T_v.

**Rate matching.** A role-disaggregated deployment chooses replica counts R_d and R_v so that R_d / T_d(n_d) equals
R_v / T_v(n_v):
- If T_v < T_d, one decoder replica serves k = floor(T_d / T_v) DiT replicas.
- If T_v > T_d, several decoder replicas serve one DiT. This applies directly when decode units are independent
  (audio chunks). With per-stream causal state (video), it scales across streams or by migrating that state.

**When it pays, and when it does not.** Disaggregation loses when:
- splitting a fixed budget starves the stage that scales: LingBot at 8 chips, 20.0 fps co-located vs 17.5 split
  [M];
- the decoder share is small.

The model predicts both boundaries [H: validated in Section 6.5].

**Worked example (measured, LingBot 480, 16 chips).**
- The decoder costs 16 × 143 ms = 2.29 chip-seconds per chunk co-located, against 0.32 on its own chip [M].
- A whole chunk costs 7.1 chip-seconds co-located, against 3.3 for 8+1 [M].

## 4 Role disaggregation as an OS abstraction

A pipeline is a directed set of **roles** (conditioner, denoiser, decoder) connected by **typed hand-offs**. The
operating layer owns five things:

1. **Sizing.** The chips per replica for each role, from its efficiency curve.
2. **Replication and rate matching.** Replicas per role, from the cost model, at any N:M ratio.
3. **Placement.** Same node (NeuronLink) or across nodes (EFA). The hand-off is small enough that placement is
   nearly free [M: 0.17 ms per 1.1 MB write].
4. **State.** Which role state is per stream (KV cache, causal decoder cache), and what migrates on failover.
5. **Failure domains.** A decoder replica fails independently. In our measurements a decoder failure cost a ~3 s
   switch with 140/140 chunks delivered [M], where a co-located replica restarts entirely (minutes).

This is the scheduling problem an OS solves for process pipelines, applied to accelerator roles. The open question
is the policy: how many replicas of each role, and where.

## 5 [SYSTEM]: architecture and implementation

**Resources.**
- Kubernetes DRA claims allocate Neuron devices. Each device is paired with its own EFA through devicegroup1_id.
- Two-device claims take directly linked pairs (same topology_x). Any other pair fails collective setup [M].
- Bigger claims are count requests inside a device group. Per-device pairing constraints at 16 devices timed out
  the scheduler [M].

**Transport.**
- NIXL (LIBFABRIC backend, FI_HMEM_NEURON) registers Neuron device memory by its runtime virtual address
  (nrt_tensor_get_va).
- The DiT writes each unit one-sided into one of 4 decoder slots, with a per-unit notification.
- A dead peer fails only the transfer in flight. Connections to dead peers are parked rather than removed, because
  removal propagated a cancellation into the next peer's first write [M].

**Discovery and failover.**
- The denoiser picks a decoder per session from the Ready decoder pods.
- On a failed write it sets that decoder aside, then retries the unit on another.
- With no decoder available it refuses the session (503).

**Planner [H].** It takes the measured efficiency curves and the target (fps, latency, budget), and outputs
n_d, n_v, R_d, R_v and placement.

**Implementation.** [Lines of code; the components above; Kubernetes manifests.] Disclosure: coding agents were
used in building and measuring the system; the authors designed the system and the experiments and wrote this
paper.

## 6 Evaluation

Setup: Trainium3 trn3-dev1 (16 chips per node, 2 nodes), Neuron SDK 2.33, LNC2, MXFP8 DiT, bf16 decoder. The same
model flags in every configuration. One stream unless stated otherwise.

**6.1 Are both stages compute-bound?** [H] Device profiles of the dominant DiT and decoder graphs: MFU,
Tensor/Vector/DMA busy time. Expected: both Tensor-dominated, with DMA well below its limit. This contrasts with
the LLM decode stage.

**6.2 Per-stage scaling.**
- Measured efficiency curves: Table [M]; LingBot at 480 and 720p, SA3 at 380 s.
- [H] Fit each to a serial fraction plus a communication term.

**6.3 Equal-budget comparison (the core matrix).**
- 480: Table [M], as in RESULTS.md. Disaggregation wins at 16 chips and loses at 8.
- 720p: co-located 8 chips does not fit [M]; 16+1 gives 1.27x [M].
- SA3: monolith 2 chips 6.15 s, 2+1 split 5.40 s, for 1800 s of audio [M].

**6.3b Resolution: memory, not only speed.** At 720p (704x1280 grid, 10560 tokens a chunk vs 4320) [M]:
co-located 8 chips does not fit in device memory once the decoder loads, while the 8-chip DiT alone fits and runs at
8.5 fps split [M]; co-located 4 chips does not fit either way [M]. At 16 chips the split gains 1.27x (16+1: 12.7 vs
10.0 fps) [M], less than at 480 (1.37x), because attention grows with the square of the tokens and the decoder's share
falls (~33% of the DiT step vs 48%) [M]. So at high resolution disaggregation decides whether a chip budget can serve
the model at all (the memory term); the speed term returns at larger DiT chip counts, where the decoder's flat time
again dominates [H: 720p, 16-chip DiT + decoders on the second node vs the cost model's co-located 32-chip estimate].

**6.4 Moving the stage ratio.**
- [H] SA3 at 8, 25 and 50 steps; LingBot at 4 and 8 steps; CFG on and off.
- The gain follows (T_d + T_v) / max(T_d, T_v) toward 2x per stream at T_d ≈ T_v.

**6.5 Model accuracy and extrapolation.** [M] From per-stage measurements alone (DiT alone, VAE pod, inline VAE),
monolith = P_d(N) + T_v(N) and split = max(P_d(n_d), T_v(n_v)) predict all 15 measured LingBot 480 cells within 1.4%
(3 monoliths, 12 splits); the 720p inline VAE implied at 16 chips is 258 ms vs 254 measured; streams per VAE
P_d / T_v predicts 3.7 and 2.1 vs 3 and 2 run without slowdown. Scaling extrapolation: P_d(n) = a/n + b fitted on 2, 4, 8
chips predicts 334 ms at 16 vs 375 measured (12%); with a communication term, P_d(n) = a/n + b + c log2 n fitted on 2, 4,
8 predicts 385 ms at 16 (+2.5%) [M]. The VAE pod curve fitted on 1, 2, 4 chips predicts 168 ms at 8 vs 153 (+10%):
conservative, it overstates the decoder's cost [M] (perf/disagg/model_check.py).
- [H] Fit on 480 and predict 720p; fit on 380 s windows and predict 190 s. Prediction error within z%.
- [H] Project to larger DiTs (FLOPs scale with parameters × tokens, the decoder with pixels or samples).

**6.6 Rate-matched replicas.** [H] N:M = 1:1, 2:1, 3:1 (and 1:2 for audio). Streams per chip at ≥ 16 fps, and cost
per stream-hour: the Pareto frontier of co-located against disaggregated.

**6.7 Latency.** [M, LingBot 480, press-to-photon over 100 presses, median / p90]: 4-chip monolith 1114 / 1416 ms
(15.5 fps); 4 DiT + 1 VAE chip 1221 / 1471 (17.5 fps; decode of the chunk's first frame 314 vs 62 ms inline); 4 DiT +
4 VAE chips 1114 / 1371 (17.6 fps; decode 209 ms). A decoder sized for latency matches the monolith's median, beats its
p90 and keeps the throughput gain. **6.7b In-pod partitioned baseline** [M]: DiT and VAE containers on disjoint chips of
ONE pod (NIXL inside the pod) give 17.51 fps at 4 + 1 and 25.74 at 8 + 1, the same as across pods (17.54, 25.76): the
gain is the decoupling of the stages; separate pods cost nothing and add per-role scaling, sharing and placement.
**Previous 6.7 text:** [H] Press-to-photon latency (wait, DiT, hand-off, decoder, network, client buffer), as a CDF over
100 presses. Extra decoder chips lower the decode time per unit: 391 → 237 ms from 1 to 4 chips [M].

**6.8 Resilience.** Decoder pod deletion and device-fault taint: ~3 s switch, 140/140 chunks [M]. Scaling the
decoder pool 0 → N under load [H].

## Claims, considerations and constraints (added 2026-11-20, from the SA3 step sweeps and N:M work)

Claims
- C1 Rate matching works in both directions. Slow DiT relative to the decoder: one decoder replica serves several DiT
  replicas (N:1). [M LingBot, one 1-chip VAE, LINGBOT_VAE_SESSIONS=4: 1/2/3 streams of 2-chip DiTs 10.07 / 10.06 / 10.08
  fps each; 1/2 streams of 4-chip DiTs 17.52 / 17.53 fps each; frames byte-identical alone vs shared; only the decode
  latency grows with queueing, 486 -> 723 -> 833 ms.] [H: SA3 one VAE for 1/2/4/8 DiT jobs at 25 steps; running].
  Fast DiT relative to the decoder: one DiT needs more decoder capacity (bigger or more decoder replicas) [M: SA3 at 2
  steps, 1800 s audio: monolith 3.06 s, 2 DiT + 1 VAE chip 3.75 s (VAE-bound, loses), 2 + 2 VAE chips 2.51 s (1.22x)].
- C2 The binding side is predicted by T_d vs T_v, which the step count moves [M: SA3 2-chip DiT, 1800 s: 8 steps
  6.15 -> 5.48 s with 1 VAE chip; 4 steps 4.07 -> 3.89 s (1 chip) / 3.32 s (2 chips); 25 steps 14.88 -> 14.39 s;
  50 steps 27.81 -> 27.02 s]. Few-step distilled DiTs (SA3 8, LingBot 4) make the decoder a larger share: the trend in
  the field raises the value of disaggregation.
- C3 Memory term of N:M: the decoder's weights are small next to a chip's HBM (Wan2.1 VAE ~0.25 GB bf16, estimate), so
  one decoder chip holds M sessions' causal caches while it has compute to spare; the DiT fills its chips with weights +
  KV cache (720p co-located 8 chips does not fit [M]) [M: 427 MiB causal cache per session per rank; 4 sessions 1.7 GiB; the VAE rank's device use ~3 GiB].

- C4 Horizontal scaling per role (the OS claim): one decoder replica serves about T_d / T_v DiT streams (LingBot 2-chip
  DiT ~3.7, measured 3 without slowdown; 4-chip ~2, measured 2; SA3 25 steps ~4.5, measured 4, 6 saturate [M]); beyond
  that ratio the operating layer adds decoder replicas and spreads sessions over them (least-loaded placement), keeping
  the ratio while streams scale out; DiT and decoder pools grow independently [M: 2 one-chip VAEs (LINGBOT_VAE_SESSIONS=4) x 6 streams of 2-chip DiTs, LINGBOT_VAE_LEAST_LOADED=1: placed 3 + 3,
  every stream 10.01-10.05 fps vs 10.07 alone; 14 chips for 6 streams] [H: 3 x 9].

Considerations
- K1 Sizing is a deploy-time decision. Steps, resolution / window length, chunk size and model are fixed per service
  before serving, so T_d and T_v are known in advance and the planner picks n_d, n_v and the N:M ratio from measured
  per-stage curves; nothing depends on the input content at run time.
- K2 When a service mixes configurations (steps, resolutions, durations) or the mix shifts, each role's pool scales on
  its own (e.g. add decoder pods); a co-located replica has a fixed DiT:decoder ratio and cannot.
- K3 Larger decoder replicas buy latency, not throughput, once the decoder keeps up: LingBot decode per chunk with a
  4-chip DiT 391 / 273 / 237 ms on 1 / 2 / 4 VAE chips at the same 17.5 fps [M].

Constraints
- X1 Stateful decoders (LingBot's causal 3-D VAE) keep a per-stream cache: one stream's chunks cannot spread over
  decoder replicas; N:M shares a decoder across streams (per-session cache, swapped per chunk) [implemented:
  LINGBOT_VAE_SESSIONS], or moves the cache on failover. Stateless decoders (SA3 chunks) spread freely.
- X2 Splitting a fixed budget can starve the stage that scales: LingBot 8 chips co-located 20.0 fps vs 4 + x 17.5 [M];
  SA3 2 chips co-located 6.15 s vs 1 + 1 7.50 s [M]. Disaggregation pays when it adds chips the co-located replica
  cannot use, or frees chips the decoder wastes.
- X3 A DiT that cannot shard further (SA3 TP16 fails: head split) caps the co-located replica; only a separate
  decoder role can add chips [M].

## 7 Discussion

- Where disaggregation does not pay (8-chip LingBot).
- Stateful decoders and state migration.
- Heterogeneous decoder hardware [H: trn2 decoder].
- Network isolation for the role APIs: NetworkPolicy, authenticated pairing.

## 8 Related work

- LLM prefill/decode disaggregation: DistServe, Splitwise, Mooncake, NVIDIA's study (arXiv 2506.05508).
- Diffusion serving systems.
- Streaming video generation and world models.
- Pipeline and model parallelism.
- Cluster schedulers for heterogeneous jobs.

## 9 Conclusion

Two-stage generative pipelines are limited by the coupling of two compute-bound stages, not by either stage alone.
Treating the stages as independently sized, replicated and placed roles, an operating-system responsibility,
recovers the lost chip time. On Trainium with EKS, DRA, EFA and NIXL it does so with a hand-off cost below a
millisecond.
