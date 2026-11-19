# DiT/VAE disaggregation for streaming video world models: research plan

Working title: **"Split the Denoiser from the Decoder: Disaggregated Serving for Streaming Video World Models
on AWS Trainium"**. Modeled on NVIDIA's "Beyond the Buzz: A Pragmatic Take on Inference Disaggregation"
(arXiv 2506.05508), which studies prefill/decode for LLMs, but for the diffusion pattern: a DiT that
denoises latent chunks and a VAE that decodes them to pixels, in a live, steerable stream.

## 1. The research problem

A streaming video world model (LingBot-World v2, 14B causal DiT + Wan2.1 VAE) turns a prompt, a start image
and a stream of actions (camera / robot commands) into video, chunk by chunk (here 3 latent frames = 12 video
frames per chunk). Each chunk passes through two phases with very different characters:

| | DiT (denoise) | VAE (decode) |
|---|---|---|
| Work per chunk | 4 denoise steps over the window's tokens + a KV-cache commit | 3 latent frames to 12 RGB frames, causal 3-D convolutions |
| Bound by | matmul (Tensor engine), collectives (TP/CP) | memory traffic, convolution halos, spatial tiling |
| Parallelism that suits it | TP x CP (sequence parallel) over chips | spatial tiles (height x width) over cores |
| State it carries | rolling KV cache (attention memory, sink + window) | causal temporal feature cache (previous frames' activations) |
| Output size per chunk | 1.1 MB latent (16 x 3 x 60 x 96 fp32) | 13 MB of pixels (12 x 480 x 768 x 3) |

Co-located (one pod), the two phases run one after the other on the same chips: each chunk pays DiT + VAE,
and each phase runs with the other's parallelism and memory layout. Disaggregated, the VAE runs on its own
chips, the DiT starts chunk n+1 while chunk n decodes, and only the 1.1 MB latent crosses the network.

**Question.** When, why and by how much does disaggregating the DiT and the VAE improve a streaming video
world model's per-stream frame rate, action-to-photon latency, streams per chip and cost; what DiT:VAE ratio
and placement is optimal for a target frame rate; and what does it take on Trainium (EFA, NIXL, EKS, DRA) to
make it reliable (failover, elasticity)?

### How it differs from LLM prefill/decode disaggregation
1. **Both phases run every step, not once per request.** Prefill runs once, decode per token. Here DiT and VAE
   alternate every chunk for the whole stream, so the win is pipelining (overlap), not phase isolation alone.
2. **Both sides are stateful.** The DiT keeps a KV cache, the VAE a causal temporal cache. Failover of either
   side loses state (a VAE switch costs one fresh first chunk); LLM disaggregation moves state once.
3. **The transfer is tiny.** 1.1 MB per chunk (~1.3 MB/s per stream at 16 fps) vs GBs of KV cache, so the
   interconnect is never the bottleneck (measured: 0.17 ms per chunk over EFA with NIXL). That opens placement
   choices LLM P/D cannot use: cross-node, cross-instance-type, many DiTs per VAE.
4. **The SLO is a frame rate, not tokens/s/user.** A stream must sustain real time (e.g. 16 fps), with a bounded
   action-to-photon latency (robotics: closed-loop control; games: input lag) and time-to-first-frame.
5. **Quality couples to placement.** The VAE's tiling and precision depend on how many cores it gets; inline on
   8 LNC2 ranks one tiling decoded black frames, and a 2-minute stream behaves differently online vs offline.

## 2. Hypotheses
- **H1 (pipelining).** Disaggregation raises per-stream fps from 1/(T_dit + T_vae) toward 1/max(T_dit, T_vae).
  Measured so far: 2-chip co-located 11.53 fps -> 2 + 1 chips split 14.65 fps (+27%, 2 minutes, trn3).
- **H2 (rate matching).** Since T_vae < T_dit, one VAE can serve k DiTs (fan-in k:1). The optimal k is about
  T_dit / T_vae (plus headroom), which lowers chips per stream and cost per stream-hour vs co-located.
- **H3 (bandwidth is not the constraint).** Latent transfer needs ~1-2 MB/s per stream; EFA RDMA with NIXL makes
  placement (same node, cross node) almost free in latency (< 1 ms), unlike LLM KV transfer.
- **H4 (latency).** Action-to-photon latency drops because the decode of chunk n no longer delays the
  denoise of chunk n+1; the press-to-frame critical path becomes wait + DiT + transfer + VAE step + network.
- **H5 (where it does not pay).** For a single stream with enough chips (4-chip monolith 14-15 fps) or a small
  VAE share of time, co-location is as good and simpler. Map that boundary (resolution, model size, steps).
- **H6 (resilience).** A disaggregated VAE pool turns a VAE failure into a ~3 s blip with no lost chunks
  (measured: pod delete and device-fault taint), vs a full restart of a co-located pod (minutes).

## 3. Metrics (measured, not simulated)
- **Per-stream fps** (sustained over 2 minutes), and whether it meets real time (16 fps).
- **Time to first frame** (session start -> first frame at the client).
- **Action-to-photon latency**: key press -> first frame showing it, decomposed into wait / DiT / transfer /
  VAE / network / client buffer (the web client already reports this breakdown).
- **Throughput**: streams per chip at >= real time; frames per second per chip.
- **Cost**: chip-hours per stream-hour (and $ with on-demand trn3 / trn2 prices as stated assumptions).
- **Quality**: per-frame PSNR vs the CPU fp32 reference on the gated 81-frame clip; drift over 2 minutes
  (saturation, sharpness, frame-to-frame change); visual review.
- **Resilience**: chunks lost and pause length on VAE failover; recovery time.

Headline figure, in the paper's style: a **Pareto frontier of per-stream fps (x) vs streams per chip (y)**,
co-located vs disaggregated, across configurations.

## 4. Design space
| Axis | Values |
|---|---|
| DiT chips / parallelism | 1, 2, 4, 8 trn3 chips; TP4 x CP{1,2,4,8}; LNC2 (and LNC1) |
| VAE chips / tiling | 1, 2 chips; LNC1 / LNC2; 2x2, 2x4, 4x4 tiles |
| DiT : VAE ratio | 1:1, 2:1, 3:1, 4:1 (one VAE serving several streams) |
| Placement | same node / cross node; trn3 DiT + trn2 VAE (heterogeneous) |
| Transport | inline (co-located), HTTP host copy, torch-neuronx RDMA, NIXL one-sided write |
| Workload | 480x768 (and 720p if it compiles), 4 denoise steps, chunk 3; idle vs active steering |
| Precision | MXFP8 DiT (gated), bf16 DiT; bf16 VAE |
| Streams | 1, 2, 4 concurrent sessions |

## 5. Experiments
1. **Phase characterization.** Profile one chunk of each phase (neuron-profile / system trace): engine
   utilization, HBM traffic, collectives; DiT vs VAE roofline position. Chart: per-engine busy time per phase.
2. **Co-located vs disaggregated, single stream.** Same chip budget (2, 3, 4, 8 chips): fps, TTFF, latency.
   Chart: fps per configuration, stacked time per chunk (DiT / VAE / transfer / overhead).
3. **Rate matching (fan-in).** k DiTs : 1 VAE for k = 1..4 (a VAE pool shared by sessions): fps per stream vs
   k, VAE utilization. Chart: fps vs k; chips per stream vs k.
4. **Throughput / cost Pareto.** Enumerate the measured configurations; per-stream fps vs streams per chip;
   cost per stream-hour. Chart: Pareto frontier, co-located vs split (the paper's Fig. 1 equivalent).
5. **Transport and placement.** Inline / HTTP / torch RDMA / NIXL; same vs cross node: transfer time, CPU
   time, fps. Chart: per-chunk transfer latency CDF; fps by transport.
6. **Action-to-photon latency.** Scripted key presses; decomposition by stage. Chart: stacked latency bars,
   co-located vs split; CDF over 100 presses.
7. **Heterogeneous VAE.** VAE on trn2 (or a single trn3 core group) with the DiT on trn3: fps and cost.
8. **Resilience and elasticity.** VAE pod delete, device-fault taint, scale VAE pool 0 -> N under load: lost
   chunks, pause, recovery. Chart: timeline of fps with failure events.
9. **Long-stream stability.** 2-minute streams online vs offline, with and without re-anchoring every 800
   frames; drift metrics over time. Chart: drift metric vs time.
10. **Sensitivity.** Resolution, denoise steps, DiT precision: when does the VAE share grow enough that
    disaggregation matters more (or less). Chart: VAE time share vs configuration, with the fps gain.

## 6. Use cases the results speak to
- **Interactive world models / games**: a real-time frame rate and a low input lag per player.
- **Robotics**: a world model as a closed-loop simulator (action in, video out) needs bounded action-to-photon
  latency; policy training needs many streams in parallel (throughput, cost).
- **Synthetic data**: offline long rollouts (throughput per chip, quality over minutes).

## 7. Stack (all open source except the hardware)
Trainium3 (trn3-dev1), Neuron SDK 2.33, PyTorch native (torch-neuronx, torch.compile), NKI kernels, MXFP8,
LingBot-World v2; EFA with libfabric; NIXL (LIBFABRIC backend, Neuron device memory); Amazon EKS with the
Neuron DRA driver (claims with paired EFA: m-lnc2-efa, s-lnc2-efa); the serving code in this repo
(serve.py, vae_serve.py, wan/vae_pool.py, wan/nixl_link.py). Measurement tools: perf/drift_probe.py,
kiro_mx4/e2e.sh + perf/e2e/gate.sh (accuracy gate), neuron-profile, enhanced neuron-top.

## 8. What I need to run it
- A session with AWS credentials for KEEP-trn3-dev1 (EKS, pods, DRA claims), for several hours unattended.
- Chips: one trn3 node to myself for most experiments (16 devices: up to 8 for DiTs, the rest for VAEs);
  the trn2 node for experiment 7; both nodes for cross-node placement.
- The demo pods can be paused during runs (I restore them at the end).

## 9. Deliverables
A paper draft (docs/disagg/paper.md, figures in docs/disagg/fig/), every number traced to a measured run, the
raw results (CSV / JSON) and the scripts to reproduce each figure.
