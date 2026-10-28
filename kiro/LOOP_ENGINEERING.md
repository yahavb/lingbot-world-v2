# loop_engineering: lingbot-world-v2 causal_fast on trn2.48xlarge

One 81-frame 480×768 video from `examples/03`, chunk 3, window 12, sink 1, 4 denoise steps,
seed 42. Measured on one trn2.48xlarge (16 Trainium2 devices) in a `wan5b-env:1` container
(torch 2.14.0, torch-neuronx 2.14.3, nki 0.7.0).

## 16 devices (64 ranks, LNC2)

| | Code | Layout | GEN (s) | VAE (s) | Total (s) | fps |
|---|---|---|---|---|---|---|
| Baseline (the job) | upstream `main` @ `6cf1011` | TP8 × CP8 | 8.95 | 2.95 | 11.90 | 6.81 |
| Baseline with `const_max_ag` (estimate, not measured; see below) | upstream `main` | TP8 × CP8 | ~6.7 | 2.95 | ~9.7 | ~8.4 |
| Layout change only | `loop_engineering`, flags off | TP4 × CP16 | 22.46 | 3.01 | 25.47 | 3.18 |
| This branch, the job's layout | `loop_engineering`, flags on | TP8 × CP8 | 5.73 | 3.04 | 8.76 | 9.24 |
| **This branch** | `loop_engineering`, flags on | TP4 × CP16 | **5.75** | 3.01 | **8.75** | **9.26** |

**Measured: 1.36× end to end against `main` as it runs on this host (6.81 → 9.26 fps), 1.56× on
GEN. Assumed: ~1.1× against `main` with its default `const_max_ag` kernel, which does not run
here; this is the conservative estimate and the number to quote.** With our flags the two
layouts give the same result (5.73 vs 5.75 s GEN), so on 16 devices the gain is the code.

**The `const_max_ag` estimate.** `main` defaults to `ATTENTION_BACKEND=const_max_ag`, a fused
CP all-gather + attention kernel that subtracts an assumed row max (15.0) instead of computing
it. On this host it fails while compiling (see below), so `main` was measured with `cte`.
Upstream measured it 9.8% faster per DiT forward than `cte` (TP8 × CP4, 768 res). At this
setting it would also skip the `cte` KV assembly that makes `main`'s second chunk slow (482 vs
~150 ms per forward, ~1.66 s of 8.95 s), in 36 of 40 layers (the 4 hot layers stay exact):
8.95 − 1.49 = 7.46 s, × 0.902 = 6.73 s GEN, 9.68 s total, 8.37 fps → 9.68 / 8.75 = 1.106×.
Applying only the 9.8% gives ~1.26×; the lower figure is the one assumed.

- GEN is the steady-state second pass (`[BENCH] GEN time`). VAE is the steady-state call of
  `decode_latents_width_sharded.py`. fps = 81 / (GEN + VAE). These are the job's own definitions
  (`lbw-v2-i2v-job.yaml`); `kiro/run_job16.sh` runs the job's two steps with its args and env.
- All measured runs use `ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte` and launch asynchronously
  (`NEURON_LAUNCH_BLOCKING=0` is torch-neuronx's default; the job does not set it). This
  branch's flags: `LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1`. Both TP8 × CP8 runs use
  the core order and precomputed prompt embeddings described below.
- The VAE step is the same code in every run; 2.95–3.04 s is run-to-run spread.
- Every measured video passes `tests/smoke_test_video.py` (81 frames, 768×480, not blank, not
  frozen).
- Startup, not in the numbers above: with `LINGBOT_LOAD_CONCURRENCY=16` the 64-rank weight load
  takes ~27 s instead of ~380 s (one rank at a time, the default).

Per DiT forward (ms; a chunk is 5 forwards):

| | chunk 0 | chunk 1 | chunk 2 | chunks 3–6 (steady) | first step of a chunk |
|---|---|---|---|---|---|
| Baseline, TP8 × CP8 | ~140 | ~482 | ~165 | ~191 | +35 |
| Layout change only | ~117 | ~3140 | ~160 | ~203 | +25 |
| This branch, TP8 × CP8 | ~109 | ~116 | ~130 | ~136 | +35 |
| This branch, TP4 × CP16 | ~100 | ~117 | ~125 | ~136 | +25 |

The layout change alone is slower: the upstream `cte` path's KV assembly lowers pathologically for
the two-chunk window at CP16 (3.1 s per forward, 15.7 s of that run's 22.5 s). The merged
attention removes the assembly, and SP removes the TP work that does not shrink with CP.

### What the baseline changes about the job, and why

The job runs TP8 × CP8 with `ATTENTION_BACKEND=const_max_ag`. On this host:

1. **`cte` instead of `const_max_ag`.** `const_max_ag` fails inside the NKI kernel call while the
   block is traced; nki's `_traceback_util` wrapper then calls `RuntimeError.add_note`, which
   Dynamo cannot trace, so the kernel's message is lost. It fails the same way at TP4 × CP2 on two
   adjacent devices and at TP8 × CP8 with the core order below, so it is the kernel on this SDK.
2. **A core order.** With the default rank-to-device mapping, TP8 at LNC2 pairs devices (0,1),
   (2,3), …, so each CP8 group is every other device, {0, 2, …, 14}. On this host's 4 × 4 torus
   (device d = row d // 4, column d % 4) that set is two column rings with no link between them,
   and collective setup fails (`Failed to find device to device paths`). torch-neuronx gives local
   rank i the i-th core of a comma-separated `NEURON_RT_VISIBLE_CORES`, so `kiro/core_order.py`
   lists the cores in an order that pairs linked devices and makes each CP8 group one ring
   ({0,3,7,4,8,11,15,12} and {1,2,6,5,9,10,14,13}). No code change.
3. **Precomputed prompt embeddings.** With that core order, large collectives over the whole
   64-rank world abort in the runtime (`rdh_scheduler::verify_read_safety`) while TP and CP group
   collectives work (`kiro/mesh_test.py`). The only world-wide large collective in `generate.py`
   is the inline T5, so it runs once with the default order (`kiro/encode_prompt.py`, the same
   upstream T5-TP code over 64 ranks) and its output goes in through `--prompt_embeds_path`.
   T5 is outside GEN. The decode step uses the default core order, as the job's does.

## 2 devices (8 ranks, devices 14 + 15)

| | Layout | GEN (s) | VAE (s) | Total (s) | fps |
|---|---|---|---|---|---|
| Baseline (upstream code path, `cte`, `NEURON_LAUNCH_BLOCKING=1` as `run_fast_neuron.sh` sets on trn2) | TP8 across 2 devices | 83.6 | 9.16 | 92.8 | 0.87 |
| This branch, same flags as above, async launch | TP4 × CP2 | 14.29 | 6.26 | 20.55 | 3.94 |

4.5× end to end on 2 devices. This comparison includes the layout and the launch mode, and
neither carries over to 16 devices: there the job already launches asynchronously, and its TP8
pairs move 1/8 of the tokens across the device link instead of all of them.

## What changed, measured

Single compiled DiT block (real weights of block 10) via `kiro/block_bench.py`, ms per block,
TP4 × CP2 on 2 devices unless noted. "lead" is how many earlier KV chunks the window holds
(3 = steady state, 1 = the second chunk).

| Change | Before | After |
|---|---|---|
| TP inside a device, CP across devices (TP4 × CP2 vs TP8 over 2 devices; 2-device only, see the 16-device table) | 66.19 | 19.03 |
| `LINGBOT_SP=1` Megatron sequence parallelism inside the TP group | 19.03 | 12.0 |
| async launch (`NEURON_LAUNCH_BLOCKING=0`): removes a ~1.4 ms host gap per block | 12.0 | 10.5 |
| `LINGBOT_CTE_MERGE=1`, lead 1 (the second chunk) | 22.07 | 9.87 |
| `LINGBOT_CTE_MERGE=1`, lead 3 | 12.0 | 12.07 |
| `LINGBOT_FILM_A2A=1`: per-chunk camera FiLM prep (once per chunk per block) | 7.88 | 2.60 |
| TP4 × CP16 on 16 devices, all flags, async | | 3.39–3.45 (lead 1: 2.75) |

- **Layout.** TP8 at LNC2 spans two devices, so every row-parallel all-reduce crosses a device
  link. On 2 devices, where those all-reduces carry every token, keeping TP inside a device and
  putting CP across devices is 3.5× per block. On 16 devices the job's CP8 already cuts that
  traffic 8×, and the layout pays off only together with SP and the merged attention.
- **Sequence parallelism.** Norms, modulation, gating and residual adds run on 1/TP of the tokens;
  the row-parallel all-reduces become reduce-scatters and the column-parallel inputs all-gathers.
- **Merged attention (`LINGBOT_CTE_MERGE`).** The upstream `cte` path copies every KV chunk into a
  zero-filled, 512-aligned buffer before each attention call. In a profile of the steady-state
  block that copy is 1.7 ms of 10.4 ms on the critical path, and for two chunks its lowering is
  pathological (21 ms on its own). The merged path keeps the cache N-major and runs one
  `attention_cte` call per chunk with unnormalized output and softmax statistics, merged with the
  usual max rescale. The copy grows with the KV window, not with tokens per rank, so it matters
  more as CP grows.
- **FiLM all-to-all.** Under SP each rank needs its token shard of the camera FiLM params at full
  width. One all-to-all inside the compiled `cam_film` graph replaces an eager full-width
  all-gather, slice and clone. Bitwise identical to the gather path (checked on rank 0's tensors).
- **Memory.** `_cam_films` builds one block at a time instead of holding all 40 blocks'
  full-width params at once, token shards are cloned so they do not pin the full-width tensors in
  the cache, prewarm frees its camera caches, and the conditioning image is VAE-encoded before the
  DiT loads (`LINGBOT_PRE_ENCODE=1`, default). Without these, TP4 ran out of device memory (in the
  conditioning encode on 1 device, in the KV-cache allocation on 2).

## Correctness

- **Attention padding bias (fixed by `LINGBOT_CTE_MERGE`).** The upstream `cte` exact path pads KV
  to a multiple of 512 with zeros and passes no bounds, so the zero keys enter every softmax
  denominator. Against an fp32 reference (`kiro/attn_merge_test.py`) this scales flat heads' output
  by 0.993 with a full window and 0.939 in the first chunk. The merged path's gain is 1.0000 on
  every head, with lower error on flat heads (rel. RMS 0.0028 vs 0.0078).
- **Run-to-run.** Rerunning the same layout gave bit-identical latents (2-device TP8).
- **Across layouts.** Latents from different layouts diverge with chunk index (rel. RMS ~0.004 in
  the first frame, 0.25–0.36 by the last). SP vs no-SP at the same layout, which differ only in
  reduction order, diverge just as much, so this is the autoregressive rollout amplifying
  rounding differences, not one layout being wrong. Frames from each stay sharp and coherent.

## Flags added (all off by default unless noted)

| Flag | Effect |
|---|---|
| `LINGBOT_SP=1` | sequence parallelism inside the TP group (chunked-KV path) |
| `LINGBOT_CTE_MERGE=1` | N-major KV chunks + per-chunk `attention_cte` merge (exact, no KV assembly) |
| `LINGBOT_CTE_MERGE_K=nld` | A/B only: K as [n, L, d] instead of pre-transposed (slower) |
| `LINGBOT_FILM_A2A=1` | FiLM token shards via all-to-all (needs `LINGBOT_SP=1`) |
| `LINGBOT_FILM_FULL=1` | full-width FiLM params once per chunk without SP |
| `LINGBOT_AR_BF16=1` | row-parallel reductions in bf16 (no measured gain) |
| `LINGBOT_PRE_ENCODE` (default 1) | VAE-encode the conditioning image before the DiT loads |
| `LINGBOT_LOAD_CONCURRENCY=N` | N ranks load the DiT weights per turn (default 1); needs ~10–13 GB host RAM per loading rank |
| `LINGBOT_DUMP_LATENTS=<path>`, `LINGBOT_MEMLOG=1` | diagnostics |

## Scripts (`kiro/`)

Paths assume the container layout used here (`/work/repo`, `/work/ckpt`, `/work/logs`).

| Script | Use |
|---|---|
| `run_job16.sh <repo> <name> [KEY=VAL…]` | the job's generate + decode steps and fps, any repo |
| `core_order.py` | `NEURON_RT_VISIBLE_CORES` order that lets TP8 × CP8 build its collectives here |
| `mesh_test.py` | does a TP × CP rank layout build and run its collectives on this host |
| `encode_prompt.py` | precompute the prompt embeddings (upstream T5-TP, 64 ranks) |
| `run_lbw.sh <name> <nproc> <lnc> <cores> [KEY=VAL…]` | one end-to-end `generate.py` run |
| `run_bb.sh`, `block_bench.py` | one compiled DiT block, any TP/CP/LEAD, optional profile |
| `attn_merge_test.py` | exactness and speed of the attention paths vs fp32 |
| `prof_block.py` | per-op timeline of one block execution from a neuron-explorer profile |
| `cmp_latents.py`, `frames_cmp.py` | latent and frame comparisons |

Reproduce the 16-device table (`/work/repo_upstream` = upstream `main`, `/work/repo` = this branch):

```bash
# baseline: the job's TP8 x CP8
cd /work/repo_upstream && PYTHONPATH=$PWD torchrun --nproc_per_node=64 \
  /work/repo/kiro/encode_prompt.py /work/out/prompt_embeds_job.pt
bash /work/repo/kiro/run_job16.sh /work/repo_upstream base16_job TP=8 CP=8 \
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte DEC_DEFAULT_CORES=1 \
  NEURON_RT_VISIBLE_CORES=$(python3 /work/repo/kiro/core_order.py) \
  PROMPT_EMBEDS=/work/out/prompt_embeds_job.pt
# layout change only / this branch
bash /work/repo/kiro/run_job16.sh /work/repo base16_off TP=4 CP=16 \
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=0 LINGBOT_CTE_MERGE=0 LINGBOT_FILM_A2A=0
bash /work/repo/kiro/run_job16.sh /work/repo ours16 TP=4 CP=16 \
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1 \
  NEURON_LAUNCH_BLOCKING=0
# this branch on the job's layout
bash /work/repo/kiro/run_job16.sh /work/repo ours16_tp8 TP=8 CP=8 \
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1 \
  NEURON_LAUNCH_BLOCKING=0 DEC_DEFAULT_CORES=1 LINGBOT_LOAD_CONCURRENCY=16 \
  NEURON_RT_VISIBLE_CORES=$(python3 /work/repo/kiro/core_order.py) \
  PROMPT_EMBEDS=/work/out/prompt_embeds_job.pt
```
