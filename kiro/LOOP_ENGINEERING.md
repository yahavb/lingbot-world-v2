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
| This branch, first push | `loop_engineering`, flags on | TP4 × CP16 | 5.75 | 3.01 | 8.75 | 9.26 |
| This branch, 4.8 s push (`q6_pair`, below) | `loop_engineering`, flags on | TP4 × CP16 | 4.48 | 0.81 | 5.29 | 15.32 |
| This branch (`q11_host`, below) | `loop_engineering`, flags on | TP4 × CP16 | 4.01 | 0.77 | 4.78 | 16.94 |
| **This branch, latest** (`q14_vae2d`, below) | `loop_engineering`, flags on | TP4 × CP16 | **4.04** | **0.33** | **4.37** | **18.55** |

**Measured: 1.36× end to end against `main` as it runs on this host (6.81 → 9.26 fps), 1.56× on
GEN. Assumed: ~1.1× against `main` with its default `const_max_ag` kernel, which does not run
here; this is the conservative estimate and the number to quote.** With our flags the two
layouts give the same result (5.73 vs 5.75 s GEN), so on 16 devices the gain is the code.

**Update, 4.8 s push: 5.29 s per video, 2.25× against `main` as measured (11.90 s) and 1.83×
against the assumed `const_max_ag` baseline (9.68 s).** The target is 4.8 s (2.0× against the
assumed baseline). Two of the steps are not bit-exact; see the next section.

## Toward 4.8 s per video, measured step by step (16 devices)

Each row is `kiro/run_job16.sh` on all 16 devices at TP4 × CP16 and adds to the row above (flags
in the command below). Every video passes `tests/smoke_test_video.py`.

| Run | Change | GEN (s) | VAE (s) | Total (s) | fps | Numerics |
|---|---|---|---|---|---|---|
| `ours16` | first push | 5.75 | 3.01 | 8.75 | 9.26 | |
| `q1_cheap` | skip the last chunk's unused KV-commit forward; no per-phase timing syncs; token pad 32 | 5.68 | 3.02 | 8.70 | 9.31 | the pad changes the compiled shapes: latents rel. RMS 0.139 vs `ours16` |
| `q2_vaefused` | VAE: whole decoder blocks compiled, width halos inside the graphs | 5.70 | 1.36 | 7.07 | 11.47 | video within 1.8e-5 of the per-conv decode |
| `q3_vaebf16` | VAE: bf16 convolutions (fp32 norms, softmax, output conv), 4 latent frames per step | 5.68 | 0.80 | 6.48 | 12.51 | video PSNR 56.6 dB vs the fp32 decode (8-bit: 53.9 dB, max 23 levels) |
| `q5_imgonly` | conditioning image stays on the host; encode-cache key from the image bytes | 4.88 | 0.81 | 5.69 | 14.25 | latents bit-identical to `q3` |
| `q6_pair` | attention: cached chunks first, two KV chunks per `attention_cte` call | 4.48 | 0.81 | 5.29 | 15.32 | block output rel. RMS 1.3e-3; latents 0.078 vs `q5` |
| `q8_plk_film` | camera Plücker condition compiled; all 40 blocks' FiLM params in one graph per chunk | 4.29 | 0.81 | 5.09 | 15.91 | 106 of 46.4 M bf16 Plücker values move by one ulp; latents 0.157 vs `q6` |
| `q10_head` | head + its gathers compiled; cross-attention cache stored in the kernel's layout | 4.25 | 0.81 | 5.05 | 16.03 | latents 0.163 vs `q8` |
| `q11_host` | noise and timesteps uploaded once per generation; no per-forward `empty_cache`; no head on cache-commit forwards; VAE 3 frames per step | 4.01 | 0.77 | 4.78 | 16.94 | latents bit-identical to `q10`; video PSNR 56.6 dB vs fp32 decode |
| `q12_vaeo1` | VAE: compiler `-O1`, norm sums as a matmul, [W, H] layout, 4 frames per step | 4.03 | 0.55 | 4.58 | 17.69 | video PSNR 56.58 dB vs fp32 decode |
| `q14_vae2d` | VAE: 4 × 16 height × width tiles instead of 64 width slices of a padded latent | 4.04 | 0.33 | 4.37 | 18.55 | video PSNR 56.55 dB vs the unpadded fp32 decode (see below) |

**The VAE width padding changed the video.** At 64 ranks the width-sliced decode pads the 96-wide
latent to 128 by replicating its last column, and the replicated columns reach into the real ones
through the convolutions: the padded fp32 decode is only 43.55 dB (8-bit max error 156) from an
unpadded fp32 decode at 32 ranks. The 2-D tiles need no padding, and the 2-D bf16 decode is 56.55 dB
from the unpadded fp32 decode, the same distance as the 1-D bf16 decodes have from their fp32
references. The upstream job (`main`) decodes with the same padding.

Latent differences grow with chunk index under any rounding change (`q6` vs `q5`: rel. RMS 0.000
in chunk 0 to 0.162 in chunk 6), as they do between the TP8 × CP8 and TP4 × CP16 layouts running
identical code (0.134); see Correctness.

```bash
bash /work/repo/kiro/run_job16.sh /work/repo q14_vae2d TP=4 CP=16 \
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1 \
  NEURON_LAUNCH_BLOCKING=0 LINGBOT_LOAD_CONCURRENCY=16 LINGBOT_SKIP_LAST_COMMIT=1 LINGBOT_NO_SYNC=1 \
  LINGBOT_TOKEN_PAD=32 LINGBOT_ATTN_EARLY=2 LINGBOT_ATTN_PAIR=1 LINGBOT_PLUCKER_COMPILE=1 \
  LINGBOT_FILM_BATCH=1 LINGBOT_HEAD_COMPILE=1 LINGBOT_CROSS_NMAJOR=1 LINGBOT_PREUPLOAD=1 \
  LINGBOT_EMPTY_CACHE_SEQ=1000000000 LINGBOT_COMMIT_NOHEAD=1 \
  LINGBOT_VAE_FUSED=1 LINGBOT_VAE_BF16=1 LINGBOT_VAE_CC_FLAGS=-O1 LINGBOT_VAE_NORM=mm \
  LINGBOT_VAE_T=4 LINGBOT_VAE_2D=1
```

Where the time went at `q3`, from synced phase timers (`LINGBOT_FWD_PHASES=1`) and block profiles:

- GEN: 34 DiT forwards, 4.18 s in the 40 blocks; the head 5.5 ms per forward (0.19 s); the
  per-chunk camera FiLM 22 ms (0.15 s); the camera Plücker condition 0.14 s; the encode-cache key
  0.33 s (a device round trip of the full-resolution photo, fixed in `q5`).
- One steady-state block (TP4 × CP16, `kiro/block_bench.py` + profile): 3.2 ms, with the Tensor
  engine busy 0.5 ms. The four attention calls take 1.1–1.4 ms and the cores idle ~0.6 ms waiting
  for the CP all-gathers of K and V. Pairing the KV chunks takes the block from 3.28 to 2.88 ms.
- VAE: 0.81 s of device time (35 ms of host issue).

Measured and not adopted:

- `LINGBOT_FUSE_COMMIT=1` (each chunk's KV commit inside the next chunk's first forward): GEN 4.21
  vs 4.03 s. The first fused forward's output is 1.1e-2 (rel. RMS) from the unfused one, so it is
  correct, but a fused forward costs about as much as the two it replaces (blocks 220 ms + 32 ms
  of embedding and FiLM vs ~255 ms), and the next step recomputes the chunk's FiLM.
- Compiler `-O1` on the DiT block: 3.23 vs 2.87 ms. Token pad 4 (272 instead of 288 tokens): 2.90
  ms (the attention runs 3 query tiles of 128 either way). In-graph copies of the cached chunks
  for the attention kernel: 2.89 ms.

- A compiled head with unpatchify inside the graph: neuronx-cc fails on its CP all-gather
  (`NCC_IBIR158`, access pattern out of bounds). Gathering the 2-D [tokens, 64] rows instead
  compiles (`q10`).
- One fused [3 × local dim] QKV matmul with one norm all-reduce: 2.94 vs 2.88 ms per block.
  Making K/V depend on Q so that Q is issued first: no change. One all-gather of stacked K and V
  instead of two: 3.12 vs 2.87 ms. Compiler `--model-type=transformer` / `-O3` on the DiT block:
  2.86 / 2.90 vs 2.87 ms. TP8 × CP8 with the same flags: 3.09 vs 2.88 ms.
- VAE: `--model-type=unet-inference` 0.74 s, `-O3` 0.78 s, vs 0.77 s (`-O1`: 0.65 s); decoding on
  [W, H] instead of [H, W] tensors (`LINGBOT_VAE_SWAPHW=1`) 0.61 vs 0.65 s at `-O1` but 0.83 vs
  0.77 s at `-O2`; skipping the halo exchange entirely (wrong pixels, diagnostic) is slower,
  0.95 s, so the halos are not the VAE cost.
- `LINGBOT_CROSS_XLA=1`, `LINGBOT_KV_WRITE_LATE=1`: block time 3.43–3.45 ms vs 3.43 ms.
- A 12-device DiT + 4-device VAE split: the runtime builds collectives only for 1, 2, 4, 8, 16 or
  a multiple of 32 logical cores, so the 48-rank DiT fails at init.
- `LINGBOT_VAE_HALO=pairs`: 2-rank groups on one device fail collective setup. `hier` is
  bit-identical and took the per-conv decode from 2.95 to 2.75 s; the fused decoder supersedes it.
- `LINGBOT_VAE_T`: 1 / 4 / 10 / 20 frames per step = 1.35 / 1.19 / 1.21 / 1.93 s (fp32), so the
  number of graph launches does not limit the VAE.

Every experiment, including the refuted ones, is in `kiro/perf_ledger.jsonl`.

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
| `LINGBOT_SKIP_LAST_COMMIT=1` | skip the last chunk's KV-commit forward (nothing reads it) |
| `LINGBOT_NO_SYNC=1` | drop the per-phase timing syncs in the chunk loop (per-forward times become host issue times) |
| `LINGBOT_TOKEN_PAD=N` | pad each rank's tokens to a multiple of N (32 at CP16: 270 → 288 instead of 384) |
| `LINGBOT_VAE_FUSED=1` | decode with `wan/modules/vae2_1_fused.py`: compiled decoder blocks, halos in the graphs |
| `LINGBOT_VAE_BF16=1`, `LINGBOT_VAE_T=k` | fused decoder: bf16 convolutions; k latent frames per step after the first |
| `LINGBOT_VAE_HALO=hier` | per-conv decoder: hierarchical halo exchange (bit-identical) |
| `LINGBOT_ATTN_EARLY=1/2` | merged `cte` under CP: cached chunks first; the new chunk gathered in cache layout (2: gathers issued first) |
| `LINGBOT_ATTN_PAIR=1` | with `LINGBOT_ATTN_EARLY`: two KV chunks per kernel call, the first as the kernel's prefix-cache prior |
| `LINGBOT_PLUCKER_COMPILE=1` | camera Plücker condition as one compiled graph (`LINGBOT_PLUCKER_CHECK=1` logs the difference) |
| `LINGBOT_FILM_BATCH=1` | all 40 blocks' FiLM params in one compiled graph per chunk |
| `LINGBOT_HEAD_COMPILE=1` | head + TP/CP gathers of the patch rows as one compiled graph |
| `LINGBOT_CROSS_NMAJOR=1` | cross-attention K/V cache stored in `attention_cte`'s layout (bit-identical) |
| `LINGBOT_PREUPLOAD=1` | all per-step noise draws and the timesteps moved to the device once (bit-identical) |
| `LINGBOT_EMPTY_CACHE_SEQ=N` | per-forward `empty_cache` only for chunks of ≥ N tokens (default 4096 fired at 480p) |
| `LINGBOT_COMMIT_NOHEAD=1` | skip the head on KV-cache commit forwards (their output is discarded) |
| `LINGBOT_FUSE_COMMIT=1` | each chunk's KV commit inside the next chunk's first forward (`forward_fused`); measured slower |
| `LINGBOT_VAE_2D=1` | fused VAE: 4 × 16 height × width tiles of the unpadded latent |
| `LINGBOT_VAE_CC_FLAGS=…`, `LINGBOT_DIT_CC_FLAGS=…` | extra compiler options for the decode / generate process (`+` for spaces) |
| `LINGBOT_VAE_NORM=mm`, `LINGBOT_VAE_SWAPHW=1` | fused VAE: norm sums as a matmul; decode on [W, H] tensors |
| `LINGBOT_DUMP_PRED=dir:tags` | diagnostic: save chosen forwards' outputs (`kiro/cmp_dump.py` compares two runs) |
| `LINGBOT_CROSS_XLA=1`, `LINGBOT_KV_WRITE_LATE=1` | A/B only, no gain |
| `LINGBOT_FWD_PHASES=1` | diagnostic: synced per-phase timings of every forward and of the pre-loop setup |

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
| `run_vae.sh`, `vae_bench.py`, `vae_cmp.py` | VAE decode bench (any decoder mode, optional profile); slice-by-slice output comparison |
| `prof_crit.py` | per-core op timeline and idle gaps of one execution from a neuron-explorer profile |
| `perf_ledger.jsonl` | every experiment of the 4.8 s push, refuted ones included |
| `devices_free.sh`, `queue_runner.sh` | host-side job queue that starts a job only when no process holds a Neuron device |

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
