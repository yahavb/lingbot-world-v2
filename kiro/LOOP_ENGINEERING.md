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
| This branch (`q14_vae2d`, below) | `loop_engineering`, flags on | TP4 × CP16 | 4.04 | 0.33 | 4.37 | 18.55 |
| This branch (`q16_key`, below) | `loop_engineering`, flags on | TP4 × CP16 | 3.94 | 0.31 | 4.24 | 19.10 |
| This branch (`q20_kbnw`, below) | `loop_engineering`, flags on | TP4 × CP16 | 3.66 | 0.31 | 3.97 | 20.43 |
| **This branch, latest** (`q33_a`, below) | `loop_engineering`, flags on | TP4 × CP16 | **2.93** | **0.31** | **3.24** | **25.02** |

**Measured: 1.36× end to end against `main` as it runs on this host (6.81 → 9.26 fps), 1.56× on
GEN. Assumed: ~1.1× against `main` with its default `const_max_ag` kernel, which does not run
here; this is the conservative estimate and the number to quote.** With our flags the two
layouts give the same result (5.73 vs 5.75 s GEN), so on 16 devices the gain is the code.

**Update, 4.8 s push: 5.29 s per video, 2.25× against `main` as measured (11.90 s) and 1.83×
against the assumed `const_max_ag` baseline (9.68 s).** The target is 4.8 s (2.0× against the
assumed baseline). Two of the steps are not bit-exact; see the next section.

**Update: 4.24 s per video (19.10 fps), 2.81× against `main` as measured and 2.28× against the
assumed `const_max_ag` baseline**, still 4 denoise steps.

**Update: 3.97 s per video (20.43 fps), 3.00× against `main` as measured and 2.44×
against the assumed `const_max_ag` baseline**, still 4 denoise steps. The steps since `q16` change
the attention kernel's data movement only, but not bit-exactly; see the numerics column and the
note below the table.

**Update, latest: 3.24 s per video (25.02 fps), 3.68× against `main` as measured and 2.99×
against the assumed `const_max_ag` baseline**, still 4 denoise steps. Four runs of the final flag set
gave 24.99–25.04 fps, so 25 fps is reached with no margin. The two large steps are the constant-offset
attention in 34 of the 40 layers (`q25`) and the work before the first block (`q30`–`q33`). Since
`q25`, repeat runs of one config are no longer bit-identical (rounding level; see Correctness).

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
| `q15_filmall` | every chunk's camera FiLM params in one compiled graph before the chunk loop; VAE 6 frames per step | 3.98 | 0.31 | 4.29 | 18.88 | FiLM params vs the per-chunk path (`LINGBOT_FILM_CHECK=1`): 0.3% of elements differ, max \|d\| 7.8e-3, rel. RMS 8e-5, over all 7 chunks, 40 blocks and 64 ranks; latents 0.125 vs `q14`; video PSNR 56.57 dB vs the unpadded fp32 decode |
| `q16_key` | encode-cache key: the image held in one memory block and hashed in place, checksums in 4 threads | 3.94 | 0.31 | 4.24 | 19.10 | latents bit-identical to `q15`; a repeat run gave 3.92 + 0.31 s (19.15 fps) |
| `q18_vgroup` | attention: `attention_cte` vendored as `kernels/attention_cte_lb.py`; V tiles loaded 8 per DMA instead of one each | 3.74 | 0.30 | 4.04 | 20.07 | kernel output bitwise equal to nkilib's on random inputs at every DiT call shape (`kiro/attn_bitcmp.py`); block output (RMS 0.515) on all 64 ranks at lead 0: 44 bitwise equal, the rest max \|d\| 5.2e-3, rel. RMS ≤ 9.9e-5; latents 0.146 vs `q16`; a repeat run is bit-identical |
| `q20_kbnw` | attention: K read in the CP all-gather's rank-block layout (no K permute); denoise steps attend the gathered K/V without writing the cache slot, which the commit forward writes | 3.66 | 0.31 | 3.97 | 20.43 | block output vs nkilib on all 64 ranks: lead 1 bitwise equal, leads 0 and 3 max \|d\| ≤ 5.2e-3 (rel. RMS ≤ 1.4e-4), lead 3 with the skipped write ≤ 6.2e-3 (7.9e-5); latents 0.139 vs `q16` (chunk 0: 0.013, chunk 6: 0.271), 0.131 vs `q18` |
| `q21_chq` | attention: the self-attention calls chained in the kernel (each continues the previous call's online softmax, the last normalizes) instead of merged in the graph; cross-attention output in the projection's layout; queries padded to 32 rows instead of 128 | 3.47 | 0.31 | 3.78 | 21.45 | latents 0.138 vs `q20` (first frame 0.005, last 0.271) |
| `q22_vtm` | V gathered and cached token-major, as the projection leaves it (no V permute after the CP gather) | 3.35 | 0.30 | 3.65 | 22.19 | latents 0.158 vs `q21_chq` (last frame 0.27–0.32) |
| `q23_pf` | the kernel's prefix-cache prior always whole (no dynamic length mask on its K tiles) | 3.34 | 0.30 | 3.64 | 22.23 | block output bitwise equal |
| `q24_ior256` | runtime I/O descriptor cache of 256 entries (`NEURON_RT_IO_RING_CACHE_SIZE=256`) | 3.30 | 0.30 | 3.60 | 22.49 | latents bitwise equal to `q23_pf` |
| `q25_cmax_ior` | constant-offset self-attention in the 34 layers whose softmax row maxes stay within [−8.3, 71.5] (`LINGBOT_CMAX=2`, offset 30; see below) | 3.06 | 0.30 | 3.36 | 24.12 | block output vs the exact chained calls: rel. RMS ≤ 1.4e-3 on all 64 ranks; latents 0.151 vs `q23_pf` (last frame 0.26) |
| `q26_ior1024` | descriptor cache of 1024 entries | 3.04 | 0.31 | 3.35 | 24.21 | runtime only |
| `q30_start` | KV caches kept across generations instead of new zero-filled ones; host setup (seeded noise draws, encode-cache key) in background threads; the camera FiLM precompute issued right after the Plücker condition; no syncs before the chunk loop | 2.99 | 0.30 | 3.29 | 24.62 | same draws and graphs; latents 0.083 vs `q26`, the size of the run-to-run spread (0.058 between `q33_a` and `q33_b`) |
| `q33_a` | camera embedding MLP on each CP rank's own tokens only; a denoise step's x0 and re-noised input in one compiled graph; no garbage collection inside a generation | 2.93 | 0.31 | 3.24 | 25.02 | latents of `q33_b` 0.150 vs `q23_pf` (last frame 0.27); repeats `q33_b` 2.93 + 0.30 s (25.03 fps), `q34_up` 2.94 + 0.30 s (24.99), `q34_upemb` 2.93 + 0.30 s (25.04) |

**The VAE width padding changed the video.** At 64 ranks the width-sliced decode pads the 96-wide
latent to 128 by replicating its last column, and the replicated columns reach into the real ones
through the convolutions: the padded fp32 decode is only 43.55 dB (8-bit max error 156) from an
unpadded fp32 decode at 32 ranks. The 2-D tiles need no padding, and the 2-D bf16 decode is 56.55 dB
from the unpadded fp32 decode, the same distance as the 1-D bf16 decodes have from their fp32
references. The upstream job (`main`) decodes with the same padding.

Latent differences grow with chunk index under any rounding change (`q6` vs `q5`: rel. RMS 0.000
in chunk 0 to 0.162 in chunk 6), as they do between the TP8 × CP8 and TP4 × CP16 layouts running
identical code (0.134); see Correctness.

**Why `q18` and `q20` are not bit-exact, and why they are kept.** Both change only which DMAs
bring K and V into the kernel and whether the new chunk's K/V pass through the cache slot first;
no arithmetic changes. Still, on some ranks the block output moves by up to 6.2e-3 (output RMS
0.515). The same happens with no kernel change at all: the nkilib kernel with only the skipped
cache write leaves 40 of 64 ranks off by up to 4.6e-3, and changing nothing but how the kernel's
DMA descriptors are generated (`LINGBOT_CTE_KV_DGE=none`, `q17`) moves the latents by 0.065. The
evidence points to rounding, not a wrong value: the runs are deterministic (`q18` and its repeat
are bit-identical), the vendored kernel is bitwise equal to nkilib's on random inputs at every
call shape, and in a single-core comparison with K in rank blocks only the fp32 row sum differs,
in 446 of 3840 elements, by at most 4.9e-4 (one fp32 ulp at its magnitude), with the output and
row max bitwise equal. End to end, `q20` drifts from `q16` by 0.139 (rel. RMS of the latents),
inside the range of the accepted rounding changes above (`q6` 0.078, `q8` 0.157, `q10` 0.163,
`q15` 0.125) and of the layout control (0.134), and its video passes the smoke test.

**The constant-offset attention (`q25`).** The online softmax spends part of every kernel call on the
running row max, the rescaling of the accumulator and a transpose of the probabilities. A diagnostic
run (`LINGBOT_ROWMAX_GRAPH=1`) recorded each layer's per-head softmax row maxes over a whole
generation on all 64 ranks: 34 layers stay within [−8.3, 71.5], while layers 2, 13, 33, 34, 38 and
39 reach 126–864 in one or two heads. The 34 layers compute softmax(QKᵀ − 30)·V with an fp32
[V | 1] accumulator (`kernels/attention_cmax2_lb.py`, one call over the cached chunks and one over
the new chunk). That is the exact softmax up to rounding while every row max m satisfies
−85 < m − 30 < 78 (no underflow of a whole row, no overflow of exp or of its fp32 sums over the
17280-key window), a margin of at least 36 on both sides of the measured range. The other six layers
keep the exact chained `attention_cte` calls. One core, four chunks: 0.35–0.37 ms instead of
0.70 ms; the block at lead 3: 1.97 instead of 2.31 ms. A version with one offset per layer, passed to
the block as a tensor, tracked the exact path through chunk 3 and then diverged (latents 0.461 vs
`q23_pf`, last frame 0.95); the cause was not found and it was dropped.

**The start of a generation (`q30`–`q33`).** In a rank-0 runtime system profile of `q26`
(`SYSPROF_DIR`, `kiro/sysprof_gaps.py`) the first block ran 248 ms into GEN: 320 KV-cache zero
fills (~35 ms), the cross-attention cache fills (~11 ms), ~90 ms of gaps while the device waited for
host work (the seeded CPU noise draws and their upload; other ranks at the first block's barrier) and
the 66 ms FiLM graph issued after them. With the `q33` flags the first block starts ~150 ms in. The
steady host trace of `q34_up` (`LINGBOT_HOST_TRACE=1`) shows the Plücker graph issued by 31 ms, the
FiLM graph by 59 ms and the first forward at 140 ms, after the background thread's 7.7 M normal
samples. One run with the `q33` flags (`q30_startfs`) took 6.30 s of GEN with 0.7 s of host time;
the same flags gave 2.90 s under the system profile and 2.93–2.94 s in four more runs. The cause of
that run was not found.

```bash
bash /work/repo/kiro/run_job16.sh /work/repo q33_a TP=4 CP=16 \
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1 \
  NEURON_LAUNCH_BLOCKING=0 LINGBOT_LOAD_CONCURRENCY=16 LINGBOT_SKIP_LAST_COMMIT=1 LINGBOT_NO_SYNC=1 \
  LINGBOT_TOKEN_PAD=32 LINGBOT_ATTN_EARLY=2 LINGBOT_ATTN_PAIR=1 LINGBOT_PLUCKER_COMPILE=1 \
  LINGBOT_FILM_BATCH=1 LINGBOT_HEAD_COMPILE=1 LINGBOT_CROSS_NMAJOR=1 LINGBOT_PREUPLOAD=1 \
  LINGBOT_EMPTY_CACHE_SEQ=1000000000 LINGBOT_COMMIT_NOHEAD=1 \
  LINGBOT_VAE_FUSED=1 LINGBOT_VAE_BF16=1 LINGBOT_VAE_CC_FLAGS=-O1 LINGBOT_VAE_NORM=mm \
  LINGBOT_VAE_T=6 LINGBOT_VAE_2D=1 LINGBOT_FILM_ALL=1 LINGBOT_KEY_THREADS=4 LINGBOT_IMG_BLOCK=1 \
  LINGBOT_CTE_LOCAL=1 LINGBOT_CTE_VGROUP=8 NKI_ENABLE_TRACE_CACHE=0 LINGBOT_K_BLOCKED=1 \
  LINGBOT_STEP_NOWRITE=1 \
  LINGBOT_CTE_CHAIN=1 LINGBOT_CROSS_TPOUT=1 LINGBOT_CTE_QPAD=32 LINGBOT_V_TOKEN_MAJOR=1 \
  LINGBOT_CTE_PRIOR_FULL=1 LINGBOT_CMAX=2 NEURON_RT_IO_RING_CACHE_SIZE=1024 \
  LINGBOT_KV_REUSE=1 LINGBOT_START_OVERLAP=1 LINGBOT_FILM_SHARD=1 LINGBOT_SCHED_FUSE=1 LINGBOT_GC_OFF=1
```
`q20_kbnw` ran the first ten lines of these flags; each later row adds its own.

Where the time goes at `q33`:
- GEN on the device (rank-0 system profile of the `q33` flags): 2.73 s busy and 0.17 s idle of
  2.90 s. The 1360 block executions take ~2.61 s (1.92 ms each on average).
- One steady block at lead 3 (1.90 ms, device profile): Tensor engine busy 1.12 ms, Scalar 0.75,
  Vector 0.65, GpSimd 0.56. The new chunk's K all-gather runs 364–576 µs and its V all-gather
  717–893 µs, with every engine idle from ~720 µs until V lands. The compiler issues the V gather's
  trigger on GpSimd 140–200 µs after the K gather ends, whatever the program order (`LINGBOT_KV_REAL=2`)
  or the engine of the kernel's copies (`LINGBOT_CMAX_DGE`); one coalesced K + V gather
  (`LINGBOT_KV_COALESCE`) only starts at ~650 µs, after the V projection, so the wait moves instead of
  shrinking. The SP collectives cost ~220 µs on the critical path.

Where the time went in one steady-state block at `q20` (2.53–2.60 ms, TP4 × CP16, profiled):
- The K and V all-gathers over CP16 take ~170 and ~150 µs and run one after the other; for about
  350 µs before the first attention call every engine of the core is idle.
- Each of the two attention calls spends its first ~100 µs loading K and V (the GpSimd engine
  generates their DMA descriptors) before its compute starts; inside a call no engine is more
  than ~60% busy.
- Merging the two calls' partial results costs ~150 µs on the critical path, almost all of it
  laying out the [n, 128, groups] softmax statistics against the [n, Lq, d] outputs (768
  32-element stream transposes on Vector, ~280 one-partition copies on Scalar); the merge
  arithmetic itself is ~10 µs.
- The TP collectives of sequence parallelism add ~220 µs, mostly on the critical path; the FFN
  ~310 µs; cross-attention ~250 µs including its norm and projections.

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
- `LINGBOT_ATTN_REM=1` (the 14 query rows past the last full 128-row tile as fp32 matmuls and
  softmax instead of a third kernel tile): 2.93 vs 2.87 ms per block, output rel. RMS 8.0e-4.
- VAE frames per step on the 2-D tiles: T = 6 / 8 / 10 gave 308 / 299 / 313 ms, but at T = 8 and
  10 the first decode after warmup took 2.6–2.7 s. End to end (`q28`) T = 8 gave 0.301 s, inside
  T = 6's spread (0.300–0.307 s), so T = 6 is kept.

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
- `LINGBOT_LEAD_AFTER_KV=1` (the cached-chunk attention calls ordered after the K/V projections,
  so they run while the CP all-gathers are in flight): 2.65 vs 2.60 ms per block. The calls do
  overlap the gathers but stall, because the gathers, the kernel's K/V loads and its exp
  transposes share the DMA engines.
- `LINGBOT_Q_FIRST=1` (Q projected before K and V): 2.62 vs 2.67 ms with grouped V loads alone,
  2.73 vs 2.63 ms together with the skipped cache write.
- `LINGBOT_CTE_KV_DGE=none` (the kernel's K/V DMA descriptors built by the runtime): 2.67 vs
  2.88 ms per block, but GEN 4.26 vs 3.94 s (`q17`), because the descriptors are rebuilt whenever
  the 40 blocks bind different buffers. Hardware descriptor generation: 1.06–1.15 vs 0.92 ms for
  the steady block's two calls.
- `LINGBOT_ATTN_SPLIT=1` (the 14 remainder query rows of every CP rank gathered and attended to
  each rank's own KV shard): 3.12 vs 2.88 ms per block; each extra kernel call costs ~0.15 ms
  whatever its size.
- V tiles double-buffered across heads (`LINGBOT_CTE_KV_DBUF=v`, `q19`): 20.05 vs 20.07 fps; K and
  V double-buffered do not fit in SBUF (`NCC_INLA001`). One kernel call over all four KV chunks as
  one 17280-key window: 0.72 vs 0.68 ms for the two calls.
- `LINGBOT_CMAX=1` (the constant-offset kernel on plain K/V layouts, loaded per tile): 2.18 ms per
  block at lead 3, against 1.97 ms for `LINGBOT_CMAX=2` in the production layouts.
- `NEURON_RT_IO_RING_CACHE_SIZE=64` (inside noise) and `=4096` (GEN +20 ms against 1024);
  `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=4` (+9 ms).
- `LINGBOT_EMBED_LOCAL=1` (each CP rank embeds only its own tokens): GEN −5 ms, inside the spread.
  `LINGBOT_START_OVERLAP=2` (the noise uploads in the draw thread too): no gain.
  `LINGBOT_CROSS_QPAD=32` (cross-attention queries padded to 32 rows): +0.06 ms per block.
- `LINGBOT_KV_REAL=1…4` (K/V of the real token rows only; 2–4 also put the V gather first in
  program order), `LINGBOT_CMAX_DGE=hwdge_sync` (the constant-offset kernel's copies on the Sync
  engine) and `LINGBOT_KV_COALESCE=1`: the block within ±0.05 ms of the base, inside its spread (see
  the `q33` profile note). `LINGBOT_KV_COALESCE` needs the coalesced all-gather in torch-neuronx's
  CollectiveLegalization table (added at import); without it its group falls back to the whole
  world and neuronx-cc rejects it (`NCC_IVRF100`).
- The block executed on other buffers every call (`NBLK=4` or `8` copies with their own weights, KV
  and caches, as the pipeline's 40 blocks do): 1.93–1.96 ms against 2.01–2.07 ms for one copy, so
  rebinding buffers costs nothing.

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
- **Run-to-run.** Rerunning the same layout gave bit-identical latents (2-device TP8), and on 16
  devices up to `q24`. Since the constant-offset attention (`q25`), two runs of one config agree
  bitwise through chunk 1 and then drift at rounding level (rel. RMS 0.058–0.079 overall, last frame
  0.14–0.18). The constant-offset kernels (20 calls on one core, `kiro/cmax_det.py`) and the compiled
  block (64 ranks, separate processes, leads 2 and 3) are bitwise repeatable on equal inputs, so the
  source is not found yet; those tests repeat identical inputs and would miss a read that races a
  buffer's previous contents.
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
| `LINGBOT_FILM_ALL=1` | every chunk's FiLM params in one compiled graph before the chunk loop (needs `LINGBOT_FILM_A2A`, `LINGBOT_TOKEN_PAD`) |
| `LINGBOT_FILM_CHECK=1` | diagnostic: in the first generation, compare `LINGBOT_FILM_ALL`'s FiLM params with the per-chunk ones on every rank |
| `LINGBOT_KEY_THREADS=N` | encode-cache key: per-slice crc32 + adler32 of every image byte in N threads, read in place when possible |
| `LINGBOT_IMG_BLOCK=1` | `generate.py`: copy the conditioning image into one memory block (lets the key read it in place) |
| `LINGBOT_ATTN_REM=1` | A/B only: the partial last query tile as fp32 matmuls (slower) |
| `LINGBOT_CROSS_XLA=1`, `LINGBOT_KV_WRITE_LATE=1` | A/B only, no gain |
| `LINGBOT_FWD_PHASES=1` | diagnostic: synced per-phase timings of every forward and of the pre-loop setup |
| `LINGBOT_CTE_LOCAL=1` | `attention_cte` from `kernels/attention_cte_lb.py`, the installed nkilib kernel vendored for tuning (changes marked `lingbot:`); with any `LINGBOT_CTE_*` knob set `NKI_ENABLE_TRACE_CACHE=0`, or a cached trace from another setting is reused |
| `LINGBOT_CTE_VGROUP=G`, `LINGBOT_CTE_KGROUP=G` | vendored kernel: G consecutive V / K tiles per DMA (V: 8 adopted; each software-DGE DMA costs ~0.75 µs of GpSimd time whatever its size) |
| `LINGBOT_K_BLOCKED=1` | the K cache holds each chunk as the CP all-gather leaves it, rank blocks [cp, n, d, l]; the vendored kernel reads it with `k_blocks` (no K permute) |
| `LINGBOT_STEP_NOWRITE=1` | denoise-step forwards attend the new chunk's gathered K/V directly and skip the dead write into its cache slot (the commit forward writes it) |
| `LINGBOT_CTE_KV_DGE=…`, `LINGBOT_CTE_KV_DBUF=1\|v`, `LINGBOT_CTE_VGROUP_MIN_TILES` / `_MAX_TILES` | A/B only: vendored kernel DMA descriptor mode, K/V double buffering, which calls group their V loads |
| `LINGBOT_Q_FIRST=1`, `LINGBOT_LEAD_AFTER_KV=1`, `LINGBOT_ATTN_SPLIT=1` | A/B only: issue-order tricks around the CP gathers, remainder query rows split over CP (no gain) |
| `LINGBOT_DUMP_BLOCKS=<dir>` | diagnostic: rank 0 saves every block's output of the first forward (`kiro/cmp_blocks.py` compares two runs) |
| `LINGBOT_CTE_CHAIN=1` | vendored kernel: the self-attention calls chain (each continues the previous call's online softmax, the last normalizes into the output projection's layout) instead of being merged in the graph |
| `LINGBOT_CROSS_TPOUT=1`, `LINGBOT_CTE_QPAD=32` | cross-attention output in the projection's layout; self-attention queries padded to 32 rows (288 instead of 384) |
| `LINGBOT_V_TOKEN_MAJOR=1` | V gathered and cached token-major ([L, n, d] memory under the [n, L, d] shape), read so by the kernel (no V permute) |
| `LINGBOT_CTE_PRIOR_FULL=1` | the kernel's prefix-cache prior always whole: no dynamic length mask on its K tiles (bitwise equal) |
| `LINGBOT_CMAX=1\|2`, `LINGBOT_CMAX_OFFSET=c` | constant-offset self-attention softmax(QKᵀ − c)·V in the 34 layers of `_CMAX_LAYERS` (c = 30 by default); 2 = `kernels/attention_cmax2_lb.py`, whole-chunk loads in the production K/V layouts (adopted) |
| `LINGBOT_ROWMAX_GRAPH=1` | diagnostic: every layer's per-head softmax row-max range over each generation, printed per rank (`[ROWMAX]`, `[ROWMAX_X]` for cross-attention) |
| `LINGBOT_KV_REUSE=1` | keep the self- and cross-attention KV caches for the next generation instead of allocating and zero-filling new ones (every slot is written before it is read) |
| `LINGBOT_START_OVERLAP=1\|2` | seeded noise draws and the encode-cache key in background threads, the FiLM precompute right after the Plücker condition, no syncs before the chunk loop (2: the noise uploads in the thread too); same draws, same order |
| `LINGBOT_FILM_SHARD=1` | with `LINGBOT_FILM_ALL`: the camera embedding MLP on each CP rank's own tokens only |
| `LINGBOT_SCHED_FUSE=1` | a denoise step's x0 and the re-noised next input in one compiled graph |
| `LINGBOT_GC_OFF=1` | no cyclic garbage collection inside a generation (re-enabled after it) |
| `LINGBOT_HOST_TRACE=1` | diagnostic: host time of each step before the chunk loop, no syncs |
| `LINGBOT_EMBED_LOCAL=1`, `LINGBOT_CROSS_QPAD=32`, `LINGBOT_KV_REAL=1…4`, `LINGBOT_CMAX_DGE=hwdge_sync\|hwdge_act`, `LINGBOT_KV_COALESCE=1` | A/B only, no gain (see "Measured and not adopted") |
| `NEURON_RT_IO_RING_CACHE_SIZE=1024` | runtime setting, not a code flag: I/O descriptor cache entries (GEN −42 ms at 256 against the default, a further −19 ms at 1024) |

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
| `hash_bench.py`, `hash_bench64.py` | encode-cache key cost on the host, one process / N processes at once |
| `block_bench.py` `SAVE_REFDIR=` / `REFDIR=` | every rank saves / compares its own block output (a rank-0 comparison misses errors that reach rank 0 only through the next block's gather) |
| `attn_tile_bench.py` | `attention_cte` cost vs query tiles, key length and call count, one core, kernel only |
| `attn_bitcmp.py` | vendored vs nkilib `attention_cte`, bitwise, at the DiT call shapes (`K_BLOCKS=16` for the rank-blocked K) |
| `cmp_blocks.py` | per-block output comparison of two `LINGBOT_DUMP_BLOCKS` runs |
| `prof_nodes.py`, `prof_op.py`, `prof_kernel.py`, `prof_cc.py`, `prof_dma.py` | neuron-explorer profile views: framework-node timeline, one op's instructions per engine, a kernel's engine busy in bins, collectives against compute, HBM traffic per op |
| `prof_rings.py` | collective spans per CC core and all-engine idle intervals of one block execution |
| `run_job16.sh … SYSPROF_DIR=<dir>`, `sysprof_rank0.sh`, `sysprof_gaps.py` | runtime system profile of `generate.py` on rank 0; per-NEFF device time and the idle gaps of the steady generation |
| `attn_chain_test.py` | `LINGBOT_CTE_CHAIN` on one core: chained calls vs the merged path and fp32, and their time |
| `cmax_test.py`, `cmax_det.py` | the constant-offset kernels vs the chained `attention_cte` calls and fp32, one core; their run-to-run determinism |
| `nkl_cte_cmp.py` | an nki-library `attention_cte` branch against `main` on one core: time per call and bitwise comparison of every output (`CASES=table` for the full table) |

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
