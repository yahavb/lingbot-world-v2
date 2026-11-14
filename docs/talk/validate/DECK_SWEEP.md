# Deck sweep: is every number LingBot-World v2 on trn3?

Deck: `demo/talk-deck.html` on `riv26-demo` (identical to `~/Downloads/talk-deck_v2.html`, md5 a03a9769).
Tags:
- **OK**: already LingBot on trn3, or a trn3 hardware spec from the AWS docs.
- **WAN**: a Wan2.2-TI2V-5B number (often trn2). Replace with the LingBot row named here.
- **TRN2**: a LingBot number measured on trn2 (`kiro/LOOP_ENGINEERING.md` 16-device tables). Re-measure on trn3, at most 8 chips.
- **NONE**: no number to check.

Replacement numbers are in `RESULTS.md`, same slide ids.

## Main slides

| Slide | Deck number(s) | Tag | Fix |
|---|---|---|---|
| S0 | section timings | NONE | |
| S0a | Wan2.2-TI2V-5B, 704x544, one trn2 device, 76.6 -> 8.7 s | WAN | LingBot, 8 trn3 chips: 21.6 -> 4.48 s (RESULTS S9c) |
| S1a | 14B, 40 blocks, 7 chunks x 3 latent frames, 4,320 tokens/chunk, 17,280 keys, 4 steps, no CFG, 34 forwards, 1,360 block calls | OK | measured: 35 forwards when the last commit runs (eager, compiled), 34 with `LINGBOT_SKIP_LAST_COMMIT=1` (q20, main) |
| S1b-S1d | none | NONE | |
| S2b | Trainium3: 8 NeuronCore-v4, 144 GiB HBM, 4.9 TB/s, NeuronLink-v4 2.56 TB/s, 16 CC-Cores, 32 MiB SBUF | OK | AWS docs |
| S2c | 16 chips, 128 cores, 4x4 2-D torus, chip 0 neighbours 1, 3, 4, 12 | OK, but the note says "read off neuron-ls on a trn2.48xlarge" | confirm with neuron-ls on trn3 (job logs print it) |
| S2d | "one NEFF for all 30 blocks" | WAN | LingBot: 40 blocks share block graphs per chunk position |
| S3a | 76.6 s eager (DiT 44.3, VAE 32.3); code `inference_neuron_ti2v.py` | WAN | 21.6 s (DiT 19.2, VAE 2.03); code link should be LingBot's `generate.py` |
| S4a | prototype: FFN compiled, all-reduce outside (`models/tp_utils.py`) | WAN (code) | LingBot's piecewise path: `model_fast.py` "selective torch.compile" comment |
| S4b | `models/fast_dit.py`; FFN 3072 -> 14336 -> 3072 | WAN | LingBot `model_fast.py` per-block compile (COMPILE_SCOPE=blocks); FFN 5120 -> 13824 -> 5120 |
| S4c | 4.4 -> 1.91 s/step, 76.6 -> 36.8 s, 30 blocks, 600 calls | WAN | 0.54 -> 0.173 s per forward, 21.6 -> 8.09 s |
| S4d | animation, no numbers | NONE | |
| S4e | Wan VAE: compile 58 min, 11.5-12.9 s; static 12.8-14.2 s; eager + NKI 1.7 s | WAN, and the conclusion flips | LingBot: eager 2.03 s, compiled blocks 1.39 s, final 0.41 s (compiling wins) |
| S4f | Wan VAE block, conv at 75% vs 12% of peak, 1.7 vs 11.5-12.9 s | WAN | drop, or use B4i |
| S4g | 12% vs 75% of peak, 11.5 vs 1.7 s | WAN | the general point stands; the example is Wan |
| S5a | block 56.1 ms, 3 gaps ~22 ms, 3 heads x 2 CFG x 7,854 tokens, pipeline 21.9 s (1.90 s/step) | WAN | 8.98 ms block, 6 gaps 1.54 ms, 10 heads, pipeline 13.4 s (`TORCH_NEURONX_ENABLE_NKI_SDPA=0`) |
| S5b | SDPA red 5-34 ms | WAN | line `attention.py:802`; demo pairs in `pairs/` |
| S5c | idle windows 7-15, 18-25, 28-35 ms | WAN | LingBot gap times from the pair (pending the demo job's timeline) |
| S5d | Tensor 39%, one engine 36%, all idle 16%; Vector 12.8 ms, Scalar 7.7 ms | WAN | 45.3%, 28.5%, 16.6%; 2.93 / 1.87 ms |
| S5e | none | NONE | |
| S5f | 18.2 -> 5.3 ms per call, 7,854 tokens | WAN | per-call numbers from the demo job (pending) |
| S5g | 16.2 -> 13.2 s per video | WAN, and the conclusion flips | LingBot: explicit cte 11.0 s vs native SDPA (auto cte) 8.09 s |
| S5h | 56.1 -> 17.3 ms; self-attn ~5 ms (was ~32); pipeline 9.7 s / 8.7 s | WAN | 8.98 -> 4.37 ms; attention span 7.16 -> 2.52 ms |
| S5i | 56.1/17.3 ms, idle 16% -> 1.1%, Tensor 39% -> 79%, 3+ engines 16% -> 34%; VAE conv3d 98%/95% | WAN | 8.98/4.37 ms, idle 16.6% -> 17.1%, Tensor 45% -> 56%, 3+ engines 22% -> 28% |
| S5j | VAE 6.0 -> 2.6 s, nkilib conv3d | WAN | LingBot VAE is compiled blocks: 2.03 -> 1.39 -> 0.41 s |
| S7a | 8 chips 18.64 fps, 16 chips 25.02 fps | TRN2 | 8 trn3 chips: 18.08 fps (q20), 17.3 (main). 16 chips over budget |
| S7b | TP4 x CP16 (16 devices), TP4 x CP8 (half a trn3) | OK (layouts) | |
| S7d | VAE 9.0 -> 6.0 s (`models/vae_tiled.py`); 17.88 fps half a trn3 | WAN; 17.88 OK | VAE 18.2 -> 2.03 s (1 -> 32 cores); 17.88 re-measured 18.08 |
| S9a | step 0 within bf16 floor; VAE 44.6 dB | WAN (44.6) | c0s0 0.0108-0.0137; VAE 52.7 dB |
| S9b | Wan journey 76.6 / 36.8 / 16.1 / 10.8 / 8.7 s; A100 16.92 s | WAN | LingBot journey 21.6 / 8.09 / 4.48 s (RESULTS S9c); no A100 number for LingBot |
| S9c | 16 trn2 devices: 11.90 -> 3.24 s, 6.81 -> 25.02 fps; 8 devices with check 4.40 s, 18.40 fps | TRN2 | 8 trn3 chips: rows in RESULTS S9c; check-on row pending (job 4) |
| S9d | 17.3 fps, ~1.07 s key press, ~0.58 s per chunk, half a trn3 | OK (trn3 deploy) but not re-measured | pending: serving deploy |
| S11b | `serve_ti2v.py`, 8 ranks, T5 on CPU | WAN | LingBot `serve.py` / `lbw-deploy.yaml` |
| S11c | VAE warm-up 186 / 21 / 3.6 s | WAN | LingBot main: GEN pass 1 151.0 s, pass 2 4.28 s; VAE first decode still compiles (job 1 `dec1.log`) |
| S11d | device busy 96.9% -> 99.5% (`device_unipc.py`); ffmpeg vfork 90 ms; `wan2-ti2v-5b-deploy.yaml` | WAN | LingBot host fixes: `LINGBOT_VAE_GATHER1` frames 0.32 s sooner (kiro/e2e/LEDGER.md, 4 devices) |

## Backups

| Slide | Deck number(s) | Tag | Fix |
|---|---|---|---|
| B2a | Trainium2 device | WAN/TRN2 | marked superseded; leave |
| B4h | LingBot VAE, 64 ranks: 3.01 / 1.36 / 0.80 / 0.55 / 0.33 / 0.31 s; 43.5 / 56.6 dB | TRN2 (16 devices) | re-measure the ladder on 8 trn3 chips (job 4, decode-only) |
| B4i | ~3 s -> 1.36 s -> 0.31 s, 64 ranks | TRN2 | same as B4h |
| B6a | copy 1.7 ms of a 10.4 ms block; heads scaled 0.94-0.99; CP16 3.1 s | TRN2 | `LINGBOT_CTE_MERGE` off/on at TP4 x CP8 (job 4) |
| B6b | GEN 8.95 -> 5.73 s; 3.28 -> 2.88 ms/block; 4.88 -> 4.48 s (ATTN_PAIR) | TRN2 | job 4 flag ladder |
| B6c | 1.7/10.4 ms; 3.28 -> 2.88 ms; GEN 3.94 -> 3.66 s | TRN2 | job 4 |
| B6d | 0.70 -> 0.35-0.37 ms/call; 0 of 1.85 M blocks redone; 18.64 -> 18.40 fps on 8 devices | TRN2 | check on/off on 8 trn3 chips (job 4) |
| B6e | GEN 3.94 -> 3.74 -> 3.66 s; -250 us/block; 4.26 vs 3.94 s | TRN2 | job 4 |
| B6f | block 2.53-2.60 ms at TP4 x CP16 | TRN2 | block profile of main at TP4 x CP8 (job 1 profile is the compiled native-SDPA config; needs one for main) |
| B6g | row max ranges, 0.70 -> 0.36 ms, GEN 3.30 -> 3.06 s, 22.49 -> 24.12 fps (16 devices) | TRN2 | job 4 (constant offset on/off at 8 chips) |
| B6h | code only | NONE | |
| B7c | Wan2.2 layouts on one device, 29.4 -> 20.6 ms | WAN | LingBot layouts on 8 trn3 chips are in `kiro/LOOP_ENGINEERING.md` "Half an instance" (TP4xCP8 17.88, TP8xCP4 16.70, TP2xCP16 14.74 fps) |
| B9e | none | NONE | |
| B10 | 44.6 dB, 76.6 -> 36.8 s, 56.1 -> 17.3 ms, 29.4 -> 20.6 ms | WAN | use the RESULTS rows |
| B11e | "A Trainium2 device" | WAN/TRN2 | say Trainium3 |

## Count

Main slides: OK 5 (S1a, S2b, S2c needs a trn3 neuron-ls confirm, S7b, S9d needs a re-measure),
WAN 24, TRN2 2 (S7a, S9c). Backups: TRN2 9 (B4h, B4i, B6a-B6g), WAN 4 (B2a, B7c, B10, B11e).
