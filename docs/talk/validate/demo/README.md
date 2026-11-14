# S5b / S5h demo: LingBot attention, torch SDPA vs the NKI kernel

One LingBot rank's self-attention (TP4 x CP8, 480x768, steady state): q [1, 10, 640, 128],
k/v [1, 10, 17280, 128], bf16, one trn3 LNC2 core, image torch-neuronx 2.14.0 / SDK 2.33.
Job `demo_job.yaml` (claim `s-lnc2`, node ip-192-168-192-229), measured 2026-11-14.

## Run it

```bash
NEURON_LOGICAL_NC_CONFIG=2 TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_sdpa python3 sdpa_vs_nki.py --mode sdpa
NEURON_LOGICAL_NC_CONFIG=2 TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_nki  python3 sdpa_vs_nki.py --mode nki
```

- `--mode sdpa` calls `F.scaled_dot_product_attention` (S5b). The script sets
  `TORCH_NEURONX_ENABLE_NKI_SDPA=0`, because otherwise the compiler puts in an NKI kernel by itself.
- `--mode nki` calls nkilib `attention_cte`, the way LingBot does (S5h). Its Q x K and P x V are
  `nisa.nc_matmul` inside the K-tile loop.

## The pairs (on this Mac, not in git)

`../pairs/demo_sdpa/sdpa.neff` + `sdpa.ntff`, and `../pairs/demo_nki/nki.neff` + `nki.ntff`.
Open each with `neuron-explorer view -n <neff> -s <ntff>` (Linux) or the Neuron Explorer UI.

## What to point at

| | SDPA (S5b) | NKI attention_cte (S5h) |
|---|---|---|
| Device time | 2.687 ms | 1.574 ms (1.7x) |
| Host-timed per call (median of 20) | 2.607 ms | 1.636 ms |
| Tensor busy | 45.4% | 51.2% |
| Every engine idle | 4.2% | 1.4% |
| 3+ engines busy | 37.3% | 41.3% |
| Softmax work (Vector + Scalar) while Tensor idle | 1.275 ms | 0.202 ms |
| Longest Tensor gap | **0.71 -> 1.88 ms (1.17 ms)** | 0.01 -> 0.45 ms (the first K/V loads) |
| Scalar busy (exp) | 1.78 ms | 0.40 ms |
| rel-RMS vs fp32 CPU | 0.0024 | 0.0035 |

In the SDPA trace, find the Tensor row between 0.71 and 1.88 ms. Tensor is idle, and Vector and
Scalar are busy on the softmax from `sdpa_vs_nki.py` (the `F.scaled_dot_product_attention` line).
In the NKI trace, after the first 0.45 ms of loads, Tensor runs `nc_matmul` from `attention_cte.py`
while Vector does the softmax of the previous K tile.

The same story on the full LingBot block (with collectives, rank 1 of 32) is in
`../pairs/1_sdpa_math_rank1` (8.98 ms) and `../pairs/2_attention_cte_rank1` (4.37 ms); see `../RESULTS.md` S5a and S5h.
