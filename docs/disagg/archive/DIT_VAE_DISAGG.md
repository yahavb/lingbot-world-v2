# LingBot-World v2: DiT and VAE disaggregated over EFA RDMA

Notes for a future paper. Every number below was measured on KEEP-trn3-dev1 (trn3, 480x768, MXFP8,
2026-11-19) with the 2-minute online probe (`perf/drift_probe.py`, 161 chunks, no key presses).

## Setup
- `lbw-dit`: DiT + T5 on 2 trn3 devices (`m-lnc2-efa`, 8 ranks, TP4 x CP2, LNC2), `serve.py`.
- `lbw-vae`: VAE on 1 trn3 device (`s-lnc2-efa`, 4 ranks, LNC2, 2 x 2 tiles), `vae_serve.py`, N replicas.
- Each chunk's latent (16 x 3 x 60 x 96 fp32, 1.1 MB) goes from DiT rank 0's HBM to a VAE rank 0's HBM
  over EFA with NIXL (ai-dynamo/nixl v1.5.2, LIBFABRIC backend, `FI_HMEM_NEURON`): a one-sided WRITE into
  one of 4 slots the VAE registered, plus a per-chunk notification (`wan/nixl_link.py`). HTTP carries only
  the chunk's metadata and brings the JPEG frames back.
- EFA claim templates pair each Neuron device with its own EFA (`devicegroup1_id` of a Neuron device is
  its paired EFA's PCI address).

## Results
| 2-minute online session | Inline VAE (2 chips) | Split, VAE on 1 more chip |
|---|---|---|
| Stream fps | 11.53 | 14.65 to 14.75 |
| DiT time per chunk | 676 ms | 519 ms |
| Latent transfer (NIXL write, 1.1 MB) | - | 0.16 to 0.18 ms (about 6 GB/s) |

## High availability
The DiT picks a VAE per session from the Ready, not-terminating `app=lbw-vae` pods (`wan/vae_pool.py`,
production-stack's pod-discovery rules), and fails over to another VAE if a write or HTTP request fails.
No VAE: the session is refused with `503 VAE_SERVICE_UNAVAILABLE`.

| Test | Result |
|---|---|
| VAE pod deleted mid-session | switched in about 3 s, 140/140 chunks, no pause over 1.5 s |
| VAE's Neuron device tainted NoExecute mid-session | switched in about 3 s, 140/140 chunks, no pause |
| New session after the last session's VAE was deleted while idle | 40/40 chunks |
| No VAE ready | 503 VAE_SERVICE_UNAVAILABLE |

Lessons from getting there:
- torch-neuronx's native RDMA backend binds a process to its first rendezvous store and caches EFA
  endpoints by transport rank for the life of the process; aborting a link to a dead peer, or a stale
  cancelled completion, broke the next link. NIXL isolates each transfer: a dead peer fails only the
  write in flight.
- With NIXL, do not `remove_remote_agent` a dead VAE: its cancelled write lands on the next VAE's first
  write. Park the connection; retry a failed write.
- Register Neuron memory by its runtime device address (`nrt_tensor_get_va`), not `tensor.data_ptr()`.
- NIXL builds without CUDA in about 30 s against the image's libfabric; the PyPI wheel is a CUDA build.

## Future work
- **Network isolation.** Today any pod in the cluster can call a VAE's `/link` and `/decode` or the DiT's
  `/session` (ClusterIP only; nothing on the load balancer). Add NetworkPolicies, e.g. `lbw-vae` accepts
  port 8000 only from `app=lbw-dit`, `lbw-dit` only from `app=lbw-web`. On EKS this needs the VPC CNI
  network policy agent enabled (`--enable-network-policy=true`; it is off on KEEP-trn3-dev1).
- **Authenticated pairing.** The `/link` reply hands out NIXL agent metadata without authentication; a
  per-session token from the DiT, checked by the VAE, would keep a foreign agent from attaching.
- **Many DiTs to many VAEs.** Tested 1 DiT : 2 VAEs. A VAE serves one session at a time; load-aware
  selection (least busy VAE) and VAE capacity limits are next.
- **First-chunk cost after a switch.** A new VAE starts its causal decode cache empty; carrying the cache
  over, or a short overlap, would remove the small visual glitch at the switch.
