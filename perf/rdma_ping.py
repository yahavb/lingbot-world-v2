"""RDMA ping between two pods with wan/rdma_link.py: the DiT side sends latent-sized tensors,
the VAE side receives them and checks every value, then the timing of N transfers.

    VAE pod:  ROLE=vae RDMA_STORE=0.0.0.0:29600 torchrun --nproc_per_node=1 perf/rdma_ping.py
    DiT pod:  ROLE=dit RDMA_STORE=<vae ip>:29600 torchrun --nproc_per_node=1 perf/rdma_ping.py
"""
import logging
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wan.rdma_link import RdmaLink  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
role = os.environ["ROLE"]
torch.neuron.set_device(0)
dist.init_process_group(backend="neuron")          # the pod's own world (1 rank here)
dev = torch.device("neuron:0")
link = RdmaLink(role, dit_world=int(os.environ.get("DIT_WORLD", "8")))
shape = (16, 3, 60, 96)                            # one chunk's latent
N = int(os.environ.get("N", "50"))
ok, times = 0, []
for i in range(N):
    if role == "dit":
        t = torch.full(shape, float(i), dtype=torch.float32).to(dev)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        link.send(t).wait()
        torch.neuron.synchronize()
        times.append(time.perf_counter() - t0)
    else:
        t = torch.empty(shape, dtype=torch.float32, device=dev)
        t0 = time.perf_counter()
        link.recv(t).wait()
        torch.neuron.synchronize()
        times.append(time.perf_counter() - t0)
        ok += int(bool((t.cpu() == float(i)).all()))
mb = 4 * 16 * 3 * 60 * 96 / 1e6
med = sorted(times)[len(times) // 2]
print(f"RDMA_PING role={role} n={N} {mb:.2f} MB each, median {med*1000:.2f} ms "
      f"({mb / med / 1000:.2f} GB/s)" + (f", correct {ok}/{N}" if role == "vae" else ""), flush=True)
