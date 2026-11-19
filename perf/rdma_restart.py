"""A VAE process that stays up, and DiT processes that come and go (each with a new transport rank):
does a new DiT process link to the long-lived VAE? (The failover case after a DiT restart.)

    VAE pod:  ROLE=vae torchrun --nproc_per_node=1 perf/rdma_restart.py
    DiT pod:  ROLE=dit VAE_HOST=<ip> DIT_RANK=<n> GEN=<g> torchrun --nproc_per_node=1 perf/rdma_restart.py
              (run several times, new DIT_RANK and GEN each time)
"""
import importlib.util as _u
import logging
import os
import time

import torch
import torch.distributed as dist

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wan", "rdma_link.py")
_s = _u.spec_from_file_location("rdma_link", _p); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
role = os.environ["ROLE"]
torch.neuron.set_device(0)
dist.init_process_group(backend="neuron")
dev = torch.device("neuron:0")
shape = (16, 3, 60, 96)
VAE_RANK = 1
if role == "vae":
    store = R.host_store()
    link = None
    for k in range(int(os.environ.get("ROUNDS", "3"))):
        gen = int(store.get(f"want/{k}"))          # the k-th DiT process announces (gen, its rank)
        dit_rank = int(store.get(f"rank/{k}"))
        if link is not None and os.environ.get("CLOSE", "1") == "1":
            link.close()
        link = R.RdmaLink("vae", store, gen, dit_rank, VAE_RANK, timeout_s=60)
        good = 0
        for i in range(5):
            t = torch.empty(shape, device=dev)
            try:
                link.recv(t).wait(); torch.neuron.synchronize(); good += int(bool((t.cpu() == float(gen * 10 + i)).all()))
            except Exception as e:
                print(f"RESTART vae round {k}: recv failed: {str(e)[:120]}", flush=True); break
        print(f"RESTART vae round {k}: DiT rank {dit_rank} gen {gen}: correct {good}/5", flush=True)
    time.sleep(5)
else:
    store = R.client_store(os.environ["VAE_HOST"], timeout_s=300)
    k, gen, rank = int(os.environ["ROUND"]), int(os.environ["GEN"]), int(os.environ["DIT_RANK"])
    store.set(f"want/{k}", str(gen)); store.set(f"rank/{k}", str(rank))
    link = R.RdmaLink("dit", store, gen, rank, VAE_RANK, timeout_s=60)
    try:
        for i in range(5):
            link.send(torch.full(shape, float(gen * 10 + i)).to(dev)).wait()
        torch.neuron.synchronize()
        print(f"RESTART dit round {k} rank {rank}: sent 5", flush=True)
    except Exception as e:
        print(f"RESTART dit round {k} rank {rank}: send failed: {str(e)[:120]}", flush=True)
