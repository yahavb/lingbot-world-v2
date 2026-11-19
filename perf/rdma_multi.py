"""Checks the two things session-time VAE failover needs from wan/rdma_link.py:
  1. one process opens several links in a row (new gen, new transport ranks), each works;
  2. a receive whose peer never sends fails within the op timeout instead of hanging.

    VAE pod:  ROLE=vae torchrun --nproc_per_node=1 perf/rdma_multi.py
    DiT pod:  ROLE=dit VAE_HOST=<vae ip> torchrun --nproc_per_node=1 perf/rdma_multi.py
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
store = R.host_store() if role == "vae" else R.client_store(os.environ["VAE_HOST"], timeout_s=600)
shape = (16, 3, 60, 96)
for gen in (3, 4, 5):
    link = R.RdmaLink(role, store, gen, timeout_s=180)
    t_up = time.time()
    good = 0
    for i in range(5):
        v = float(gen * 100 + i)
        if role == "dit":
            link.send(torch.full(shape, v).to(dev)).wait()
        else:
            t = torch.empty(shape, device=dev); link.recv(t).wait(); good += int(bool((t.cpu() == v).all()))
    link.close()
    print(f"MULTI gen {gen}: 5 transfers in {time.time() - t_up:.1f} s" + (f", correct {good}/5" if role == "vae" else " sent"), flush=True)
# 2. timeout: the VAE waits for a tensor the DiT never sends
link = R.RdmaLink(role, store, 6, timeout_s=10)
if role == "vae":
    t = torch.empty(shape, device=dev); t0 = time.time()
    try:
        link.recv(t).wait()
        print("TIMEOUT test: recv returned (unexpected)", flush=True)
    except Exception as e:
        print(f"TIMEOUT test: recv failed after {time.time() - t0:.1f} s: {type(e).__name__}: {str(e)[:160]}", flush=True)
else:
    time.sleep(40)
print("MULTI done", flush=True)
time.sleep(5)
