"""One DiT process links to VAE A, then switches to VAE B (another process, another transport rank).
VAE_A_EXIT=1: A exits right after its round (it "dies" before the DiT switches).

    VAE pods: ROLE=vae VAE_RANK=<r> [EXIT_AFTER=1] torchrun --nproc_per_node=1 perf/rdma_switch.py
    DiT pod:  ROLE=dit VAE_A=<ip> VAE_B=<ip> torchrun --nproc_per_node=1 perf/rdma_switch.py
"""
import importlib.util as _u
import logging
import os
import sys
import time

import torch
import torch.distributed as dist

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wan", "rdma_link.py")
_s = _u.spec_from_file_location("rdma_link", _p); R = _u.module_from_spec(_s); _s.loader.exec_module(R)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
role = os.environ["ROLE"]
if role == "store":                            # a plain TCPStore server, no Neuron device
    st = dist.TCPStore("0.0.0.0", R.STORE_PORT, None, is_master=True, wait_for_workers=False, multi_tenant=True)
    print("SWITCH store up", flush=True); time.sleep(3600); sys.exit(0)
torch.neuron.set_device(0)
dist.init_process_group(backend="neuron")
dev = torch.device("neuron:0")
shape = (16, 3, 60, 96)
DIT = 8 * 3
SHARED = os.environ.get("STORE_HOST")          # one store for every process (the fix under test)
if role == "vae":
    store = R.client_store(SHARED, timeout_s=300) if SHARED else R.host_store()
    me = int(os.environ["VAE_RANK"])
    gen = int(store.get(f"gen/{me}"))
    link = R.RdmaLink("vae", store, gen, DIT, me, timeout_s=60)
    good = 0
    for i in range(5):
        t = torch.empty(shape, device=dev)
        try:
            link.recv(t).wait(); torch.neuron.synchronize(); good += int(bool((t.cpu() == float(gen * 10 + i)).all()))
        except Exception as e:
            print(f"SWITCH vae {me}: recv failed: {str(e)[:120]}", flush=True); break
    print(f"SWITCH vae {me}: gen {gen}: correct {good}/5", flush=True)
    if os.environ.get("EXIT_AFTER") == "1":
        print(f"SWITCH vae {me}: exiting (dies)", flush=True)
        os._exit(0)
    time.sleep(3600)
else:
    for k, host, vae_rank in ((0, os.environ["VAE_A"], 8 * 5), (1, os.environ["VAE_B"], 8 * 7)):
        store = R.client_store(SHARED or host, timeout_s=300)
        gen = 200 + k
        store.set(f"gen/{vae_rank}", str(gen))
        link = R.RdmaLink("dit", store, gen, DIT, vae_rank, timeout_s=60)
        try:
            for i in range(5):
                link.send(torch.full(shape, float(gen * 10 + i)).to(dev)).wait()
            torch.neuron.synchronize()
            print(f"SWITCH dit -> vae {vae_rank}: sent 5", flush=True)
        except Exception as e:
            print(f"SWITCH dit -> vae {vae_rank}: send failed: {str(e)[:120]}", flush=True)
        if k == 0:
            link.close()
            time.sleep(15)          # A may die meanwhile
    time.sleep(3600)
