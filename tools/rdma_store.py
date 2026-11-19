"""lbw-rdma-store: the one TCPStore every lbw-dit and lbw-vae process rendezvouses on (wan/rdma_link.py).
No Neuron device; run with TORCH_DEVICE_BACKEND_AUTOLOAD=0."""
import os
import time
from datetime import timedelta

import torch.distributed as dist

store = dist.TCPStore("0.0.0.0", int(os.environ.get("RDMA_STORE_PORT", "29600")), None, is_master=True,
                      timeout=timedelta(hours=24), wait_for_workers=False, multi_tenant=True)
print("lbw-rdma-store: listening", flush=True)
while True:
    time.sleep(3600)
