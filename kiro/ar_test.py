"""Minimal multi-rank all_reduce on Neuron: torchrun --nproc_per_node=N kiro/ar_test.py"""
import os

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
torch.neuron.set_device(local)
dist.init_process_group(backend="neuron", init_method="env://", rank=rank, world_size=world)
x = torch.full((1024,), float(rank + 1), device=f"neuron:{local}")
dist.all_reduce(x)
torch.neuron.synchronize()
print(f"rank {rank}/{world}: all_reduce -> {x[0].item()} (expect {world * (world + 1) / 2})", flush=True)
dist.destroy_process_group()
