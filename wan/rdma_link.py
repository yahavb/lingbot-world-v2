"""Point-to-point device-memory RDMA between two pods (lbw-dit rank 0 <-> lbw-vae rank 0).

The same mechanism as torchft's Neuron process group (rpsilva-aws/torchft @ 5dda867,
torchft/neuron.py `_create_neuron_rdma_process_group`): a TCPStore both pods reach, and
torch-neuronx's native EFA backend (`create_neuron_cpp_backend`) registered on a plain
ProcessGroup, then `send` / `recv` of neuron tensors. Only the latents cross; the TCPStore
carries the rendezvous.

Transport ranks follow torchft's layout (pod slot x pod world + local rank): the DiT pod is
slot 0, the VAE pod slot 1, so DiT rank 0 is transport rank 0 and VAE rank 0 is transport
rank `dit_world` (passed in), which keeps the two pods' endpoint ids apart.

Env (both pods): RDMA_STORE=<host>:<port> (the VAE hosts the store), FI_PROVIDER=efa,
FI_EFA_USE_DEVICE_RDMA=1, FI_EFA_FORK_SAFE=1, CCOM_SOCKET_IFNAME=eth0.
"""
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist


class RdmaLink:
    """role "dit" (group rank 0, store client) or "vae" (group rank 1, store host)."""

    def __init__(self, role, dit_world, store=None, timeout_s=600):
        from torch._C._distributed_c10d import Backend as CBackend
        from torch_neuronx.distributed.cpp_backend import create_neuron_cpp_backend

        assert role in ("dit", "vae"), role
        host, port = (store or os.environ["RDMA_STORE"]).rsplit(":", 1)
        self.rank = 0 if role == "dit" else 1
        self.peer = 1 - self.rank
        timeout = timedelta(seconds=timeout_s)
        self.store = dist.TCPStore(host if role == "dit" else "0.0.0.0", int(port), 2,
                                   is_master=(role == "vae"), timeout=timeout,
                                   wait_for_workers=False)
        global_ranks = [0, int(dit_world)]
        options = CBackend.Options(backend="neuron", timeout=timeout)
        options.global_ranks_in_group = global_ranks
        backend = create_neuron_cpp_backend(self.store, options, self.rank, 2, max(global_ranks) + 1)
        pg = dist.ProcessGroup(self.store, self.rank, 2)
        bt = dist.ProcessGroup.BackendType.CUSTOM
        pg._set_default_backend(bt)
        pg._register_backend(torch.device("neuron"), bt, backend)
        self.pg = pg
        self.tag = 0
        logging.info(f"rdma link up: {role} group rank {self.rank}, transport ranks {global_ranks}, "
                     f"store {host}:{port}")

    def send(self, t):
        """Post a send of device tensor t to the peer; returns the Work (call .wait())."""
        self.tag += 1
        return self.pg.send([t], self.peer, self.tag)

    def recv(self, t):
        """Post a receive into device tensor t from the peer; returns the Work."""
        self.tag += 1
        return self.pg.recv([t], self.peer, self.tag)
