"""Point-to-point device-memory RDMA between two pods (lbw-dit rank 0 <-> one lbw-vae rank 0).

The same mechanism as torchft's Neuron process group (rpsilva-aws/torchft @ 5dda867,
torchft/neuron.py `_create_neuron_rdma_process_group`): a TCPStore both pods reach, and
torch-neuronx's native EFA backend (`create_neuron_cpp_backend`) registered on a plain
ProcessGroup, then `send` / `recv` of neuron tensors. Only the latents cross; the store
carries the rendezvous.

One link per (DiT, VAE) pairing, so a DiT can move to another VAE mid-stream. Each link has
a generation number `gen`: its rendezvous keys live under a PrefixStore of their own, and
both ends take fresh transport ranks (torch-neuronx caches EFA endpoints by transport rank
for the life of a process, which is why torchft gives a replacement pod a new transport
slot): DiT = 2 * gen, VAE = 2 * gen + 1, in a fixed domain of GLOBAL ranks.

Env: FI_PROVIDER=efa, FI_EFA_USE_DEVICE_RDMA=1, FI_EFA_FORK_SAFE=1, CCOM_SOCKET_IFNAME=eth0.
"""
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist

# Transport ranks: DiT = 2 * gen, VAE = 2 * gen + 1, in one fixed domain of GLOBAL ranks for the life of
# the process (the native backend sizes its endpoint tables once; a growing world size broke the 2nd link).
GLOBAL = int(os.environ.get("RDMA_GLOBAL_RANKS", "512"))
STORE_PORT = int(os.environ.get("RDMA_STORE_PORT", "29600"))


def host_store(timeout_s=None):
    """The VAE pod's one rendezvous store, for every link it will ever take."""
    t = timedelta(seconds=timeout_s or int(os.environ.get("RDMA_TIMEOUT_S", "3600")))
    return dist.TCPStore("0.0.0.0", STORE_PORT, None, is_master=True, timeout=t,
                         wait_for_workers=False, multi_tenant=True)


def client_store(host, timeout_s=60):
    """The DiT's connection to one VAE's store."""
    return dist.TCPStore(host, STORE_PORT, None, is_master=False,
                         timeout=timedelta(seconds=timeout_s), wait_for_workers=False)


class RdmaLink:
    """role "dit" (group rank 0) or "vae" (group rank 1), on `store` (a TCPStore)."""

    def __init__(self, role, store, gen, timeout_s=None):
        from torch._C._distributed_c10d import Backend as CBackend
        from torch_neuronx.distributed.cpp_backend import create_neuron_cpp_backend

        assert role in ("dit", "vae"), role
        self.rank = 0 if role == "dit" else 1
        self.peer = 1 - self.rank
        self.gen = int(gen)
        timeout = timedelta(seconds=timeout_s or int(os.environ.get("RDMA_OP_TIMEOUT_S", "60")))
        self.timeout = timeout
        pstore = dist.PrefixStore(f"lbw-rdma/{self.gen}", store)
        assert 0 <= self.gen < GLOBAL // 2, f"link gen {self.gen} outside the transport domain {GLOBAL}"
        global_ranks = [2 * self.gen, 2 * self.gen + 1]
        options = CBackend.Options(backend="neuron", timeout=timeout)
        options.global_ranks_in_group = global_ranks
        backend = create_neuron_cpp_backend(pstore, options, self.rank, 2, GLOBAL)
        pg = dist.ProcessGroup(pstore, self.rank, 2)
        bt = dist.ProcessGroup.BackendType.CUSTOM
        pg._set_default_backend(bt)
        pg._register_backend(torch.device("neuron"), bt, backend)
        self.pg = pg
        self.tag = 0
        logging.info(f"rdma link up: {role}, gen {self.gen}, transport ranks {global_ranks}")

    def send(self, t):
        """Post a send of device tensor t to the peer; returns the Work (call .wait())."""
        self.tag += 1
        return self.pg.send([t], self.peer, self.tag)

    def recv(self, t):
        """Post a receive into device tensor t from the peer; returns the Work."""
        self.tag += 1
        return self.pg.recv([t], self.peer, self.tag)

    def close(self):
        """Abort this link's native group (torchft aborts a generation before it builds the next)."""
        try:
            self.pg.abort()
        except Exception as e:  # a dead peer can make abort itself fail; the link is gone either way
            logging.warning(f"rdma link gen {self.gen}: abort: {type(e).__name__}: {e}")
