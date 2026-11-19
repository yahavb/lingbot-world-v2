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
slot): each process has one transport rank for its life, a restarted pod a new one.

Env: FI_PROVIDER=efa, FI_EFA_USE_DEVICE_RDMA=1, FI_EFA_FORK_SAFE=1, CCOM_SOCKET_IFNAME=eth0.
"""
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist

# Every process has ONE transport rank for its life, in a fixed domain of GLOBAL ranks (the native backend
# caches EFA endpoints by transport rank per process; a new rank in an old process never connected).
GLOBAL = int(os.environ.get("RDMA_GLOBAL_RANKS", "512"))
STORE_PORT = int(os.environ.get("RDMA_STORE_PORT", "29600"))
# torchft lays transport ranks out as slot * stride + local rank (stride = the pod's world), so a rank's
# residue is the process's local rank. Our link ends are always local rank 0, so a rank is a multiple of
# SLOT (a multiple of both pods' worlds: DiT 8, VAE 4).
SLOT = int(os.environ.get("RDMA_SLOT", "8"))


def new_transport_rank(rng):
    """A fresh transport rank for this process: a random slot, times SLOT (local rank 0)."""
    return SLOT * rng.randrange(1, GLOBAL // SLOT)


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

    def __init__(self, role, store, gen, dit_rank, vae_rank, timeout_s=None):
        from torch._C._distributed_c10d import Backend as CBackend
        from torch_neuronx.distributed.cpp_backend import create_neuron_cpp_backend

        assert role in ("dit", "vae"), role
        self.rank = 0 if role == "dit" else 1
        self.peer = 1 - self.rank
        self.gen = int(gen)
        timeout = timedelta(seconds=timeout_s or int(os.environ.get("RDMA_OP_TIMEOUT_S", "60")))
        self.timeout = timeout
        pstore = dist.PrefixStore(f"lbw-rdma/{self.gen}", store)
        # transport ranks are per PROCESS, fixed for its life (torch-neuronx caches EFA endpoints by them):
        # a new link between the same two processes reuses them; a restarted pod brings a new one.
        global_ranks = [int(dit_rank), int(vae_rank)]
        assert all(0 <= r < GLOBAL for r in global_ranks) and global_ranks[0] != global_ranks[1], global_ranks
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
