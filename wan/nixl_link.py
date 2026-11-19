"""DiT -> VAE latents over NIXL (ai-dynamo/nixl, LIBFABRIC backend: EFA, Neuron device memory).

The VAE's rank 0 registers SLOTS latent-sized buffers in Neuron memory and hands its agent metadata and
the buffers' descriptors to a DiT in the reply to POST /link. The DiT's rank 0 WRITEs each chunk one-sided
into slot seq % SLOTS with the notification b"seq:<n>"; the VAE waits for that notification before it
decodes chunk n. A dead VAE fails only the write in flight (perf/nixl_ping.py: no stale errors on the next
peer), so the DiT just moves on to another VAE.

Memory is registered by its Neuron runtime device address (nrt_tensor_get_va); a torch-neuronx tensor's
data_ptr() is not that address. Build: tools/nixl/build_nixl.sh (once per node, cached).
"""
import base64
import ctypes
import logging
import os
import time
import uuid

import torch

SLOTS = int(os.environ.get("NIXL_SLOTS", "4"))
log = logging.getLogger("nixl_link")


def nrt_va(t):
    """The Neuron runtime device address of tensor t: nrt_tensor_get_va on its nrt_tensor_t, + offset."""
    import torch_neuronx
    cap = torch_neuronx._C._get_nrt_tensor_capsule(t)
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype, api.PyCapsule_GetName.argtypes = ctypes.c_char_p, [ctypes.py_object]
    api.PyCapsule_GetPointer.restype, api.PyCapsule_GetPointer.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    ptr = api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))
    nrt = ctypes.CDLL("libnrt.so.1")
    nrt.nrt_tensor_get_va.restype, nrt.nrt_tensor_get_va.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
    return nrt.nrt_tensor_get_va(ptr) + t.storage_offset() * t.element_size()


def _agent(prefix):
    from nixl_cu12._api import nixl_agent, nixl_agent_config
    name = f"{prefix}-{uuid.uuid4().hex[:8]}"
    return nixl_agent(name, nixl_agent_config(backends=["LIBFABRIC"])), name


def _b64(b):
    return base64.b64encode(b).decode()


class NixlVae:
    """VAE rank 0: SLOTS receive buffers, and the notifications that say which one holds which chunk."""

    def __init__(self, shape, dev):
        self.buf = torch.zeros((SLOTS,) + tuple(shape), dtype=torch.float32, device=dev)
        torch.neuron.synchronize()
        self.slot_bytes = self.buf[0].numel() * 4
        va = nrt_va(self.buf)
        self.agent, self.name = _agent("lbw-vae")
        self.agent.register_memory(self.agent.get_reg_descs([(va, SLOTS * self.slot_bytes, 0, "")], "VRAM"))
        self.slot_descs = [_b64(self.agent.get_serialized_descs(
            self.agent.get_xfer_descs([(va + i * self.slot_bytes, self.slot_bytes, 0)], "VRAM"))) for i in range(SLOTS)]
        self.seen = set()
        log.info(f"nixl vae: agent {self.name}, {SLOTS} slots x {self.slot_bytes} B at {va:#x}")

    def info(self):
        return {"transport": "nixl", "meta": _b64(self.agent.get_agent_metadata()), "slots": self.slot_descs}

    def wait(self, seq, timeout_s):
        """The slot holding chunk `seq`, once its write has landed; raises after timeout_s."""
        t_end = time.time() + timeout_s
        while seq not in self.seen:
            for msgs in self.agent.get_new_notifs().values():
                for m in msgs:
                    if m.startswith(b"seq:"):
                        self.seen.add(int(m[4:]))
            if seq in self.seen:
                break
            if time.time() > t_end:
                raise TimeoutError(f"nixl: chunk {seq} not written within {timeout_s} s")
            time.sleep(0.0002)
        self.seen.discard(seq)
        return self.buf[seq % SLOTS]


class NixlDit:
    """DiT rank 0: one send buffer; a connection per VAE."""

    def __init__(self, shape, dev):
        self.buf = torch.zeros(tuple(shape), dtype=torch.float32, device=dev)
        torch.neuron.synchronize()
        self.nbytes = self.buf.numel() * 4
        va = nrt_va(self.buf)
        self.agent, self.name = _agent("lbw-dit")
        self.agent.register_memory(self.agent.get_reg_descs([(va, self.nbytes, 0, "")], "VRAM"))
        self.local = self.agent.get_xfer_descs([(va, self.nbytes, 0)], "VRAM")
        log.info(f"nixl dit: agent {self.name}, send buffer {self.nbytes} B at {va:#x}")

    def connect(self, info):
        remote = self.agent.add_remote_agent(base64.b64decode(info["meta"]))
        slots = [self.agent.deserialize_descs(base64.b64decode(d)) for d in info["slots"]]
        return (remote, slots)

    def disconnect(self, conn):
        """Park the connection; do NOT remove_remote_agent: that cancels the dead VAE's pending write, and
        the cancellation ("Operation canceled") landed on the next VAE's first write."""
        self.parked = getattr(self, "parked", [])
        self.parked.append(conn)

    def write(self, conn, x, seq, timeout_s=10):
        """Write chunk `seq` (device tensor x) into the VAE's slot seq % SLOTS; raises if it fails."""
        self.buf.copy_(x)
        torch.neuron.synchronize()
        remote, slots = conn
        h = self.agent.initialize_xfer("WRITE", self.local, slots[seq % len(slots)], remote, f"seq:{seq}".encode())
        try:
            st = self.agent.transfer(h)
            t_end = time.time() + timeout_s
            while st not in ("DONE", "ERR"):
                if time.time() > t_end:
                    raise TimeoutError(f"nixl write of chunk {seq} timed out")
                st = self.agent.check_xfer_state(h)
            if st != "DONE":
                raise RuntimeError(f"nixl write of chunk {seq}: {st}")
        finally:
            try:
                self.agent.release_xfer_handle(h)
            except Exception:
                pass
