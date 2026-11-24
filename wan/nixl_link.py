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
# LINGBOT_VAE_SESSIONS=K: a VAE serves up to K DiT sessions at once; each gets its own block of SLOTS buffers and tags
# its notifications with its session id ("s<sid>:<seq>"), so sessions never share a slot or a sequence number
SESSIONS = int(os.environ.get("LINGBOT_VAE_SESSIONS", "1"))
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


def _load_nixl():
    from nixl_cu12._api import nixl_agent, nixl_agent_config
    return nixl_agent, nixl_agent_config


# Load NIXL's native module when this module is imported, on the importing (main) thread, not on first use:
# in torchft the first use ran on a background thread while others ran, and loading the module there segfaulted
# 2 of 64 ranks (torchft 6bb8e0a). Here the first use can be the LINGBOT_NIXL_ASYNC worker. LINGBOT_NIXL_EAGER=0: off.
if os.environ.get("LINGBOT_NIXL_EAGER", "1") != "0" and os.environ.get("LINGBOT_VAE_TRANSPORT", "nixl") == "nixl":
    try:
        _load_nixl()
    except ImportError as _e:   # NIXL absent: fail when an agent is made, not at import
        logging.getLogger("nixl_link").warning(f"nixl: eager load failed: {_e}")


def _agent(prefix):
    nixl_agent, nixl_agent_config = _load_nixl()
    name = f"{prefix}-{uuid.uuid4().hex[:8]}"
    return nixl_agent(name, nixl_agent_config(backends=["LIBFABRIC"])), name


def _b64(b):
    return base64.b64encode(b).decode()


class NixlVae:
    """VAE rank 0: SLOTS receive buffers, and the notifications that say which one holds which chunk."""

    def __init__(self, shape, dev):
        self.buf = torch.zeros((SLOTS * SESSIONS,) + tuple(shape), dtype=torch.float32, device=dev)
        torch.neuron.synchronize()
        self.slot_bytes = self.buf[0].numel() * 4
        va = nrt_va(self.buf)
        self.agent, self.name = _agent("lbw-vae")
        self.agent.register_memory(self.agent.get_reg_descs([(va, SLOTS * SESSIONS * self.slot_bytes, 0, "")], "VRAM"))
        self.slot_descs = [_b64(self.agent.get_serialized_descs(
            self.agent.get_xfer_descs([(va + i * self.slot_bytes, self.slot_bytes, 0)], "VRAM")))
            for i in range(SLOTS * SESSIONS)]
        self.seen = set()
        log.info(f"nixl vae: agent {self.name}, {SLOTS} slots x {self.slot_bytes} B at {va:#x}")

    def info(self, sid=0):
        """Link info for session `sid`: its block of SLOTS slots (sid < SESSIONS)."""
        b = (sid % SESSIONS) * SLOTS
        return {"transport": "nixl", "meta": _b64(self.agent.get_agent_metadata()),
                "slots": self.slot_descs[b:b + SLOTS], "sid": sid}

    def wait(self, seq, timeout_s, sid=0):
        """The slot holding session `sid`'s chunk `seq`, once its write has landed; raises after timeout_s."""
        key = (sid, seq)
        t_end = time.time() + timeout_s
        while key not in self.seen:
            for msgs in self.agent.get_new_notifs().values():
                for m in msgs:
                    if m.startswith(b"seq:"):
                        self.seen.add((0, int(m[4:])))
                    elif m.startswith(b"s") and b":" in m:
                        a, b_ = m[1:].split(b":", 1)
                        self.seen.add((int(a), int(b_)))
            if key in self.seen:
                break
            if time.time() > t_end:
                raise TimeoutError(f"nixl: session {sid} chunk {seq} not written within {timeout_s} s")
            time.sleep(0.0002)
        self.seen.discard(key)
        return self.buf[(sid % SESSIONS) * SLOTS + seq % SLOTS]


class NixlDit:
    """DiT rank 0: one send buffer; a connection per VAE. The agent can be made before the buffer's shape is known
    (create_agent at startup, torchft f4194dc), so the first session does not wait for it."""

    def __init__(self, shape=None, dev=None):
        self.agent, self.name = _agent("lbw-dit")
        self.buf = None
        if shape is not None:
            self.ensure_buf(shape, dev)

    def ensure_buf(self, shape, dev):
        """Allocate and register the send buffer (once; the latent shape is fixed for the process's life)."""
        if self.buf is not None:
            return
        self.buf = torch.zeros(tuple(shape), dtype=torch.float32, device=dev)
        torch.neuron.synchronize()
        self.nbytes = self.buf.numel() * 4
        va = nrt_va(self.buf)
        self.agent.register_memory(self.agent.get_reg_descs([(va, self.nbytes, 0, "")], "VRAM"))
        self.local = self.agent.get_xfer_descs([(va, self.nbytes, 0)], "VRAM")
        log.info(f"nixl dit: agent {self.name}, send buffer {self.nbytes} B at {va:#x}")

    def connect(self, info):
        remote = self.agent.add_remote_agent(base64.b64decode(info["meta"]))
        slots = [self.agent.deserialize_descs(base64.b64decode(d)) for d in info["slots"]]
        return (remote, slots, int(info.get("sid", 0)))

    def disconnect(self, conn):
        """Park the connection; do NOT remove_remote_agent: that cancels the dead VAE's pending write, and
        the cancellation ("Operation canceled") landed on the next VAE's first write."""
        self.parked = getattr(self, "parked", [])
        self.parked.append(conn)

    def write(self, conn, x, seq, timeout_s=10):
        """Write chunk `seq` (device tensor x) into the VAE's slot seq % SLOTS; raises if it fails."""
        self.buf.copy_(x)
        torch.neuron.synchronize()
        remote, slots, sid = conn
        tag = f"s{sid}:{seq}" if sid else f"seq:{seq}"
        _tx = time.perf_counter()
        h = self.agent.initialize_xfer("WRITE", self.local, slots[seq % len(slots)], remote, tag.encode())
        try:
            st = self.agent.transfer(h)
            t_end = time.time() + timeout_s
            while st not in ("DONE", "ERR"):
                if time.time() > t_end:
                    raise TimeoutError(f"nixl write of chunk {seq} timed out")
                st = self.agent.check_xfer_state(h)
            if st != "DONE":
                raise RuntimeError(f"nixl write of chunk {seq}: {st}")
            self.last_xfer_ms = (time.perf_counter() - _tx) * 1000   # the one-sided write alone (after the device sync)
        finally:
            try:
                self.agent.release_xfer_handle(h)
            except Exception:
                pass
