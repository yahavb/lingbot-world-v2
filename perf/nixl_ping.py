"""NIXL ping between two pods: the DiT side WRITEs latent-sized Neuron tensors one-sided into a buffer the
VAE side registered, each write with a notification; the VAE checks every value. LIBFABRIC backend (EFA,
FI_HMEM_NEURON). Metadata goes through the shared rendezvous store (lbw-rdma-store).

    VAE pod: ROLE=vae python3 perf/nixl_ping.py      DiT pod: ROLE=dit python3 perf/nixl_ping.py
    (after eval "$(bash tools/nixl/build_nixl.sh)")
"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from nixl_cu12._api import nixl_agent, nixl_agent_config

role = os.environ["ROLE"]
N = int(os.environ.get("N", "50"))
torch.neuron.set_device(0)
dev = torch.device("neuron:0")
shape = (16, 3, 60, 96)
nbytes = 4 * 16 * 3 * 60 * 96
host = os.environ.get("RDMA_STORE_HOST", "lbw-rdma-store.default.svc.cluster.local")
store = dist.TCPStore(host, 29600, None, is_master=False, timeout=timedelta(seconds=600), wait_for_workers=False)
run = os.environ.get("RUN", "1")
key = lambda k: f"nixl-ping/{run}/{k}"

def nrt_va(t):
    """The Neuron runtime's device address of tensor t (what NIXL and libfabric FI_HMEM_NEURON need; a
    torch-neuronx tensor's data_ptr() is not it): nrt_tensor_get_va on the tensor's nrt_tensor_t, + offset."""
    import ctypes
    import torch_neuronx
    cap = torch_neuronx._C._get_nrt_tensor_capsule(t)
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype, api.PyCapsule_GetName.argtypes = ctypes.c_char_p, [ctypes.py_object]
    api.PyCapsule_GetPointer.restype, api.PyCapsule_GetPointer.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    ptr = api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))
    nrt = ctypes.CDLL("libnrt.so.1")
    nrt.nrt_tensor_get_va.restype, nrt.nrt_tensor_get_va.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
    return nrt.nrt_tensor_get_va(ptr) + t.storage_offset() * t.element_size()


buf = torch.zeros(shape, dtype=torch.float32, device=dev)
torch.neuron.synchronize()
addr = nrt_va(buf)
print(f"NIXL {role}: data_ptr {buf.data_ptr():#x}, nrt va {addr:#x}", flush=True)
agent = nixl_agent(role, nixl_agent_config(backends=["LIBFABRIC"]))
agent.register_memory(agent.get_reg_descs([(addr, nbytes, 0, "")], "VRAM"))
print(f"NIXL {role}: agent up, registered {nbytes} B of Neuron memory at {addr:#x}", flush=True)

if role == "vae":
    store.set(key("meta"), agent.get_agent_metadata())
    store.set(key("descs"), agent.get_serialized_descs(agent.get_xfer_descs([(addr, nbytes, 0)], "VRAM")))
    good, t0 = 0, time.time()
    for i in range(N):
        want = f"chunk-{i}".encode()
        while True:
            notifs = agent.get_new_notifs()
            if any(want in msgs for msgs in notifs.values()):
                break
            time.sleep(0.0002)
        good += int(bool((buf.cpu() == float(i + 1)).all()))
        store.set(key(f"ack/{i}"), b"1")
    print(f"NIXL_PING vae: {N} writes received, correct {good}/{N}", flush=True)
else:
    remote = agent.add_remote_agent(store.get(key("meta")))
    rdescs = agent.deserialize_descs(store.get(key("descs")))
    ldescs = agent.get_xfer_descs([(addr, nbytes, 0)], "VRAM")
    times = []
    for i in range(N):
        buf.fill_(float(i + 1)); torch.neuron.synchronize()
        t0 = time.perf_counter()
        h = agent.initialize_xfer("WRITE", ldescs, rdescs, remote, f"chunk-{i}".encode())
        st = agent.transfer(h)
        while st not in ("DONE", "ERR"):
            st = agent.check_xfer_state(h)
        times.append(time.perf_counter() - t0)
        agent.release_xfer_handle(h)
        if st == "ERR":
            print(f"NIXL_PING dit: write {i} ERR", flush=True); break
        store.get(key(f"ack/{i}"))           # the VAE has checked this value before the buffer changes
    med = sorted(times)[len(times) // 2]
    print(f"NIXL_PING dit: {len(times)} writes, median {med * 1000:.2f} ms ({nbytes / med / 1e9:.2f} GB/s)", flush=True)
time.sleep(5)
