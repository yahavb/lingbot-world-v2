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
agent = nixl_agent(f"{role}-{run}", nixl_agent_config(backends=["LIBFABRIC"]))
agent.register_memory(agent.get_reg_descs([(addr, nbytes, 0, "")], "VRAM"))
print(f"NIXL {role}: agent up, registered {nbytes} B of Neuron memory at {addr:#x}", flush=True)

if role == "vae":
    store.set(key("meta"), agent.get_agent_metadata())
    store.set(key("descs"), agent.get_serialized_descs(agent.get_xfer_descs([(addr, nbytes, 0)], "VRAM")))
    good, t0 = 0, time.time()
    exit_at = int(os.environ.get("EXIT_AT", "-1"))        # die mid-stream, with a write in flight
    for i in range(N):
        if i == exit_at:
            print(f"NIXL_PING vae-{run}: dying at write {i}", flush=True)
            os._exit(0)
        want = f"chunk-{i}".encode()
        while True:
            notifs = agent.get_new_notifs()
            if any(want in msgs for msgs in notifs.values()):
                break
            time.sleep(0.0002)
        good += int(bool((buf.cpu() == float(i + 1)).all()))
        store.set(key(f"ack/{i}"), b"1")
    print(f"NIXL_PING vae-{run}: {N} writes received, correct {good}/{N}", flush=True)
else:
    # RUNS=a,b: write to VAE a; when it dies (a write ERRs or is never acknowledged), drop it and go on with b
    ldescs = agent.get_xfer_descs([(addr, nbytes, 0)], "VRAM")
    for r in os.environ.get("RUNS", run).split(","):
        k = lambda x: f"nixl-ping/{r}/{x}"
        remote = agent.add_remote_agent(store.get(k("meta")))
        rdescs = agent.deserialize_descs(store.get(k("descs")))
        times, ok, why = [], 0, "all done"
        for i in range(N):
            buf.fill_(float(i + 1)); torch.neuron.synchronize()
            t0 = time.perf_counter()
            h = agent.initialize_xfer("WRITE", ldescs, rdescs, remote, f"chunk-{i}".encode())
            st = agent.transfer(h)
            t_end = time.time() + 10
            while st not in ("DONE", "ERR") and time.time() < t_end:
                st = agent.check_xfer_state(h)
            times.append(time.perf_counter() - t0)
            agent.release_xfer_handle(h)
            if st != "DONE":
                why = f"write {i}: {st}"; break
            try:
                store.wait([k(f"ack/{i}")], timedelta(seconds=5))
            except Exception:
                why = f"write {i}: no ack (peer gone)"; break
            ok += 1
        med = sorted(times)[len(times) // 2] if times else 0
        print(f"NIXL_PING dit -> vae-{r}: {ok} writes acknowledged, median {med * 1000:.2f} ms; stop: {why}", flush=True)
        agent.remove_remote_agent(remote)
time.sleep(5)
