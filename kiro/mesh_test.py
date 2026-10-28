"""Can this host build TP8 x CP collectives for a given device pairing? (LNC2)

    MESH_PAIRS="0-4,1-5,2-6,3-7,8-11,9-13,10-14,12-15" \
      torchrun --nproc_per_node=64 kiro/mesh_test.py
    MESH_TP=4 torchrun --nproc_per_node=64 kiro/mesh_test.py        # TP4 per device instead

Each pair (a-b) is one TP8 group: the 4 cores of device a then the 4 cores of device b. CP group t
is TP position t of every pair. Builds the (cp, tp) DeviceMesh from that rank layout, then runs an
eager all-reduce over TP and CP and a compiled (functional) all-reduce + all-gather over both,
checking values. torch sorts each group's ranks, so a pair's two devices get the same CP index
only if the pairs are listed so that both halves are increasing; this is asserted up front.
Without MESH_PAIRS it uses the default contiguous layout (pairs 0-1, 2-3, ...).
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch_neuronx  # noqa: F401
from torch.distributed.device_mesh import DeviceMesh

rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
# Device indices here are in RANK space (rank // 4). With NEURON_RT_VISIBLE_CORES set, local rank i
# runs on the i-th visible core, so rank-space device j is the j-th visible device.
if os.environ.get("MESH_TP", "8") == "4":
    # TP4 x CP(world/4): one TP group per device, CP across devices.
    pairs = None
    layout = [[4 * d + c for c in range(4)] for d in range(world // 4)]
else:
    spec = os.environ.get("MESH_PAIRS", ",".join(f"{2 * i}-{2 * i + 1}" for i in range(world // 8)))
    pairs = sorted(tuple(int(v) for v in p.split("-")) for p in spec.split(","))
    assert len(pairs) * 8 == world, f"{len(pairs)} pairs for world {world}"
    hi = [b for a, b in pairs]
    assert all(a < b for a, b in pairs) and hi == sorted(hi), \
        "pairs must be (lo, hi) with lo and hi increasing"
    layout = [[4 * a + c for c in range(4)] + [4 * b + c for c in range(4)] for a, b in pairs]

torch.neuron.set_device(local)
dist.init_process_group(backend="neuron")
dev = torch.device("neuron")
mesh = DeviceMesh("neuron", torch.tensor(layout), mesh_dim_names=("cp", "tp"))
tp_g, cp_g = mesh["tp"].get_group(), mesh["cp"].get_group()
tp_r, cp_r = dist.get_rank(tp_g), dist.get_rank(cp_g)
k = next(i for i, row in enumerate(layout) if rank in row)
t = layout[k].index(rank)
ok = (tp_r, cp_r) == (t, k)


def log(msg):
    if rank == 0:
        print(msg, flush=True)


log(f"[mesh_test] world {world}, TP{len(layout[0])} x CP{len(layout)}, pairs {pairs}, "
    f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES', 'all')}")
x = torch.full((4096,), float(rank), device=dev)
y = x.clone()
dist.all_reduce(y, group=tp_g)
z = x.clone()
dist.all_reduce(z, group=cp_g)
torch.neuron.synchronize()
exp_tp = float(sum(layout[k]))
exp_cp = float(sum(row[t] for row in layout))
ok &= y[0].item() == exp_tp and z[0].item() == exp_cp
log("[mesh_test] eager all-reduce over TP and CP: done")


@torch.compile(backend="neuron", fullgraph=True, dynamic=False)
def f(v):
    a = fc.all_reduce(v, "sum", mesh["tp"])
    g = fc.all_gather_tensor(v, 0, mesh["cp"])
    return a, g


a, g = f(x)
torch.neuron.synchronize()
ok &= a[0].item() == exp_tp
ok &= g.view(len(layout), -1)[:, 0].cpu().tolist() == [float(row[t]) for row in layout]
BIG = int(os.environ.get("MESH_BIG_MB", "0"))
if BIG:
    # Message size picks the collective algorithm, so also run DiT-sized (and T5-sized, world)
    # reductions; the log's last line names the step that failed if the runtime aborts.
    xb = torch.full((BIG * 1024 * 1024 // 4,), float(rank), device=dev)
    for nm, grp in (("tp", tp_g), ("cp", cp_g), ("world", None)):
        yb = xb.clone()
        dist.all_reduce(yb, group=grp)
        torch.neuron.synchronize()
        log(f"[mesh_test] eager all-reduce {BIG} MB over {nm}: done")

    @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
    def fb(v):
        return fc.all_reduce(v, "sum", mesh["tp"]), fc.all_gather_tensor(v, 0, mesh["cp"])

    ab, gb = fb(xb)
    torch.neuron.synchronize()
    ok &= ab[0].item() == exp_tp
    log(f"[mesh_test] compiled {BIG} MB all-reduce over tp + all-gather over cp: done")
res = torch.tensor([1.0 if ok else 0.0], device=dev)
dist.all_reduce(res)
torch.neuron.synchronize()
log(f"[mesh_test] RESULT {'PASS' if res.item() == world else 'FAIL'}: {int(res.item())}/{world} ranks "
    "with correct group ranks and collective values")
dist.destroy_process_group()
sys.exit(0)
