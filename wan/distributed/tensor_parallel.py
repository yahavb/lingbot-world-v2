import os

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
from torch.distributed.device_mesh import init_device_mesh

_GROUPS = {}

# DeviceMesh-backed groups. A bare dist.new_group handle does NOT lower inside a
# compiled region -- the Neuron backend rejects it with "replica id #N not seen
# in replica groups", which model_fast.py records as a hard SDK constraint and
# responded to by keeping the whole DiT eager (944 per-op NEFFs).
#
# ~/the reference implementation (parallel.py):56-62 shows that is not a
# constraint on sub-groups at all, only on bare ProcessGroups:
#   "Subgroup collectives only lower inside a compiled region when the group
#    comes from a DeviceMesh: a bare dist.new_group knows only its own ranks and
#    the Neuron lowering rejects it."
# With a DeviceMesh plus torch.distributed._functional_collectives (out-of-place
# and traceable, unlike in-place dist.all_reduce) the collective stays IN the
# graph, which is what lets them compile whole 40-block stacks at fullgraph=True.
_MESH = None
_MESH_DIMS = {}


def init_mesh_groups(cp, tp, device_type="neuron", names=None, cp_fast=False):
    """Build a (cp, tp) DeviceMesh and register its submeshes as named groups.

    tp is the FAST (last) axis, so TP ranks are CONTIGUOUS and CP is strided --
    the layout the reference implementation uses, and the one their nccom-test numbers favour: an
    all-reduce over 4 consecutive ranks measured 119 GB/s versus 33.3 GB/s over
    4 strided ranks. TP carries the frequent, latency-sensitive reduce, so it
    belongs on the fast axis.

    Also registers the derived ProcessGroups under the same names so existing
    dist.* call sites keep working unchanged.
    """
    global _MESH, _MESH_DIMS
    assert dist.is_initialized(), "init_mesh_groups: init_process_group first"
    # AXIS ORDER. Default puts tp on the fast (last) axis, so TP ranks are contiguous and
    # CP is strided -- the reference implementation's layout, and their nccom numbers favour it: an all-reduce
    # over 4 consecutive ranks measured 119 GB/s against 33.3 GB/s over 4 strided ranks,
    # and TP carries the frequent latency-sensitive reduce.
    #
    # cp_fast=True swaps them, making CP contiguous and TP strided. Needed where a
    # collective INSIDE an NKI kernel must route over the CP group: with CP strided the
    # runtime cannot build one at all --
    #   CCOM WARN Failed to find device to device paths
    #   ENC:enc_init_comm failed (2) to init a collective algorithm.
    #     reason: no_hier no_mesh replica-group: Replica-group 0: [3,11,19,27]
    # and a framework all_gather over the same group fails identically, so it is the
    # topology and not an initialisation order.
    #
    # Correctness is unaffected either way: both named groups come from THIS mesh, so the
    # width-shard index and the gather that reassembles it cannot disagree -- which is the
    # transposition bug that produced garbage video at cp4 before.
    if cp_fast:
        _MESH = init_device_mesh(device_type, (tp, cp), mesh_dim_names=("tp", "cp"))
    else:
        _MESH = init_device_mesh(device_type, (cp, tp), mesh_dim_names=("cp", "tp"))
    _MESH_DIMS = {"cp": cp, "tp": tp}
    # Register the submeshes' OWN ProcessGroups under `names`, so the eager
    # dist.* call sites and the compiled functional collectives use the same
    # partition BY CONSTRUCTION.
    #
    # They did not, and it produced garbage video at cp4 while cp1 was fine.
    # init_product_mesh built "-sp" as a CONTIGUOUS block and "-tp" as STRIDED,
    # while this DeviceMesh has tp on the fast axis so "cp" is STRIDED and "tp"
    # CONTIGUOUS -- exact transposes. At world=32, tp=8, cp=4:
    #   rank 1: dit-sp peers [0,1,2,3] idx 1   vs  mesh cp peers [1,9,17,25] idx 0
    #   rank 8: dit-tp peers [0,4,8..] idx 2   vs  mesh tp peers [8..15]     idx 0
    # So a rank sliced its width by one index and reassembled it over a different
    # set of ranks, and weights were sharded by dit-tp rank but all-reduced over
    # mesh tp. At cp=1 both collapse to [0..tp) contiguous, which is why every cp1
    # run was correct and every cp4 run was not.
    if names is not None:
        cp_name, tp_name = names
        if tp > 1:
            register_group(tp_name, _MESH["tp"].get_group())
        if cp > 1:
            register_group(cp_name, _MESH["cp"].get_group())
    return _MESH


def mesh_submesh(dim):
    """The DeviceMesh submesh for "tp"/"cp", or None when no mesh was built."""
    if _MESH is None or _MESH_DIMS.get(dim, 1) <= 1:
        return None
    return _MESH[dim]


def has_mesh():
    return _MESH is not None


def all_gather_fn(tensor, gather_dim, dim="cp"):
    """Traceable all_gather that CONCATENATES along `gather_dim`.

    dist.all_gather takes a python list of output tensors and fills them in
    place, which Dynamo cannot trace -- inside fullgraph=True it is the same class
    of failure as async_op=True. fc.all_gather_tensor returns a single new tensor
    concatenated along gather_dim, so the CP K/V gather can live inside a compiled
    block.
    """
    m = mesh_submesh(dim)
    if m is None:
        return tensor
    # ALWAYS gather on dim 0, then reorder with plain tensor ops.
    #
    # fc.all_gather_tensor with gather_dim != 0 routes through
    # torch._utils._maybe_view_chunk_cat, which Dynamo refuses to trace:
    #   Unsupported: Attempted to call function marked as skipped
    #   module: torch._utils, qualname: _maybe_view_chunk_cat,
    #   skip reason: file matches MOD_SKIPLIST
    # Gathering on dim 0 avoids that helper entirely; the caller-visible result is
    # identical because rank r contributes slice r along gather_dim, so moving the
    # rank axis next to gather_dim and merging the two reproduces the concatenation.
    g = fc.all_gather_tensor(tensor.contiguous(), 0, m)
    if gather_dim == 0:
        return g
    n = dist.get_world_size(m.get_group())
    shape = list(tensor.shape)
    g = g.view([n] + shape)                      # [n, *orig]
    perm = [i + 1 for i in range(gather_dim + 1)]      # orig dims up to gather_dim
    perm = perm[:gather_dim] + [0] + [gather_dim + 1]  # ..., n, gather_dim
    perm += [i + 1 for i in range(gather_dim + 1, len(shape))]
    g = g.permute(*perm).contiguous()
    out_shape = shape[:gather_dim] + [shape[gather_dim] * n] + shape[gather_dim + 1:]
    return g.reshape(out_shape)


def replica_groups(dim="cp"):
    """All same-kind groups as list[list[int]] partitioning [0, world).

    Ported from the reference implementation (parallel.py):_build_replica_groups. The const-max AG
    NKI kernel does its all-gather INSIDE the kernel, so it needs the whole partition as
    an nki.collectives.ReplicaGroup, not just this rank's group.

    COLLECTIVE: every rank must call it (all_gather_object). Asserts the result really
    partitions the world, because that is exactly what the Neuron lowering checks and it
    reports the failure as a cryptic MLIR error instead.
    """
    m = mesh_submesh(dim)
    if m is None:
        return [list(range(dist.get_world_size()))]
    ranks = sorted(dist.get_process_group_ranks(m.get_group()))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, tuple(ranks))
    out = []
    for g in gathered:
        g = list(g)
        if g not in out:
            out.append(g)
    out.sort()
    flat = sorted(r for g in out for r in g)
    assert flat == list(range(dist.get_world_size())), (
        f"{dim} replica groups {out} do not partition world "
        f"{dist.get_world_size()} (flat={flat})")
    return out


def all_reduce_fn(tensor, dim="tp"):
    """Traceable, OUT-OF-PLACE all_reduce(SUM) over a DeviceMesh submesh.

    Use this inside compiled regions. dist.all_reduce is in-place and not
    traceable, so it forces a graph break or fails to lower; fc.all_reduce
    returns a new tensor and stays in the graph. Callers MUST assign the result.
    """
    m = mesh_submesh(dim)
    if m is None:
        return tensor
    if tensor.dtype in (torch.bfloat16, torch.float16) and not _AR_NATIVE_DTYPE:
        # Match all_reduce's fp32 promotion: stacked per-layer bf16 reductions
        # compound rounding (see all_reduce's docstring).
        return fc.all_reduce(tensor.float(), "sum", m).to(tensor.dtype)
    return fc.all_reduce(tensor, "sum", m)


# LINGBOT_AR_BF16=1: in-graph all-reduces of bf16 tensors stay bf16 (half the bytes). Wan2.2's
# compiled DiT reduces in bf16 and measured inside the bf16 accuracy band against an fp32 reference.
_AR_NATIVE_DTYPE = os.environ.get("LINGBOT_AR_BF16", "0") == "1"


def all_gather_tokens(x, dim="tp"):
    """Traceable all-gather of token shards [1, S, C] -> [1, n*S, C] over a mesh submesh, in rank
    order (Megatron sequence parallelism). Gathers on dim 0 of the 2-D view, which avoids the
    Dynamo-skipped chunk-cat helper (see all_gather_fn)."""
    m = mesh_submesh(dim)
    if m is None:
        return x
    assert x.shape[0] == 1, "token gather assumes batch 1"
    g = fc.all_gather_tensor(x[0].contiguous(), 0, m)
    return g.unsqueeze(0)


def cols_to_token_shard(x, dim="tp"):
    """Traceable column shard [1, L, C/n] -> this rank's token shard at full width [1, L/n, C],
    with ONE all-to-all. Same result as sp_token_shard(all_gather_last_dim(x)) -- a pure data move,
    so bit-identical -- but each rank receives L/n x C instead of L x C. Column order is rank
    order (shard_column_parallel's), token shards are contiguous in rank order (sp_token_shard's).
    """
    m = mesh_submesh(dim)
    if m is None:
        return x
    n = m.size()
    assert x.shape[0] == 1, "token shard assumes batch 1"
    _, L, c = x.shape
    assert L % n == 0, f"tokens {L} not divisible by {n}"
    # Rank r sends token block j to rank j and receives block r of every rank i, stacked in i order.
    y = fc.all_to_all_single(x[0].contiguous(), None, None, m)          # [n * L/n, c]
    return y.view(n, L // n, c).permute(1, 0, 2).reshape(1, L // n, n * c)


def reduce_scatter_tokens(x, dim="tp"):
    """Traceable reduce-scatter(SUM) [1, n*S, C] -> this rank's token shard [1, S, C]. Same dtype
    policy as all_reduce_fn: bf16 is promoted to fp32 for the sum unless LINGBOT_AR_BF16=1."""
    m = mesh_submesh(dim)
    if m is None:
        return x
    assert x.shape[0] == 1, "token reduce-scatter assumes batch 1"
    x2 = x[0].contiguous()
    if x2.dtype in (torch.bfloat16, torch.float16) and not _AR_NATIVE_DTYPE:
        return fc.reduce_scatter_tensor(x2.float(), "sum", 0, m).to(x.dtype).unsqueeze(0)
    return fc.reduce_scatter_tensor(x2, "sum", 0, m).unsqueeze(0)


def init_tp_group(name, tp_degree=None, ranks=None):
    """Create and register a named process group.

    Two forms:
      init_tp_group(name, tp_degree=k)  -- group over ranks [0, k) (legacy).
      init_tp_group(name, ranks=[...])  -- group over an explicit rank list
        (TRN3 sub-mesh carving: disjoint TP/SP/VAE groups over a larger world).

    No-op (leaves `name` unregistered) when the group would have <= 1 rank, so
    callers can unconditionally call this and check is_registered(name) after.
    Must be called after dist.init_process_group(), by EVERY rank in the
    world (dist.new_group is collective); ranks not in `ranks` simply get a
    non-member handle and skip registration.
    """
    if ranks is None:
        if (tp_degree or 1) <= 1:
            return
        ranks = list(range(tp_degree))
    if len(ranks) <= 1:
        return
    group = dist.new_group(ranks=list(ranks))
    if dist.get_rank() in ranks:
        register_group(name, group)


def init_product_mesh(name, sp_size, tp_size):
    """Create an sp_size x tp_size rank mesh over world ranks [0, sp_size*tp).

    Layout: global rank r lives in
      - TP group  r % tp_size   (peers {r%tp, r%tp + tp, ...}) -- column/row
        weight sharding + all_reduce (Megatron).
      - SP group  r // tp_size  (peers [ (r//tp)*tp, (r//tp+1)*tp )) -- Ulysses
        sequence/head all_to_all.

    Registers the CALLING rank's two groups as f"{name}-tp" and f"{name}-sp"
    (skipped when that axis is size 1). Every rank in the mesh must call this
    (dist.new_group is collective across the world). Returns (sp_size, tp_size).
    """
    assert dist.is_initialized(), "init_product_mesh: init_process_group first"
    rank = dist.get_rank()
    world = sp_size * tp_size
    # No rank-bound assert: dist.new_group is collective across the WHOLE
    # world, so every rank -- including T5/VAE ranks >= sp*tp in the full
    # layout -- must reach the new_group calls below. Non-mesh ranks simply
    # never satisfy `rank in members` and register nothing.
    for s in range(sp_size):
        members = list(range(s * tp_size, (s + 1) * tp_size))
        g = dist.new_group(ranks=members) if len(members) > 1 else None
        if g is not None and rank in members:
            register_group(f"{name}-sp", g)
    for t in range(tp_size):
        members = [t + i * tp_size for i in range(sp_size)]
        g = dist.new_group(ranks=members) if len(members) > 1 else None
        if g is not None and rank in members:
            register_group(f"{name}-tp", g)
    return sp_size, tp_size


def mesh_sp_group(name):
    return f"{name}-sp"


def mesh_tp_group(name):
    return f"{name}-tp"


def register_group(name, group):
    assert name not in _GROUPS, f"group {name!r} already registered"
    _GROUPS[name] = group


def destroy_group(name):
    assert name in _GROUPS, f"group {name!r} is not registered"
    del _GROUPS[name]


def is_registered(name):
    return name in _GROUPS


def _get(name):
    assert name in _GROUPS, f"group {name!r} is not registered"
    return _GROUPS[name]


def get_world_size(name):
    return dist.get_world_size(_get(name))


def get_rank(name):
    return dist.get_rank(_get(name))


# DO NOT put @torch.compiler.disable on these.
#
# ~/wan2-i2v-14b does exactly that (models/tp_utils.py:96) to "force a graph
# break -- the compiled FFN NEFF handles only matmuls, then all-reduce runs in
# eager", and porting it here broke the run at t5.py:357 with
#   torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function
# Their compiled regions omit fullgraph=True, so a break is legal. TWO of this
# repo's pre-existing compiled regions call all_reduce from INSIDE
# fullgraph=True, where a break is an error, not a fallback:
#   t5.py:152,180  -- the T5 encoder, compiled via utils/neuron_compile.py
#   model_fast.py:236 -- _tp_rms_norm
# So the guard is not merely unnecessary here, it is incompatible.
#
# The DiT compile does not need it either: compile_dit_selective targets
# collective-free leaves, and _tp_row_parallel already runs linear(x) BEFORE the
# all_reduce, so the matmul compiles and the collective stays in eager python
# with no break required.
def all_reduce(tensor, name, op=dist.ReduceOp.SUM):
    """SUM (by default) all-reduce. No-op if `name` isn't registered
    (single-process / tp_degree<=1 callers can call this unconditionally,
    no is_registered() guard needed at each call site).

    Reduces in float32 regardless of the input dtype: summing bf16/fp16
    partial results directly compounds rounding error across many stacked
    all-reduces (one row-parallel merge per attention/FFN call, every
    layer). Returns a tensor cast back to the original dtype; in-place
    only for inputs that are already float32.
    """
    if not is_registered(name):
        return tensor
    if tensor.dtype in (torch.bfloat16, torch.float16):
        orig_dtype = tensor.dtype
        reduced = tensor.float()
        dist.all_reduce(reduced, op=op, group=_get(name))
        return reduced.to(orig_dtype)
    dist.all_reduce(tensor, op=op, group=_get(name))
    return tensor


def all_reduce_native(tensor, name, op=dist.ReduceOp.SUM):
    """All-reduce in the tensor's OWN dtype -- no fp32 promotion.

    `all_reduce` deliberately promotes bf16/fp16 to fp32 so that stacked
    per-layer reductions don't compound rounding. This variant exists for the
    opposite trade: reducing a bf16 copy halves the wire volume. At 960*1664 the
    row-parallel all-reduce moves a [6195, 5120] fp32 tensor -- 121 MB, four
    times per block, ~19.4 GB per forward -- so the transport cost is worth
    measuring against the precision cost.

    Only use where that trade has been measured; `all_reduce` remains the
    default everywhere, and precision-sensitive reductions (e.g. the QK-norm
    sum-of-squares in _tp_rms_norm) must keep using it.
    """
    if not is_registered(name):
        return tensor
    t = tensor.contiguous()
    dist.all_reduce(t, op=op, group=_get(name))
    return t


def all_gather_last_dim(tensor, name):
    """Concatenate every rank's slice of a column-parallel output along the
    last (feature) dim, restoring the full-width tensor. No-op if `name`
    isn't registered, so tp_degree<=1 callers can call it unconditionally.

    The counterpart to `all_reduce` for the other Megatron half: use this when
    each rank holds a disjoint output-feature slice and the consumer needs the
    full width. Unlike `all_reduce` there is no dtype promotion here -- a
    gather is a pure copy with no accumulation, so bf16 in gives bf16 out with
    no rounding to compound.

    Slices concatenate in rank order, which matches
    `shard_column_parallel`'s `rank * chunk` offset, so the reconstruction is
    exact rather than merely equivalent.
    """
    if not is_registered(name):
        return tensor
    group = _get(name)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor
    local = tensor.contiguous()
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local, group=group)
    return torch.cat(parts, dim=-1).contiguous()


class AsyncReduce:
    """Handle wrapping an in-flight async all_reduce.

    wait() returns the reduced tensor downcast to the input's original dtype
    (the fp32-reduce-then-downcast mirrors all_reduce). The caller MUST wait()
    before the produced tensor's first consumer; the source input must not be
    freed or overwritten until after wait(). No-op-safe when the group isn't
    registered (wait() returns the input unchanged).
    """

    def __init__(self, work, reduced, orig_dtype):
        self._work = work
        self._reduced = reduced
        self._orig_dtype = orig_dtype

    def wait(self):
        if self._work is not None:
            self._work.wait()
        if self._orig_dtype in (torch.bfloat16, torch.float16):
            return self._reduced.to(self._orig_dtype)
        return self._reduced


class AsyncRowOut:
    """The delayed row-parallel output of a Linear + async all-reduce.

    Holds the async reduce handle plus the dtype the row-parallel result must
    come back as (the pre-Linear activation dtype, matching _tp_row_parallel).
    wait() returns the reduced tensor downcast to that dtype. Awaiting is the
    caller-side sync point before the result's first consumer.
    """

    __slots__ = ("_pend", "_out_dtype", "_done")

    @classmethod
    def already_done(cls, tensor, out_dtype):
        """An AsyncRowOut whose reduce has ALREADY completed synchronously.

        For compiled regions: async_op=True is not traceable (Dynamo raises
        "Unsupported: async_op=True for distributed collectives"), so a compiled
        block reduces synchronously through the DeviceMesh and wraps the result
        here, keeping the block's _async_out call sites unchanged.
        """
        obj = cls.__new__(cls)
        obj._pend = None
        obj._out_dtype = out_dtype
        obj._done = tensor
        return obj

    def __init__(self, pend, out_dtype):
        self._pend = pend
        self._out_dtype = out_dtype

    def wait(self):
        if getattr(self, "_done", None) is not None:
            return self._done
        if self._pend is None:
            return None
        return self._pend.wait().to(self._out_dtype)


def all_reduce_async(tensor, name, op=dist.ReduceOp.SUM):
    """Issue an all_reduce asynchronously; returns an AsyncReduce whose wait()
    yields the same value all_reduce would produce. All-reduce runs in fp32
    either way (matching all_reduce). No-op when `name` isn't registered."""
    if not is_registered(name):
        return AsyncReduce(None, tensor, tensor.dtype)
    group = _get(name)
    if tensor.dtype in (torch.bfloat16, torch.float16):
        reduced = tensor.float()
        work = dist.all_reduce(reduced, op=op, group=group, async_op=True)
        return AsyncReduce(work, reduced, tensor.dtype)
    work = dist.all_reduce(tensor, op=op, group=group, async_op=True)
    return AsyncReduce(work, tensor, tensor.dtype)


def tp_degree(name):
    """Current TP degree for `name`'s group, or 1 if unregistered."""
    return get_world_size(name) if is_registered(name) else 1


def tp_rank(name):
    """Current rank within `name`'s group, or 0 if unregistered."""
    return get_rank(name) if is_registered(name) else 0


def shard_column_parallel(weight, bias, rank, tp_size):
    """Slice along dim 0 (output features): q/k/v, ffn's first Linear, ...
    `bias` may be None (e.g. T5's Linears are all bias=False)."""
    out_features = weight.shape[0]
    assert out_features % tp_size == 0, (out_features, tp_size)
    chunk = out_features // tp_size
    start = rank * chunk
    w = weight[start:start + chunk].clone()
    b = bias[start:start + chunk].clone() if bias is not None else None
    return w, b


def shard_row_parallel(weight, bias, rank, tp_size):
    """Slice along dim 1 (input features): o, ffn's last Linear, ...

    Bias is kept only on rank 0 (zeroed on every other rank) so that
    summing every rank's partial output via all-reduce adds the bias
    exactly once, not tp_size times. `bias` may be None.
    """
    in_features = weight.shape[1]
    assert in_features % tp_size == 0, (in_features, tp_size)
    chunk = in_features // tp_size
    start = rank * chunk
    w = weight[:, start:start + chunk].clone()
    if bias is None:
        b = None
    else:
        b = bias.clone() if rank == 0 else torch.zeros_like(bias)
    return w, b


def shard_norm_vector(weight, rank, tp_size):
    """Slice a 1-D RMSNorm weight vector to match a column-parallel channel range."""
    n = weight.shape[0]
    assert n % tp_size == 0, (n, tp_size)
    chunk = n // tp_size
    start = rank * chunk
    return weight[start:start + chunk].clone()
