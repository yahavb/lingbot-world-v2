"""Width-sharded decoder for Wan2_1_VAE (LingBot-World v2).

Every rank holds a full, unsharded copy of the decoder. The input latent
is split along width, each rank decodes only its own slice, and the
results are stitched back together (gather_video_width()). No weights
are sliced -- this shards the activation, not the model.

Two things need cross-rank communication:
  - Convolutions with a >1-wide kernel along width: exchange a halo with
    the neighboring ranks before convolving (_halo_exchange_w).
  - AttentionBlock (global receptive field): all-gather the full width,
    run attention redundantly on every rank, slice back to local width.

The halo/attention math is @torch.compile(backend="neuron", dynamic=False).
"""

import os
import time

import torch
import torch.distributed as dist


# fullgraph=True on the leaves below, for the same reason the DiT uses it
# (_compile_leaf in model_fast.py): they contain no collective, so a graph break is
# neither needed nor wanted, and fullgraph makes a silent sub-graph split impossible
# instead of something inferred from a log line.
#
# NOTE what this does NOT fix. Every collective here is still OUTSIDE the compiled
# regions -- _halo_exchange_w and _all_gather_w use dist.all_gather_into_tensor on a
# bare ProcessGroup (_VAE_GROUP), which cannot lower inside a compiled region at all;
# the reference implementation's parallel.py:56-62 is explicit that a subgroup collective only lowers when
# it comes from a DeviceMesh. So the decoder is ~30 tiny NEFFs per frame stitched
# together with ~30 eager all-gathers, which is exactly the shape the DiT was in
# before the mesh work. VAE_COLLECTIVE_STATS counts them so the cost is measured
# rather than assumed.
_COLL = {"halo": 0, "gather": 0, "halo_s": 0.0, "gather_s": 0.0}


_VAE_COMPILED = []


def _vae_compile(fn):
    # Say so in the log. The DiT announces "compiled 6 destination-specialised x
    # 40-block stacks"; the VAE announced nothing, so whether fullgraph was actually on
    # could only be inferred from the absence of errors, which is not evidence.
    _VAE_COMPILED.append(getattr(fn, "__name__", type(fn).__name__))
    return torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)


def vae_compile_report():
    """What is compiled in the width-sharded decode, and what is NOT.

    Three leaves, not a whole-decoder graph. The DiT's win came from going from leaves
    to one 40-block fullgraph stack per cache destination; the VAE is still at the leaf
    stage, with the norms, activations, ResidualBlock glue, Resample interpolate and the
    per-frame feat_cache all eager, and every collective outside the graph because
    dist.all_gather_into_tensor on a bare ProcessGroup cannot lower inside one.
    """
    return (f"VAE compiled regions: {len(_VAE_COMPILED)} leaves "
            f"({', '.join(_VAE_COMPILED)}) fullgraph=True; "
            f"decoder body, norms/activations and all collectives remain eager")


def vae_collective_stats():
    """(counts, seconds) of collectives issued outside every compiled region."""
    return dict(_COLL)


def reset_vae_collective_stats():
    for k in _COLL:
        _COLL[k] = 0 if isinstance(_COLL[k], int) else 0.0
import torch.nn as nn
import torch.nn.functional as F

from .vae2_1 import AttentionBlock, CausalConv3d, ResidualBlock, Resample


_VAE_GROUP = None

# LINGBOT_VAE_HALO=hier: exchange width halos hierarchically instead of all-gathering every
# rank's edges over the whole VAE group. At 64 ranks the decode issues 693 halo exchanges;
# as world all-gathers they cost ~1.1 s of a ~2.95 s decode (measured by skipping them,
# kiro/vae_bench.py VAE_DIAG=halo), while each rank needs only 2 of the 64 edges it gets.
#   step 1: all-gather the edges over the ranks of one device (a TP4-shaped group), which
#           gives every rank its neighbours on the same device and the device's two
#           boundary edges;
#   step 2: the first and the last rank of each device all-gather their device's boundary
#           edges over one-rank-per-device groups (CP16-shaped, a ring over all 16 devices),
#           which gives them the neighbouring devices' boundary edges.
# Both group shapes are ones the DiT already uses here. 2-rank groups do not work: a group of
# cores 1 and 2 of one device fails collective setup, as does a pair of unlinked devices.
# Pure data movement, so the output is bit-identical. Needs the default core order (rank r on
# device r // ranks_per_device) and the whole-world VAE group on 16 devices; anything else
# keeps the all-gather.
_HALO_MODE = os.environ.get("LINGBOT_VAE_HALO", "allgather")
_HIER = None   # (ranks per device, this device's group, first-rank group, last-rank group)


def _hier_active():
    if _HALO_MODE != "hier" or _VAE_GROUP is not None or not dist.is_initialized():
        return False
    w = dist.get_world_size()
    return w % 16 == 0 and w // 16 >= 2


def _hier_groups():
    """new_group is collective over the world: every rank builds every group, in order."""
    global _HIER
    if _HIER is None:
        w, r = dist.get_world_size(), dist.get_rank()
        per = w // 16
        dev = None
        for d in range(16):
            g = dist.new_group(ranks=list(range(per * d, per * d + per)))
            if r // per == d:
                dev = g
        first = dist.new_group(ranks=[per * d for d in range(16)])
        last = dist.new_group(ranks=[per * d + per - 1 for d in range(16)])
        _HIER = (per, dev, first, last)
    return _HIER


def _world():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(_VAE_GROUP)


def _rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank(_VAE_GROUP)


def _set_vae_group(group):
    global _VAE_GROUP
    _VAE_GROUP = group


def width_shard_latent(latent, rank=None, world=None):
    """Slice a [..., W] latent (or any tensor) into this rank's width chunk.

    Call before feeding the latent to a width-sharded decoder.
    """
    rank = _rank() if rank is None else rank
    world = _world() if world is None else world
    width = latent.shape[-1]
    assert width % world == 0, f"width={width} not divisible by world={world}"
    chunk = width // world
    return latent[..., rank * chunk:(rank + 1) * chunk].contiguous()


def gather_video_width(video_local, frame_dim=1):
    """Reassemble the full-width video, ONE FRAME AT A TIME, onto the host.

    Gathering the whole clip in one call is what OOMs at 768*1152:

      RuntimeError: NRT Execution error occurred on Neuron for operation=allgather
      Failed to allocate 1363148800 bytes on DEVICE     (1.27 GiB)

    _all_gather_w allocates a full-width buffer AND a .contiguous() copy after its
    permute, so a 3x81x768x1280 fp32 video costs 0.89 GiB twice -- ~1.78 GiB peak on a
    core that already holds the decoder. Per frame it is 11 MiB, and each gathered frame
    moves straight to host.

    This is the reference implementation's own diagnosis of the same failure (the reference benchmark:143): "the growth
    was per-call output held on device for all 485 frames at once", fixed there with
    `frames.extend(list(out.frames))  # move to host, free the device copy`.

    Returns a CPU tensor on rank 0 (None elsewhere, as before). The caller writes it to
    disk, so host is where it wants to be anyway.
    """
    n = video_local.shape[frame_dim]
    if n <= 1:
        out = _all_gather_w(video_local)
        return out.cpu() if out is not None else None
    frames = []
    for i in range(n):
        f = _all_gather_w(video_local.narrow(frame_dim, i, 1).contiguous())
        frames.append(f.cpu() if f is not None else None)
    if any(f is None for f in frames):
        return None
    return torch.cat(frames, dim=frame_dim)


def _extract_edges(x, radius):
    """Return (2, *x.shape[:-1], radius): index 0 = left edge, 1 = right edge."""
    left = x[..., :radius]
    right = x[..., -radius:]
    return torch.stack([left, right], dim=0).contiguous()


def _halo_exchange_w(x, radius):
    """All-gather the left/right `radius`-wide edges and return this rank's
    (halo_left, halo_right) -- the neighbors' adjoining edges, or None at the
    two ends of the unsharded width where there is no neighbor."""
    world = _world()
    if world == 1:
        return None, None
    rank = _rank()

    _t0 = time.perf_counter()
    edges_local = _extract_edges(x, radius)
    if _hier_active():
        per, dev_g, first_g, last_g = _hier_groups()
        c, d = rank % per, rank // per          # core within the device, device
        # Step 1: this device's slices' edges, [per, 2, ...] (index 0 = left, 1 = right edge).
        dev_edges = torch.empty((per * edges_local.shape[0],) + edges_local.shape[1:],
                                dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(dev_edges, edges_local, group=dev_g)
        dev_edges = dev_edges.reshape((per,) + edges_local.shape)
        halo_left = dev_edges[c - 1, 1] if c > 0 else None
        halo_right = dev_edges[c + 1, 0] if c < per - 1 else None
        if c == 0 or c == per - 1:
            # Step 2: the device's boundary edges -- its first slice's left edge and its last
            # slice's right edge -- from every device, [16, 2, ...].
            bnd = torch.stack([dev_edges[0, 0], dev_edges[per - 1, 1]], dim=0).contiguous()
            all_bnd = torch.empty((16 * bnd.shape[0],) + bnd.shape[1:],
                                  dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(all_bnd, bnd, group=first_g if c == 0 else last_g)
            all_bnd = all_bnd.reshape((16,) + bnd.shape)
            if c == 0 and d > 0:
                halo_left = all_bnd[d - 1, 1]       # previous device's right boundary
            if c == per - 1 and d < 15:
                halo_right = all_bnd[d + 1, 0]      # next device's left boundary
        _COLL["halo"] += 1
        _COLL["halo_s"] += time.perf_counter() - _t0
        return halo_left, halo_right
    edges_all = torch.empty(
        (world * edges_local.shape[0],) + edges_local.shape[1:],
        dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(edges_all, edges_local, group=_VAE_GROUP)
    edges_all = edges_all.reshape((world,) + edges_local.shape)

    halo_left = edges_all[rank - 1, 1] if rank > 0 else None
    halo_right = edges_all[rank + 1, 0] if rank < world - 1 else None
    _COLL["halo"] += 1
    _COLL["halo_s"] += time.perf_counter() - _t0
    return halo_left, halo_right


def _all_gather_w(x):
    """All-gather rank-ordered width slices back into the full-width tensor."""
    world = _world()
    if world == 1:
        return x
    _t0 = time.perf_counter()
    x = x.contiguous()
    gathered = torch.empty(
        (world * x.shape[0],) + x.shape[1:], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(gathered, x, group=_VAE_GROUP)
    gathered = gathered.reshape((world,) + x.shape)
    ndim = x.dim()
    perm = tuple(range(1, ndim)) + (0, ndim)
    gathered = gathered.permute(*perm).contiguous()
    _COLL["gather"] += 1
    _COLL["gather_s"] += time.perf_counter() - _t0
    return gathered.reshape(x.shape[:-1] + (x.shape[-1] * world,))


@_vae_compile
def _halo_cat_pad_conv3d(x, halo_left, halo_right, weight, bias, pad_tuple,
                         stride, dilation, groups):
    parts = [p for p in (halo_left, x, halo_right) if p is not None]
    x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    x = F.pad(x, pad_tuple)
    return F.conv3d(x, weight, bias, stride, padding=0,
                    dilation=dilation, groups=groups)


@_vae_compile
def _halo_cat_pad_conv2d(x, halo_left, halo_right, weight, bias, pad_tuple,
                         stride, dilation, groups):
    parts = [p for p in (halo_left, x, halo_right) if p is not None]
    x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    x = F.pad(x, pad_tuple)
    return F.conv2d(x, weight, bias, stride, padding=0,
                    dilation=dilation, groups=groups)




def _shard_causal_conv3d_width(conv):
    """Add width-halo exchange to a CausalConv3d's forward, in place.

    No-op for nn.Identity (ResidualBlock.shortcut when in_dim == out_dim) and
    for 1x1-width kernels (the (3,1,1) time_conv in Resample): a size-1 width
    kernel has no cross-rank dependency, so leaving the original forward is
    both correct and cheaper.
    """
    if isinstance(conv, nn.Identity):
        return conv
    kernel_w = conv.weight.shape[-1]
    if kernel_w == 1:
        return conv
    radius = kernel_w // 2
    original_padding = conv._padding  # (padW, padW, padH, padH, padT, 0)

    def width_forward(x, cache_x=None):
        padding = list(original_padding)

        if _world() > 1:
            # Synthesize a zero-valued substitute for a missing temporal
            # cache / width halo instead of branching on None. Prepending
            # `radius` zeros via cat + reducing the F.pad amount by the same
            # radius is mathematically identical to F.pad'ing the full
            # amount directly -- but it keeps the tensor shape fed to the
            # @torch.compile(dynamic=False) math below IDENTICAL across
            # every frame (first frame vs steady-state cache) and every
            # rank (edge rank missing one halo vs a middle rank with both),
            # instead of needing a separate compiled specialization for
            # each shape variant.
            if cache_x is not None and original_padding[4] > 0:
                cache_x = cache_x.to(x.device)
            elif original_padding[4] > 0:
                cache_x = torch.zeros_like(x)
            if cache_x is not None:
                x = torch.cat([cache_x, x], dim=2)
                padding[4] -= cache_x.shape[2]

            halo_left, halo_right = _halo_exchange_w(x, radius)
            if halo_left is None:
                halo_left = torch.zeros_like(x[..., :radius])
            if halo_right is None:
                halo_right = torch.zeros_like(x[..., -radius:])
            padding[0] = 0
            padding[1] = 0
            return _halo_cat_pad_conv3d(
                x, halo_left, halo_right, conv.weight, conv.bias,
                tuple(padding), conv.stride, conv.dilation, conv.groups)

        # world == 1: unchanged, original unsharded behavior (verified
        # bit-identical to the pre-refactor CPU-decode baseline).
        if cache_x is not None and original_padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return F.conv3d(x, conv.weight, conv.bias, stride=conv.stride,
                        dilation=conv.dilation, groups=conv.groups)

    conv.forward = width_forward
    return conv


def _shard_conv2d_width(conv):
    """Add width-halo exchange to a plain nn.Conv2d's forward, in place.

    Used for the 3x3 spatial conv inside Resample -- the only plain Conv2d
    in the decoder outside AttentionBlock's to_qkv/proj (those are handled by
    _shard_attention_block_width instead, via all-gather, and must NOT also
    get this halo treatment).
    """
    kernel_w = conv.kernel_size[1]
    if kernel_w == 1:
        return conv
    radius = kernel_w // 2
    pad_h, pad_w = conv.padding

    def width_forward(x):
        if _world() > 1:
            # Same zero-substitute trick as _shard_causal_conv3d_width: keep
            # the halo shape uniform across edge and middle ranks so there's
            # one compiled specialization per layer, not up to three.
            halo_left, halo_right = _halo_exchange_w(x, radius)
            if halo_left is None:
                halo_left = torch.zeros_like(x[..., :radius])
            if halo_right is None:
                halo_right = torch.zeros_like(x[..., -radius:])
            return _halo_cat_pad_conv2d(
                x, halo_left, halo_right, conv.weight, conv.bias,
                (0, 0, pad_h, pad_h),
                conv.stride, conv.dilation, conv.groups)

        x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
        return F.conv2d(x, conv.weight, conv.bias, stride=conv.stride,
                        dilation=conv.dilation, groups=conv.groups)

    conv.forward = width_forward
    return conv


@_vae_compile
def _attention_block_math(x_full, block):
    identity = x_full
    b, c, t, h, w = x_full.size()
    x_flat = x_full.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x_flat = block.norm(x_flat)

    q, k, v = block.to_qkv(x_flat).reshape(b * t, 1, c * 3,
                                       -1).permute(0, 1, 3,
                                                   2).contiguous().chunk(
                                                       3, dim=-1)
    x_flat = F.scaled_dot_product_attention(q, k, v)
    x_flat = x_flat.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

    x_flat = block.proj(x_flat)
    x_flat = x_flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return x_flat + identity


def _shard_attention_block_width(block):
    """Make an AttentionBlock exact under width sharding: all-gather the
    full width, run the unmodified attention math redundantly on every rank,
    slice back to the local width."""

    def width_forward(x):
        world = _world()
        if world > 1:
            rank = _rank()
            w_local = x.shape[-1]
            w_start, w_end = rank * w_local, (rank + 1) * w_local
            x_full = _all_gather_w(x)
        else:
            x_full = x

        y_full = _attention_block_math(x_full, block)

        if world > 1:
            return y_full[..., w_start:w_end].contiguous()
        return y_full

    block.forward = width_forward
    return block


def _shard_residual_block_width(resnet):
    """Shard every CausalConv3d inside a ResidualBlock (its `residual`
    Sequential mixes norms/activations/dropout with two CausalConv3d, rather
    than exposing conv1/conv2 directly), plus `shortcut` (no-op for
    nn.Identity or a 1x1 kernel)."""
    for layer in resnet.residual:
        if isinstance(layer, CausalConv3d):
            _shard_causal_conv3d_width(layer)
    _shard_causal_conv3d_width(resnet.shortcut)
    return resnet


def shard_vae_decoder_width(decoder):
    """Apply width sharding to a Decoder3d in place (rebinds forward() on the
    relevant submodules; no weights are copied or sliced)."""
    _shard_causal_conv3d_width(decoder.conv1)

    for m in decoder.middle:
        if isinstance(m, ResidualBlock):
            _shard_residual_block_width(m)
        elif isinstance(m, AttentionBlock):
            _shard_attention_block_width(m)

    for m in decoder.upsamples:
        if isinstance(m, ResidualBlock):
            _shard_residual_block_width(m)
        elif isinstance(m, AttentionBlock):
            _shard_attention_block_width(m)
        elif isinstance(m, Resample):
            for sub in m.resample:
                if isinstance(sub, nn.Conv2d):
                    _shard_conv2d_width(sub)
            if hasattr(m, 'time_conv'):
                _shard_causal_conv3d_width(m.time_conv)  # no-op, kernel_w=1

    for m in decoder.head:
        if isinstance(m, CausalConv3d):
            _shard_causal_conv3d_width(m)
    # decoder.head's RMS_norm: width-agnostic, so sharding leaves it alone -- but it
    # is still compiled below with every other pointwise leaf.
    # NOT compiling RMS_norm/SiLU/Upsample here, though rolling_forcing does
    # (models/vae.py:48, 55, 73). Measured on our shapes it is a 5.3x LOSS: with those 66
    # leaves compiled the WARM decode was 117.9 s against 22.4 s without, and two-pass
    # timing put 152.7 s of the 270.6 s first pass down to compilation, so the 5.3x is
    # the regions themselves -- each cheap pointwise op becomes its own NEFF launch and
    # the launch dominates the arithmetic. It presumably pays there because that pipeline
    # streams, calling the decoder once per chunk at chunk-size 3.
    return decoder
