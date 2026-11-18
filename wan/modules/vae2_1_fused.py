"""Width-sharded Wan2.1 VAE decode with whole blocks compiled and the halo exchange inside the graphs.

LINGBOT_VAE_FUSED=1. Same sharding as vae2_1_sharded.py (every rank holds the whole decoder and
decodes one width slice of the latent), different execution. vae2_1_sharded.py runs each latent
frame as ~33 single-conv NEFFs, ~200 eager ops (norms, SiLU, upsample, cache slicing) and 33
host-issued halo all-gathers; at 64 ranks that is ~2.95 s for ~0.2 s of convolution. Here every
decoder block (ResidualBlock, AttentionBlock, Resample, conv1, head) is one fullgraph NEFF whose
halo exchanges are functional collectives over a DeviceMesh, so they lower inside the graph.

Halo exchange (radius 1, every spatial conv of the decoder): rank r = P*d + c (device d, core c).
  step 1: all-gather the left/right edge columns over the P ranks of one device;
  step 2: all-gather one device-boundary edge per rank over one-rank-per-device groups: core 0
          sends its device's right boundary, core P-1 its device's left boundary.
Each rank then picks its neighbours' edges with one-hot weights, which keeps the graph identical
on every rank (rank-specific data is a graph input), and x*1 + 0 + ... is exact in fp32.

Caches hold the conv inputs of the last two frames WITH their halo columns, so only the new
frames' edges are exchanged. Missing caches are zeros, which is what the causal convs' zero
padding computes. Resample's first-frame 'Rep' case (skip the time conv, then zero-pad it on the
second frame) is a zero cache plus a first-frame graph that skips the time conv.
"""

import os
import time

import torch
import torch._dynamo
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn.functional as F

from .vae2_1 import AttentionBlock, ResidualBlock, Resample

# One compiled function per block kind serves every block of that kind; each distinct shape
# (four resolutions, first frame vs steady state) is a Dynamo cache entry of the same code object.
for _k, _v in (("cache_size_limit", 64), ("recompile_limit", 64),
               ("accumulated_cache_size_limit", 512), ("accumulated_recompile_limit", 512)):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, max(getattr(torch._dynamo.config, _k), _v))

_MESH = None
_PD = None  # (P ranks per device, D devices)


def apply_vae_cc_flags():
    """LINGBOT_VAE_CC_FLAGS ('+' for spaces): append compiler options to NEURON_CC_FLAGS for the
    decoder graphs. torch-neuronx reads NEURON_CC_FLAGS at every compile, so this must run before
    the first decode, and it applies to every later compile in the process: call it only from a
    process that compiles nothing else (the decode step). Measured at 64 ranks, bf16, T=3: -O1 decodes
    in 0.65 s against 0.77 s at the default -O2."""
    extra = os.environ.get("LINGBOT_VAE_CC_FLAGS", "").replace("+", " ").strip()
    if extra:
        os.environ["NEURON_CC_FLAGS"] = (os.environ.get("NEURON_CC_FLAGS", "") + " " + extra).strip()
    return extra


def _init_mesh():
    global _MESH, _PD
    if _MESH is None:
        from torch.distributed.device_mesh import init_device_mesh
        w = dist.get_world_size()
        per = min(int(os.environ.get("LINGBOT_VAE_PER_DEVICE", "4")), w)
        assert w % per == 0 and per >= 2, f"world {w} must be a multiple of ranks/device {per} >= 2"
        _PD = (per, w // per)
        _MESH = init_device_mesh("neuron", (w // per, per), mesh_dim_names=("dev", "core"))
    return _MESH


def tile_2d(rank=None):
    """(row block c, column block d, P, D) of a rank under LINGBOT_VAE_2D."""
    P, D = _PD
    d, c = divmod(dist.get_rank() if rank is None else rank, P)
    return c, d, P, D


def shard_latent_2d(latent):
    """[..., H, W] -> this rank's [H / P, W / D] tile (LINGBOT_VAE_2D)."""
    c, d, P, D = tile_2d()
    H, W = latent.shape[-2:]
    assert H % P == 0 and W % D == 0, f"latent {H}x{W} does not tile into {P}x{D}"
    h, w = H // P, W // D
    return latent[..., c * h:(c + 1) * h, d * w:(d + 1) * w].contiguous()


def gather_video_2d(video_local):
    """Reassemble [C, T, h, w] tiles into the full video on rank 0, one frame at a time (as
    gather_video_width does); returns a CPU tensor on rank 0, None elsewhere."""
    P, D = _PD
    n = P * D
    if os.environ.get("LINGBOT_VAE_GATHER1", "0") == "1":
        # One all-gather of the whole tile and one host copy (the per-frame loop below spent
        # ~5 ms of host time per frame, 0.45 s for 81 frames at 16 ranks). Same bytes.
        f = video_local.contiguous()
        out = torch.empty((n,) + tuple(f.shape), dtype=f.dtype, device=f.device)
        dist.all_gather_into_tensor(out, f)
        if dist.get_rank() != 0:
            return None
        C, T, h, w = f.shape
        return out.view(D, P, C, T, h, w).permute(2, 3, 1, 4, 0, 5).reshape(C, T, P * h, D * w).cpu()
    frames = []
    for i in range(video_local.shape[1]):
        f = video_local.narrow(1, i, 1).contiguous()
        out = torch.empty((n,) + tuple(f.shape), dtype=f.dtype, device=f.device)
        dist.all_gather_into_tensor(out, f)
        if dist.get_rank() == 0:
            C, _, h, w = f.shape
            out = out.view(D, P, C, 1, h, w).permute(2, 3, 1, 4, 0, 5).reshape(C, 1, P * h, D * w)
            frames.append(out.cpu())
    return torch.cat(frames, 1) if dist.get_rank() == 0 else None


def _selectors(device):
    """Per-rank one-hot weights: (left/right neighbour on this device, what to send in step 2,
    left/right neighbouring device). Zeros where there is no neighbour = the conv's zero padding.
    LINGBOT_VAE_2D: (left/right neighbour over D, upper/lower neighbour over P)."""
    P, D = _PD
    d, c = divmod(dist.get_rank(), P)
    if _TWO_D:
        wl, wr, hu, hd = torch.zeros(D), torch.zeros(D), torch.zeros(P), torch.zeros(P)
        if d > 0:
            wl[d - 1] = 1.0
        if d < D - 1:
            wr[d + 1] = 1.0
        if c > 0:
            hu[c - 1] = 1.0
        if c < P - 1:
            hd[c + 1] = 1.0
        return tuple(t.to(device) for t in (wl, wr, hu, hd))
    wl1, wr1 = torch.zeros(P), torch.zeros(P)
    if c > 0:
        wl1[c - 1] = 1.0
    if c < P - 1:
        wr1[c + 1] = 1.0
    sv = torch.tensor([1.0 if c == 0 else 0.0, 1.0 if c == P - 1 else 0.0])
    wl2, wr2 = torch.zeros(D), torch.zeros(D)
    if c == 0 and d > 0:
        wl2[d - 1] = 1.0
    if c == P - 1 and d < D - 1:
        wr2[d + 1] = 1.0
    return tuple(t.to(device) for t in (wl1, wr1, sv, wl2, wr2))


_DIAG_NOHALO = os.environ.get("LINGBOT_VAE_DIAG", "") == "nohalo"   # DIAGNOSTIC ONLY: wrong pixels
# LINGBOT_VAE_SWAPHW=1: run the decoder on [.., W, H] tensors (H innermost) instead of [.., H, W].
# Each rank's width slice is 2..18 columns, and as the innermost axis it made the compiler emit 1.35M
# single-element PE-array transposes per decode (283 ms of matmul-engine time). Every decoder op is
# symmetric in H and W (3x3 kernels transposed, padding swapped), attention is invariant to the token
# order, so the result is the same computation in another layout.
_SWAP = os.environ.get("LINGBOT_VAE_SWAPHW", "0") == "1"
_WAX = -2 if _SWAP else -1          # the sharded (width) axis
# LINGBOT_VAE_2D=1: shard height AND width. Rank P*d + c (device d, core c) decodes latent rows
# block c (of P) and columns block d (of D): at 64 ranks a 15 x 6 tile of the 60 x 96 latent instead
# of a 60 x 2 column slice (of a width padded to 128, so 16 ranks decoded padding). Halos on both
# axes, width first so the height exchange carries the corners. Not combinable with SWAPHW.
_TWO_D = os.environ.get("LINGBOT_VAE_2D", "0") == "1"
assert not (_TWO_D and _SWAP), "LINGBOT_VAE_2D and LINGBOT_VAE_SWAPHW are exclusive"
_HALO_BUF = os.environ.get("LINGBOT_VAE_HALO_BUF", "0") == "1"


def _halo_2d(x, sel):
    """[..., h, w] -> [..., h + 2, w + 2]: the tile with its neighbours' adjoining columns, then
    rows (of the column-extended tile, so the corners come from the diagonal neighbours)."""
    P, D = _PD
    wl, wr, hu, hd = (t.to(x.dtype) for t in sel)
    e = torch.stack([x[..., :1], x[..., -1:]], 0).contiguous()            # [2, ..., h, 1]
    g = fc.all_gather_tensor(e, 0, _MESH["dev"]).view((D,) + tuple(e.shape))
    bd = (D,) + (1,) * (e.dim() - 1)
    left, right = (g[:, 1] * wl.view(bd)).sum(0), (g[:, 0] * wr.view(bd)).sum(0)
    if _HALO_BUF:
        # LINGBOT_VAE_HALO_BUF=1: one [.., h + 2, w + 2] buffer written in place (tile, then its 4 edges)
        # instead of two torch.cat copies of the whole tile. Same values (pure data movement).
        h_, w_ = int(x.shape[-2]), int(x.shape[-1])
        out = x.new_empty(tuple(x.shape[:-2]) + (h_ + 2, w_ + 2))
        out[..., 1:h_ + 1, 1:w_ + 1] = x
        out[..., 1:h_ + 1, :1] = left
        out[..., 1:h_ + 1, w_ + 1:] = right
        e = torch.stack([out[..., 1:2, :], out[..., h_:h_ + 1, :]], 0).contiguous()   # [2, ..., 1, w + 2]
        g = fc.all_gather_tensor(e, 0, _MESH["core"]).view((P,) + tuple(e.shape))
        bp = (P,) + (1,) * (e.dim() - 1)
        out[..., :1, :] = (g[:, 1] * hu.view(bp)).sum(0)
        out[..., h_ + 1:, :] = (g[:, 0] * hd.view(bp)).sum(0)
        return out
    x = torch.cat([left, x, right], -1)
    e = torch.stack([x[..., :1, :], x[..., -1:, :]], 0).contiguous()      # [2, ..., 1, w + 2]
    g = fc.all_gather_tensor(e, 0, _MESH["core"]).view((P,) + tuple(e.shape))
    bp = (P,) + (1,) * (e.dim() - 1)
    up, down = (g[:, 1] * hu.view(bp)).sum(0), (g[:, 0] * hd.view(bp)).sum(0)
    return torch.cat([up, x, down], -2)


def _halo(x, sel):
    """[..., w] -> [..., w + 2]: this rank's slice with its neighbours' adjoining columns."""
    if _TWO_D:
        return _halo_2d(x, sel)
    if _DIAG_NOHALO:
        return F.pad(x, (0, 0, 1, 1) if _SWAP else (1, 1))
    P, D = _PD
    # One-hot weights in x's dtype: x * 1 + 0 is exact in bf16 too, and the concat below then
    # stays in x's dtype instead of promoting the whole activation to fp32.
    wl1, wr1, sv, wl2, wr2 = (t.to(x.dtype) for t in sel)
    nw = x.shape[_WAX]
    e = torch.stack([x.narrow(_WAX, 0, 1), x.narrow(_WAX, nw - 1, 1)], 0).contiguous()
    g1 = fc.all_gather_tensor(e, 0, _MESH["core"]).view((P,) + tuple(e.shape))
    bc = (P,) + (1,) * (e.dim() - 1)
    hl = (g1[:, 1] * wl1.view(bc)).sum(0)
    hr = (g1[:, 0] * wr1.view(bc)).sum(0)
    if D > 1:
        bnd = (g1[P - 1, 1] * sv[0] + g1[0, 0] * sv[1]).contiguous()
        g2 = fc.all_gather_tensor(bnd, 0, _MESH["dev"]).view((D,) + tuple(bnd.shape))
        bd = (D,) + (1,) * bnd.dim()
        hl = hl + (g2 * wl2.view(bd)).sum(0)
        hr = hr + (g2 * wr2.view(bd)).sum(0)
    return torch.cat([hl, x, hr], _WAX)


def _gather_w(x):
    """All-gather the width slices into the full width, in rank (= slice) order. LINGBOT_VAE_2D:
    the tiles into the full [H, W] map."""
    P, D = _PD
    if _TWO_D:
        g = fc.all_gather_tensor(x.contiguous(), 0, _MESH["core"])         # [P*b, ...]
        g = fc.all_gather_tensor(g.contiguous(), 0, _MESH["dev"])         # [D*P*b, ...]
        b, c, t, h, w = x.shape
        g = g.view((D, P) + tuple(x.shape))
        return g.permute(2, 3, 4, 1, 5, 0, 6).reshape(b, c, t, P * h, D * w)
    g = fc.all_gather_tensor(x.contiguous(), 0, _MESH["core"])
    if D > 1:
        g = fc.all_gather_tensor(g.contiguous(), 0, _MESH["dev"])
    n = P * D
    b, c, t, h, w = x.shape
    g = g.view((n,) + tuple(x.shape))
    if _SWAP:                                  # x is [b, c, t, W, H]: rank axis next to W
        return g.permute(1, 2, 3, 0, 4, 5).reshape(b, c, t, n * h, w)
    return g.permute(1, 2, 3, 4, 0, 5).reshape(b, c, t, h, n * w)


_NORM_MODE = os.environ.get("LINGBOT_VAE_NORM", "normalize")


def _rms(x, gamma, scale):
    # RMS_norm.forward, same op order (bias is the float 0.), always in fp32 (as CUDA autocast
    # runs the norm). A no-op cast in the fp32 decode.
    #
    # LINGBOT_VAE_NORM=mm: the channel sum of squares as one fp32 matmul against a ones vector
    # (channels are the contraction axis), then x * rsqrt(max(sumsq, eps^2)), which is
    # F.normalize's x / max(||x||, eps) up to fp32 rounding. In the default form the profile put
    # ~174 ms of matmul-engine time and 64 ms of GpSimd cross-lane reduce on these reductions.
    if _NORM_MODE == "mmb":
        # LINGBOT_VAE_NORM=mmb (kiro_mx4): the channel sum of squares as a bf16 matmul (fp32 accumulation, bf16 result:
        # its 2^-9 relative error moves rsqrt by 2^-10, below the bf16 rounding of the normalized output the next conv
        # reads), the scaling in fp32. The 2-chip device profile had the fp32 form at 960 fp32 matmuls (two bf16
        # passes each) and the rsqrt on GpSimd (385 us) per stage-3 ResidualBlock call.
        xb = x.to(torch.bfloat16)
        ones = torch.ones(x.shape[1], dtype=torch.bfloat16, device=x.device)
        ss = torch.einsum("bcthw,c->bthw", xb * xb, ones) if x.dim() == 5 else \
            torch.einsum("bchw,c->bhw", xb * xb, ones)
        inv = torch.rsqrt(torch.clamp(ss.float(), min=1e-24)).unsqueeze(1)
        return x.float() * inv * scale * gamma + 0.
    x = x.float()
    if _NORM_MODE == "mm":
        ones = torch.ones(x.shape[1], dtype=torch.float32, device=x.device)
        ss = torch.einsum("bcthw,c->bthw", x * x, ones) if x.dim() == 5 else \
            torch.einsum("bchw,c->bhw", x * x, ones)
        inv = torch.rsqrt(torch.clamp(ss, min=1e-24)).unsqueeze(1)
        return x * inv * scale * gamma + 0.
    return F.normalize(x, dim=1) * scale * gamma + 0.


# LINGBOT_VAE_NKI_NORM=1 (with LINGBOT_VAE_BF16): every F.silu(RMS_norm(x)).to(bf16) of the decoder (the two in each
# ResidualBlock and the head's, with LINGBOT_VAE_HEAD_BF16) as one fused NKI kernel, kernels/vae_rms_silu.py: the
# channel sum of squares as an fp32 matmul against ones [C, C] (the sum lands on every channel's partition), rsqrt on
# the Scalar engine, x * rsqrt and silu(. * scale * gamma) in the same pass. Off: the compiled torch form above.
# REFUTED in-graph (kiro/sa-vae, 2-chip LNC1 16 ranks): 2.2x faster than the torch form alone (stage 3, 4 frames:
# 276-331 vs 608 us) and -1.15 ms per stage-3 ResidualBlock in perf/vae_rb_bench.py, but the decode stayed at 813 ms
# (T=6) and with LINGBOT_VAE_NKI_HALO it costs +18 ms (662 vs 644, T=1); LINGBOT_VAE_NKI_HALO does the stage-3 part.
_NKI_NORM = os.environ.get("LINGBOT_VAE_NKI_NORM", "0") == "1"
# LINGBOT_VAE_UNIT_TIMING=1 (DIAGNOSTIC: a device sync around every block call, so slower overall): rank 0 prints the
# synced wall time per (block kind, input shape) after every decode().
_UNIT_TIMING = os.environ.get("LINGBOT_VAE_UNIT_TIMING", "0") == "1"
if _NKI_NORM:
    from kernels.vae_rms_silu import rms_silu as _nki_rms_silu


def _norm_silu(x, gamma, scale, dt):
    """F.silu(_rms(x, gamma, scale)).to(dt)."""
    if _NKI_NORM and dt == torch.bfloat16 and x.shape[0] == 1:
        c = x.shape[1]
        y = _nki_rms_silu(x.to(torch.bfloat16).reshape(c, -1), gamma.float().reshape(c, 1) * scale)
        return y.view(x.shape)
    return F.silu(_rms(x, gamma, scale)).to(dt)


# LINGBOT_VAE_NKI_HALO=1 (with LINGBOT_VAE_2D and bf16 convs; by default C <= 128: the stage-3 ResidualBlocks and
# the head with LINGBOT_VAE_HEAD_BF16): norm + SiLU + both halos + the cache concatenate of every causal conv input
# as one NKI kernel (kernels/vae_norm_halo.py). The raw (pre-norm) edges are exchanged instead of the normalized
# ones: the norm is per position, so the kernel normalizing the neighbours' raw edges computes what their own norm
# does. Removes the three full-tile torch.cat copies (column halo, row halo, cache frames) and the separate norm
# passes. perf/vae_rb_bench.py, stage-3 ResidualBlock (4 frames, one core): 5754 -> 3499 us, conv input and cache
# bit-identical to LINGBOT_VAE_NKI_NORM (SQ=fp32) + halo + cat. Decode (2-chip LNC1 16 ranks, T=1, + HEAD_BF16 and
# -O1 unet-inference): 715 -> 644 ms, vs_R 52.87 dB unchanged.
# LINGBOT_VAE_NKI_HALO_CMAX=c: use it for blocks of up to c channels (default 128 = stage 3 + head). Bit-identical
# but slower above 128 (two partition tiles): stage-2 ResidualBlock 2841 -> 3152 us, C=384 conv unit 489 -> 598 us.
_NKI_HALO = os.environ.get("LINGBOT_VAE_NKI_HALO", "0") == "1"
_NKI_HALO_CMAX = int(os.environ.get("LINGBOT_VAE_NKI_HALO_CMAX", "128"))
if _NKI_HALO:
    from kernels.vae_norm_halo import norm_halo as _nki_norm_halo
# LINGBOT_VAE_NKI_HEAD=1 (with LINGBOT_VAE_NKI_HALO and LINGBOT_VAE_HEAD_BF16): the output conv (96 -> 3, causal 3x3x3)
# as kernels/vae_head_conv.py on the halo kernel's buffer: each tap one matmul with the [96, 3] tap weight stationary
# and an offset slice of the SBUF band as the moving operand (flat-shift over the padded width), 27 taps accumulating
# in PSUM. One core, stage-3 shape: 1098 us vs 1636 for the compiler's conv, bit-identical (perf/vae_head_bench.py
# VARIANT=nki; for a 96 -> 96 conv it is 1198 vs 1133 us, so only the head uses it).
_NKI_HEAD = os.environ.get("LINGBOT_VAE_NKI_HEAD", "0") == "1"
if _NKI_HEAD:
    from kernels.vae_head_conv import head_conv as _nki_head_conv, head_conv_weights as _nki_head_conv_w
# LINGBOT_VAE_NKI_CONV=head|all (with LINGBOT_VAE_NKI_HALO; takes precedence over LINGBOT_VAE_NKI_HEAD): the head's
# conv (head) or every conv LINGBOT_VAE_NKI_HALO feeds (all: + the stage-3 ResidualBlocks) as one NKI kernel with its
# norm, halo and cache, kernels/vae_norm_halo_conv.py: the tile in bands of rows, each band normalized in SBUF and
# convolved from SBUF, so the conv input never goes through HBM and the compiler's conv is replaced by flat-shift
# matmuls (kernels/vae_head_conv.py). Kernel options (LINGBOT_VAE_NKI_CONV_SQ / _EDGE / _LA / _R) in the kernel's
# docstring. One core, stage-3 shape (perf/vae_head_bench.py VARIANT=fused, vs the LINGBOT_VAE_NKI_HALO path):
#   head (96 -> 3):            2941 -> 1683 us (bit-identical conv and cache), + SQ=bf16 1614
#   96 -> 96 conv unit:        1828 -> 1705 us (bit-identical), + SQ=bf16 1580
#   stage-3 ResidualBlock:     3497 -> 3136 us (perf/vae_rb_bench.py), + SQ=bf16 2956, + RES 2931
# (EDGE=copy: another -50..-100 us per conv in the benches, but it broke the decode's accuracy: see the kernel.)
# SQ=bf16 (bf16 squares + one-pass ones-matmul) moves ~2% of the normalized values by one bf16 ulp; the decode's
# vs_R stays 52.87 dB. Decode (2-chip LNC1 16 ranks, T=1, HEAD_BF16, -O1 unet-inference, NKI_HALO): 647 / 645 ms
# (start / end) -> NKI_HEAD=1 624, NKI_CONV=head 616, =all 602-607, =all + SQ=bf16 + LA=1 587 (52.87 dB).
# LINGBOT_VAE_NKI_CONV_CMAX=c (with =all): also the ResidualBlock convs of up to c in / out channels (default 128;
# 192 adds the stage-2 ResidualBlocks, whose norm is otherwise the compiled torch form). Stage-2 ResidualBlock
# (perf/vae_rb_bench.py): torch path 2823-2845 us, fused 2879 (fp32 squares), 2658 with SQ=bf16. Decode with
# =all SQ=bf16 LA=1: 587-588 -> 582-583 ms, vs_R 52.8743 -> 52.8755 dB (best flag set).
_NKI_CONV = os.environ.get("LINGBOT_VAE_NKI_CONV", "0")
assert _NKI_CONV in ("0", "head", "all"), _NKI_CONV
_NKI_CONV_CMAX = int(os.environ.get("LINGBOT_VAE_NKI_CONV_CMAX", "128"))
# LINGBOT_VAE_NKI_CONV_RES=1 (with =all): the ResidualBlock's shortcut add (y + h) in the second conv's kernel, on
# the PSUM evacuation (bf16 conv output + bf16 h, rounded once more, as torch's bf16 add). Bit-identical
# (perf/vae_head_bench.py VARIANT=rbres), stage-3 ResidualBlock -27..-34 us; decode neutral (587-589 vs 587 ms with
# NKI_CONV=all SQ=bf16 LA=1, 52.87 dB), so not in the best flag set.
_NKI_CONV_RES = os.environ.get("LINGBOT_VAE_NKI_CONV_RES", "0") == "1"
if _NKI_CONV != "0":
    from kernels.vae_norm_halo_conv import norm_halo_conv as _nki_norm_halo_conv
    from kernels.vae_head_conv import head_conv_weights as _nki_conv_w


def _raw_edges_2d(x, sel):
    """_halo_2d's two exchanges on x [1, C, T, H, W] without assembling the haloed tile: (left, right [.., H, 1],
    up, down [.., 1, W + 2] with the corners), zeros where there is no neighbour."""
    P, D = _PD
    wl, wr, hu, hd = (t.to(x.dtype) for t in sel)
    e = torch.stack([x[..., :1], x[..., -1:]], 0).contiguous()            # [2, ..., h, 1]
    g = fc.all_gather_tensor(e, 0, _MESH["dev"]).view((D,) + tuple(e.shape))
    bd = (D,) + (1,) * (e.dim() - 1)
    left, right = (g[:, 1] * wl.view(bd)).sum(0), (g[:, 0] * wr.view(bd)).sum(0)
    top = torch.cat([left[..., :1, :], x[..., :1, :], right[..., :1, :]], -1)
    bot = torch.cat([left[..., -1:, :], x[..., -1:, :], right[..., -1:, :]], -1)
    e = torch.stack([top, bot], 0).contiguous()                            # [2, ..., 1, w + 2]
    g = fc.all_gather_tensor(e, 0, _MESH["core"]).view((P,) + tuple(e.shape))
    bp = (P,) + (1,) * (e.dim() - 1)
    up, down = (g[:, 1] * hu.view(bp)).sum(0), (g[:, 0] * hd.view(bp)).sum(0)
    return left, right, up, down


def _use_nh(x, w):
    if _NKI_CONV == "all" and _NKI_HALO and _TWO_D and w.dtype == torch.bfloat16 and x.shape[0] == 1 and \
            max(x.shape[1], w.shape[0]) <= _NKI_CONV_CMAX:
        return True
    return _NKI_HALO and _TWO_D and w.dtype == torch.bfloat16 and x.shape[0] == 1 and x.shape[1] <= _NKI_HALO_CMAX


def _nh_fused(x, w, b, head=False):
    """Does _nh_conv take kernels/vae_norm_halo_conv.py (LINGBOT_VAE_NKI_CONV)?"""
    return (_NKI_CONV == "all" or (_NKI_CONV == "head" and head)) and b is not None and \
        max(x.shape[1], w.shape[0]) <= max(_NKI_CONV_CMAX, 128)


def _nh_conv(x, cache, gamma, scale, w, b, sel, nki_conv=False, head=False, res=None):
    """_cconv(_halo(_norm_silu(x, gamma, scale, bf16), sel), cache, w, b) via kernels/vae_norm_halo.py; nki_conv: the
    conv as kernels/vae_head_conv.py (LINGBOT_VAE_NKI_HEAD)."""
    xb = x.to(torch.bfloat16)
    l, r, u, d = _raw_edges_2d(xb, sel)
    c = x.shape[1]
    if _nh_fused(x, w, b, head):
        y, nc = _nki_norm_halo_conv(xb[0], l[0], r[0], u[0], d[0], cache[0].to(torch.bfloat16),
                                    gamma.float().reshape(c, 1) * scale, _nki_conv_w(w), b.float().reshape(-1, 1),
                                    None if res is None else res[0])
        return y.unsqueeze(0), nc.unsqueeze(0)
    buf = _nki_norm_halo(xb[0], l[0], r[0], u[0], d[0], cache[0].to(torch.bfloat16),
                         gamma.float().reshape(c, 1) * scale)
    if nki_conv:
        y = _nki_head_conv(buf, _nki_head_conv_w(w), b.float().reshape(-1, 1))
        return y.unsqueeze(0), buf[:, -2:].unsqueeze(0)
    buf = buf.unsqueeze(0)
    return F.conv3d(buf, w, b), buf[:, :, -2:]


def _cconv(xnew_h, cache_h, w, b):
    """Causal 3x3x3 conv of haloed new frames after the haloed 2-frame cache, in w's dtype."""
    xh = torch.cat([cache_h, xnew_h.to(w.dtype)], 2)
    if _TWO_D:                                   # both spatial halos are in xh already
        return F.conv3d(xh, w, b), xh[:, :, -2:]
    y = F.conv3d(F.pad(xh, (1, 1, 0, 0, 0, 0) if _SWAP else (0, 0, 1, 1, 0, 0)), w, b)
    return y, xh[:, :, -2:]


def _conv_unit(x, cache, w, b, sel):
    return _cconv(_halo(x.to(w.dtype), sel), cache, w, b)


def _res_unit(x, c1, c2, g1, s1, w1, b1, g2, s2, w2, b2, wsc, bsc, sel):
    h = F.conv3d(x, wsc, bsc) if wsc is not None else x
    if _use_nh(x, w1):
        y, c1 = _nh_conv(x, c1, g1, s1, w1, b1, sel)
    else:
        y = _norm_silu(x, g1, s1, w1.dtype)
        y, c1 = _cconv(_halo(y, sel), c1, w1, b1)
    if _use_nh(y, w2):
        if _NKI_CONV_RES and _nh_fused(y, w2, b2) and h.dtype == torch.bfloat16 and h.shape[1] == w2.shape[0]:
            y, c2 = _nh_conv(y, c2, g2, s2, w2, b2, sel, res=h)
            return y, c1, c2
        y, c2 = _nh_conv(y, c2, g2, s2, w2, b2, sel)
    else:
        y = _norm_silu(y, g2, s2, w2.dtype)
        y, c2 = _cconv(_halo(y, sel), c2, w2, b2)
    return y + h, c1, c2


def _attn_unit(x, gamma, scale, wqkv, bqkv, wproj, bproj):
    """AttentionBlock for this rank's queries against keys/values of the full width."""
    identity = x
    b, c, t, h, w = x.shape

    def qkv(z):
        zf = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, z.shape[-2], z.shape[-1])
        zf = _rms(zf, gamma, scale).to(wqkv.dtype)
        return F.conv2d(zf, wqkv, bqkv).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

    q = qkv(x)[0]
    _, k, v = qkv(_gather_w(x))
    if wqkv.dtype == torch.float32:
        o = F.scaled_dot_product_attention(q, k, v)
    else:
        # fp32 scores and softmax, as AttentionBlock does on Neuron.
        p = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * q.size(-1) ** -0.5, dim=-1)
        o = (p @ v.float()).to(wproj.dtype)
    o = o.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
    o = F.conv2d(o, wproj, bproj)
    return o.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def _up_core(x, w2, b2, sel):
    """Nearest-exact x2 upsample + 3x3 Conv2d. The halo is exchanged BEFORE upsampling (half the
    rows; nearest x2 duplicates columns, so the neighbour's last column is the same either way)."""
    b, c, t, h, w = x.shape
    xh = _halo(x.to(w2.dtype), sel)
    if _TWO_D:                                 # [.., h + 2, w + 2] -> x2 -> crop -> 3x3 valid
        xf = xh.permute(0, 2, 1, 3, 4).reshape(b * t, c, h + 2, w + 2)
        xf = xf[:, :, :, None, :, None].expand(b * t, c, h + 2, 2, w + 2, 2).reshape(
            b * t, c, 2 * h + 4, 2 * w + 4)[:, :, 1:-1, 1:-1]
        y = F.conv2d(xf, w2, b2)
        return y.reshape(b, t, y.shape[1], 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    if _SWAP:                                  # [.., W, H]: halo rows on W, pad H
        xf = xh.permute(0, 2, 1, 3, 4).reshape(b * t, c, h + 2, w)
        xf = xf[:, :, :, None, :, None].expand(b * t, c, h + 2, 2, w, 2).reshape(
            b * t, c, 2 * h + 4, 2 * w)[:, :, 1:-1]
        y = F.conv2d(F.pad(xf, (1, 1, 0, 0)), w2, b2)
        return y.reshape(b, t, y.shape[1], 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
    xf = xh.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w + 2)
    xf = xf[:, :, :, None, :, None].expand(b * t, c, h, 2, w + 2, 2).reshape(
        b * t, c, 2 * h, 2 * w + 4)[..., 1:-1]
    y = F.conv2d(F.pad(xf, (0, 0, 1, 1)), w2, b2)
    return y.reshape(b, t, y.shape[1], 2 * h, 2 * w).permute(0, 2, 1, 3, 4)


# LINGBOT_VAE_SUBPIX=1 (with LINGBOT_VAE_2D): nearest x2 upsample + 3x3 conv as four 2x2 convs on the low-resolution
# haloed input, one per output phase (py, px), interleaved: output row 2i + py reads input rows i + py + {0, 1} with
# the vertical taps folded (py = 0: w0, w1 + w2; py = 1: w0 + w1, w2), the same for columns. Exact in real arithmetic
# (the folded taps are summed in fp32, then cast), 2.25x fewer FLOPs, and the 4x upsampled input is never built.
_SUBPIX = os.environ.get("LINGBOT_VAE_SUBPIX", "0") == "1"
# LINGBOT_VAE_HEAD_BF16=1 (with LINGBOT_VAE_BF16): the output conv (96 -> 3, causal 3x3x3) in bf16 like every other
# conv (fp32 accumulation) instead of fp32. In the 2-chip LNC1 device profile that fp32 conv was the decode's single
# most expensive call: 13440 matmuls with 3 stationary columns at the fp32 rate, 6.58 ms per latent frame (16% of 812).
_HEAD_BF16 = os.environ.get("LINGBOT_VAE_HEAD_BF16", "0") == "1"
def subpix_weights(w2):
    """[C', C, 3, 3] -> [4, C', C, 2, 2]: the folded 2x2 kernel of each output phase (py, px), row-major."""
    wf = w2.float()
    def fold(k, ax, ph):              # 3 taps -> 2 along axis ax for phase ph
        t0, t1, t2 = (k.narrow(ax, j, 1) for j in range(3))
        return torch.cat([t0, t1 + t2] if ph == 0 else [t0 + t1, t2], ax)
    return torch.stack([fold(fold(wf, 2, py), 3, px) for py in (0, 1) for px in (0, 1)]).to(w2.dtype)
def _up_core_sp(x, weff, b2, sel):
    """_up_core (LINGBOT_VAE_2D) in the sub-pixel form: weff from subpix_weights."""
    b, c, t, h, w = x.shape
    xh = _halo(x.to(weff.dtype), sel)                                     # [b, c, t, h + 2, w + 2]
    xf = xh.permute(0, 2, 1, 3, 4).reshape(b * t, c, h + 2, w + 2)
    ys = [F.conv2d(xf[:, :, py:py + h + 1, px:px + w + 1], weff[2 * py + px], b2) for py in (0, 1) for px in (0, 1)]
    co = ys[0].shape[1]
    y = torch.stack(ys, 0).view(2, 2, b * t, co, h, w).permute(2, 3, 4, 0, 5, 1).reshape(b * t, co, 2 * h, 2 * w)
    return y.reshape(b, t, co, 2 * h, 2 * w).permute(0, 2, 1, 3, 4)
def _up3d_unit(x, cache_t, wt, bt, w2, b2, sel):
    b, c, t, h, w = x.shape
    xc = torch.cat([cache_t, x.to(wt.dtype)], 2)
    y = F.conv3d(xc, wt, bt).reshape(b, 2, c, t, h, w)
    y = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
    if _SUBPIX and w2.dim() == 5:      # w2 is the folded [4, C', C, 2, 2] (FusedWidthDecoder._wb_up)
        return _up_core_sp(y, w2, b2, sel), xc[:, :, -2:]
    return _up_core(y, w2, b2, sel), xc[:, :, -2:]


# LINGBOT_VAE_NKI_UP=2d|all (with LINGBOT_VAE_2D, LINGBOT_VAE_BF16): the spatial upsample (nearest x2 + 3x3 Conv2d) of
# the upsample2d Resample (2d) or of every Resample (all: + the upsample3d ones' conv after their time conv) as one NKI
# kernel on the low-resolution haloed tile (kernels/vae_up_conv.py): each output phase is a 3x3 flat-shift conv over
# the low-res band in SBUF, written to the output with stride 2, so the 4x upsampled input is never built in HBM.
# Halo: the raw edges of _raw_edges_2d (the same two exchanges as _halo). LINGBOT_VAE_NKI_UP_FOLD=1: the folded
# 2x2 phase kernels (2.25x fewer matmuls; weights folded in fp32 from the fp32 module weights, rounded once).
# LINGBOT_VAE_NKI_UP_R: low-res rows per band (default 10).
_NKI_UP = os.environ.get("LINGBOT_VAE_NKI_UP", "0")
assert _NKI_UP in ("0", "2d", "all"), _NKI_UP
_NKI_UP_FOLD = os.environ.get("LINGBOT_VAE_NKI_UP_FOLD", "0") == "1"
_NKI_UP_R = int(os.environ.get("LINGBOT_VAE_NKI_UP_R", "10"))
if _NKI_UP != "0":
    from kernels.vae_up_conv import up_conv as _nki_up_conv, up_conv_weights as _nki_up_w


def _up_core_nki(x, wk, b2, sel):
    """_up_core (LINGBOT_VAE_2D) via kernels/vae_up_conv.py; wk = up_conv_weights(w2[, fold])."""
    xb = x.to(torch.bfloat16)
    l, r, u, d = _raw_edges_2d(xb, sel)
    y = _nki_up_conv(xb[0], l[0], r[0], u[0], d[0], wk, b2.float().reshape(-1, 1), rows=_NKI_UP_R,
                     fold=_NKI_UP_FOLD)
    return y.unsqueeze(0)


def _up3d_unit_nki(x, cache_t, wt, bt, wk, b2, sel):
    """_up3d_unit with the spatial part as _up_core_nki."""
    b, c, t, h, w = x.shape
    xc = torch.cat([cache_t, x.to(wt.dtype)], 2)
    y = F.conv3d(xc, wt, bt).reshape(b, 2, c, t, h, w)
    y = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
    return _up_core_nki(y, wk, b2, sel), xc[:, :, -2:]


def _head_unit(x, cache, g, s, w, b, sel):
    if _use_nh(x, w):
        return _nh_conv(x, cache, g, s, w, b, sel, nki_conv=_NKI_HEAD and b is not None and w.shape[0] <= 128,
                        head=True)
    return _cconv(_halo(_norm_silu(x, g, s, w.dtype), sel), cache, w, b)


def _compile(fn):
    return torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)


_CONV, _RES, _ATTN = _compile(_conv_unit), _compile(_res_unit), _compile(_attn_unit)
_UP3D, _UP2D, _HEAD = _compile(_up3d_unit), _compile(_up_core), _compile(_head_unit)
_UP2D_SP = _compile(_up_core_sp)
_UP2D_N, _UP3D_N = _compile(_up_core_nki), _compile(_up3d_unit_nki)


class FusedWidthDecoder:
    """decode(z_local, scale) == WanVAE_.decode on the width-sharded latent, block-compiled."""

    def __init__(self, model):
        self.model = model
        # LINGBOT_VAE_BF16=1: bf16 convolutions (fp32 accumulation), fp32 norms and attention
        # softmax, fp32 output conv -- what the module's torch.cuda.amp.autocast(bf16) gives on
        # CUDA. On Neuron that autocast is a no-op, so the default decode is fp32 throughout.
        self.dt = torch.bfloat16 if os.environ.get("LINGBOT_VAE_BF16", "0") == "1" else None
        self._wcache = {}
        _init_mesh()
        dec = model.decoder
        self.units = [("conv", dec.conv1)]
        for m in list(dec.middle) + list(dec.upsamples):
            if isinstance(m, ResidualBlock):
                self.units.append(("res", m))
            elif isinstance(m, AttentionBlock):
                self.units.append(("attn", m))
            elif isinstance(m, Resample):
                assert m.mode in ("upsample3d", "upsample2d"), m.mode
                self.units.append(("up3d" if m.mode == "upsample3d" else "up2d", m))
            else:
                raise TypeError(type(m))
        self.units.append(("head", dec.head))
        self.sel = None
        self._ut = {}

    def _wb(self, m, keep_fp32=False):
        """(weight, bias) of a conv module in the compute dtype (cached bf16 copies); with
        LINGBOT_VAE_SWAPHW the spatial kernel axes are transposed to match the [W, H] layout."""
        if (self.dt is None or keep_fp32) and not _SWAP:
            return m.weight, m.bias
        k = (id(m), keep_fp32)
        if k not in self._wcache:
            dt = m.weight.dtype if (self.dt is None or keep_fp32) else self.dt
            w = m.weight.detach().to(dt)
            if _SWAP:
                w = w.transpose(-1, -2).contiguous()
            self._wcache[k] = (w, None if m.bias is None else m.bias.detach().to(dt))
        return self._wcache[k]

    def _wb_up(self, m):
        """The upsample's 3x3 conv: (weight, bias), or with LINGBOT_VAE_SUBPIX the folded [4, C', C, 2, 2] weight."""
        w, b = self._wb(m)
        if not (_SUBPIX and _TWO_D and not _SWAP):
            return w, b
        k = (id(m), "subpix")
        if k not in self._wcache:
            self._wcache[k] = (subpix_weights(w), b)
        return self._wcache[k]
    def _wb_upn(self, m):
        """LINGBOT_VAE_NKI_UP: the upsample conv's (kernels/vae_up_conv.py weight, bias); the fold from the fp32 module
        weight (on the host, once)."""
        k = (id(m), "nkiup", _NKI_UP_FOLD)
        if k not in self._wcache:
            w, b = self._wb(m)
            src = m.weight.detach() if _NKI_UP_FOLD else w
            wk = _nki_up_w(src.cpu(), fold=_NKI_UP_FOLD).to(torch.bfloat16).to(w.device)
            self._wcache[k] = (wk, b)
        return self._wcache[k]

    def _nki_up(self, kind):
        return _TWO_D and not _SWAP and self.dt == torch.bfloat16 and \
            (_NKI_UP == "all" or (_NKI_UP == "2d" and kind == "up2d"))

    def _zeros(self, x, c, halo=True, fp32=False):
        shape = [x.shape[0], c, 2, x.shape[3], x.shape[4]]
        if halo:
            shape[_WAX] += 2
            if _TWO_D:
                shape[-2] += 2
        dt = x.dtype if self.dt is None else (torch.float32 if fp32 else self.dt)
        return torch.zeros(shape, dtype=dt, device=x.device)

    def _step(self, x, caches, first):
        new, i = [], 0

        def cache(c, halo=True, fp32=False):
            nonlocal i
            if caches is None:
                return self._zeros(x, c, halo, fp32)
            i += 1
            return caches[i - 1]

        sel = self.sel
        for kind, m in self.units:
            if _UNIT_TIMING:
                torch.neuron.synchronize()
                _t0 = time.perf_counter()
                _key = (kind, tuple(x.shape[1:]))
            if kind == "conv":
                x, c = _CONV(x, cache(x.shape[1]), *self._wb(m), sel)
                new.append(c)
            elif kind == "res":
                r = m.residual
                c1 = cache(x.shape[1])
                c2 = cache(r[2].weight.shape[0])
                sc = m.shortcut
                wsc, bsc = (None, None) if isinstance(sc, torch.nn.Identity) else self._wb(sc)
                x, c1, c2 = _RES(x, c1, c2, r[0].gamma, r[0].scale, *self._wb(r[2]),
                                 r[3].gamma, r[3].scale, *self._wb(r[6]), wsc, bsc, sel)
                new += [c1, c2]
            elif kind == "attn":
                x = _ATTN(x, m.norm.gamma, m.norm.scale, *self._wb(m.to_qkv), *self._wb(m.proj))
            elif kind == "up3d":
                ct = cache(x.shape[1], halo=False)
                conv2d = m.resample[1]
                _up2 = _UP2D_SP if (_SUBPIX and _TWO_D and not _SWAP) else _UP2D
                if self._nki_up(kind):
                    if first:
                        x = _UP2D_N(x, *self._wb_upn(conv2d), sel)
                    else:
                        x, ct = _UP3D_N(x, ct, *self._wb(m.time_conv), *self._wb_upn(conv2d), sel)
                elif first:
                    # 'Rep': no time conv on the first frame; the second frame's time conv then
                    # zero-pads, which is what the all-zero cache does.
                    x = _up2(x, *self._wb_up(conv2d), sel)
                else:
                    x, ct = _UP3D(x, ct, *self._wb(m.time_conv), *self._wb_up(conv2d), sel)
                new.append(ct)
            elif kind == "up2d":
                conv2d = m.resample[1]
                _up2 = _UP2D_SP if (_SUBPIX and _TWO_D and not _SWAP) else _UP2D
                if self._nki_up(kind):
                    x = _UP2D_N(x, *self._wb_upn(conv2d), sel)
                else:
                    x = _up2(x, *self._wb_up(conv2d), sel)
            else:  # head
                norm, conv = m[0], m[2]
                # fp32 output conv (96 -> 3, cheap) keeps the pixel values at full precision.
                x, c = _HEAD(x, cache(x.shape[1], fp32=not _HEAD_BF16), norm.gamma, norm.scale,
                             *self._wb(conv, keep_fp32=not _HEAD_BF16), sel)
                new.append(c)
            if _UNIT_TIMING:
                torch.neuron.synchronize()
                s = self._ut.setdefault(_key, [0, 0.0])
                s[0] += 1
                s[1] += time.perf_counter() - _t0
        return x, new

    def unit_timing_report(self):
        """LINGBOT_VAE_UNIT_TIMING=1: per (unit kind, input shape) calls and synced ms since the last report."""
        if not _UNIT_TIMING:
            return
        tot = sum(v[1] for v in self._ut.values())
        lines = [f"VAEUNIT {k[0]:5s} {str(k[1]):24s} n={v[0]:3d} ms={v[1] * 1e3:8.1f} per={v[1] * 1e3 / v[0]:6.2f}"
                 for k, v in sorted(self._ut.items(), key=lambda kv: -kv[1][1])]
        if dist.get_rank() == 0:
            print("\n".join(lines) + f"\nVAEUNIT total ms={tot * 1e3:.1f}", flush=True)
        self._ut = {}

    @torch.no_grad()
    def decode(self, z, scale):
        """z: [1, z_dim, T, H, w_local] -> [1, 3, 1 + 4 (T - 1), 8 H, 8 w_local] (unclamped).

        LINGBOT_VAE_T=k: after frame 0 (which must run alone: it skips the temporal upsample),
        decode k latent frames per step. Every temporal op is a causal conv over [2-frame cache,
        new frames] or per-frame, so k frames in one step are the same computation as k steps of
        one frame; k only changes how many graph launches and halo exchanges the decode takes."""
        if self.sel is None:
            self.sel = _selectors(z.device)
        zd = self.model.z_dim
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, zd, 1, 1, 1) + scale[0].view(1, zd, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.model.conv2(z)
        if _SWAP:
            x = x.transpose(-1, -2).contiguous()
        k = max(1, int(os.environ.get("LINGBOT_VAE_T", "1")))
        out, caches = self._step(x[:, :, :1], None, first=True)
        outs, i, n = [out], 1, x.shape[2]
        while i < n:
            j = min(n, i + k)
            out, caches = self._step(x[:, :, i:j], caches, first=False)
            outs.append(out)
            i = j
        self.unit_timing_report()
        out = torch.cat(outs, 2)
        return out.transpose(-1, -2).contiguous() if _SWAP else out

    @torch.no_grad()
    def decode_chunk(self, z, scale, first, on_step=None):
        """decode() split over consecutive latent chunks: the causal caches carry over between
        calls, first=True on the first chunk. The chunks of one video decode to the same frames
        as one decode() of the whole latent. on_step(frames) gets each step's frames as soon as
        that step is issued (LINGBOT_VAE_T latent frames per step); the call then returns None."""
        if self.sel is None:
            self.sel = _selectors(z.device)
        zd = self.model.z_dim
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, zd, 1, 1, 1) + scale[0].view(1, zd, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.model.conv2(z)
        if _SWAP:
            x = x.transpose(-1, -2).contiguous()
        k = max(1, int(os.environ.get("LINGBOT_VAE_T", "1")))
        outs, i, n = [], 0, x.shape[2]

        def put(out):
            out = out.transpose(-1, -2).contiguous() if _SWAP else out
            if on_step is not None:
                on_step(out)
            else:
                outs.append(out)
        if first:
            out, self._caches = self._step(x[:, :, :1], None, first=True)
            put(out)
            i = 1
        while i < n:
            j = min(n, i + k)
            out, self._caches = self._step(x[:, :, i:j], self._caches, first=False)
            put(out)
            i = j
        return None if on_step is not None else torch.cat(outs, 2)
