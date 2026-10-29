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


def _selectors(device):
    """Per-rank one-hot weights: (left/right neighbour on this device, what to send in step 2,
    left/right neighbouring device). Zeros where there is no neighbour = the conv's zero padding."""
    P, D = _PD
    d, c = divmod(dist.get_rank(), P)
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


def _halo(x, sel):
    """[..., w] -> [..., w + 2]: this rank's slice with its neighbours' adjoining columns."""
    P, D = _PD
    # One-hot weights in x's dtype: x * 1 + 0 is exact in bf16 too, and the concat below then
    # stays in x's dtype instead of promoting the whole activation to fp32.
    wl1, wr1, sv, wl2, wr2 = (t.to(x.dtype) for t in sel)
    e = torch.stack([x[..., :1], x[..., -1:]], 0).contiguous()            # [2, *x[:-1], 1]
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
    return torch.cat([hl, x, hr], -1)


def _gather_w(x):
    """All-gather the width slices into the full width, in rank (= slice) order."""
    P, D = _PD
    g = fc.all_gather_tensor(x.contiguous(), 0, _MESH["core"])
    if D > 1:
        g = fc.all_gather_tensor(g.contiguous(), 0, _MESH["dev"])
    n = P * D
    b, c, t, h, w = x.shape
    g = g.view((n,) + tuple(x.shape))
    return g.permute(1, 2, 3, 4, 0, 5).reshape(b, c, t, h, n * w)


def _rms(x, gamma, scale):
    # RMS_norm.forward, same op order (bias is the float 0.), always in fp32 (as CUDA autocast
    # runs the norm). A no-op cast in the fp32 decode.
    return F.normalize(x.float(), dim=1) * scale * gamma + 0.


def _cconv(xnew_h, cache_h, w, b):
    """Causal 3x3x3 conv of haloed new frames after the haloed 2-frame cache, in w's dtype."""
    xh = torch.cat([cache_h, xnew_h.to(w.dtype)], 2)
    y = F.conv3d(F.pad(xh, (0, 0, 1, 1, 0, 0)), w, b)
    return y, xh[:, :, -2:]


def _conv_unit(x, cache, w, b, sel):
    return _cconv(_halo(x.to(w.dtype), sel), cache, w, b)


def _res_unit(x, c1, c2, g1, s1, w1, b1, g2, s2, w2, b2, wsc, bsc, sel):
    h = F.conv3d(x, wsc, bsc) if wsc is not None else x
    y = F.silu(_rms(x, g1, s1)).to(w1.dtype)
    y, c1 = _cconv(_halo(y, sel), c1, w1, b1)
    y = F.silu(_rms(y, g2, s2)).to(w2.dtype)
    y, c2 = _cconv(_halo(y, sel), c2, w2, b2)
    return y + h, c1, c2


def _attn_unit(x, gamma, scale, wqkv, bqkv, wproj, bproj):
    """AttentionBlock for this rank's queries against keys/values of the full width."""
    identity = x
    b, c, t, h, w = x.shape

    def qkv(z):
        zf = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, z.shape[-1])
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
    xf = xh.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w + 2)
    xf = xf[:, :, :, None, :, None].expand(b * t, c, h, 2, w + 2, 2).reshape(
        b * t, c, 2 * h, 2 * w + 4)[..., 1:-1]
    y = F.conv2d(F.pad(xf, (0, 0, 1, 1)), w2, b2)
    return y.reshape(b, t, y.shape[1], 2 * h, 2 * w).permute(0, 2, 1, 3, 4)


def _up3d_unit(x, cache_t, wt, bt, w2, b2, sel):
    b, c, t, h, w = x.shape
    xc = torch.cat([cache_t, x.to(wt.dtype)], 2)
    y = F.conv3d(xc, wt, bt).reshape(b, 2, c, t, h, w)
    y = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
    return _up_core(y, w2, b2, sel), xc[:, :, -2:]


def _head_unit(x, cache, g, s, w, b, sel):
    return _cconv(_halo(F.silu(_rms(x, g, s)).to(w.dtype), sel), cache, w, b)


def _compile(fn):
    return torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)


_CONV, _RES, _ATTN = _compile(_conv_unit), _compile(_res_unit), _compile(_attn_unit)
_UP3D, _UP2D, _HEAD = _compile(_up3d_unit), _compile(_up_core), _compile(_head_unit)


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

    def _wb(self, m, keep_fp32=False):
        """(weight, bias) of a conv module in the compute dtype (cached bf16 copies)."""
        if self.dt is None or keep_fp32:
            return m.weight, m.bias
        k = id(m)
        if k not in self._wcache:
            self._wcache[k] = (m.weight.detach().to(self.dt),
                               None if m.bias is None else m.bias.detach().to(self.dt))
        return self._wcache[k]

    def _zeros(self, x, c, halo=True, fp32=False):
        b, _, _, h, w = x.shape
        dt = x.dtype if self.dt is None else (torch.float32 if fp32 else self.dt)
        return torch.zeros((b, c, 2, h, w + (2 if halo else 0)), dtype=dt, device=x.device)

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
                if first:
                    # 'Rep': no time conv on the first frame; the second frame's time conv then
                    # zero-pads, which is what the all-zero cache does.
                    x = _UP2D(x, *self._wb(conv2d), sel)
                else:
                    x, ct = _UP3D(x, ct, *self._wb(m.time_conv), *self._wb(conv2d), sel)
                new.append(ct)
            elif kind == "up2d":
                conv2d = m.resample[1]
                x = _UP2D(x, *self._wb(conv2d), sel)
            else:  # head
                norm, conv = m[0], m[2]
                # fp32 output conv (96 -> 3, cheap) keeps the pixel values at full precision.
                x, c = _HEAD(x, cache(x.shape[1], fp32=True), norm.gamma, norm.scale,
                             *self._wb(conv, keep_fp32=True), sel)
                new.append(c)
        return x, new

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
        k = max(1, int(os.environ.get("LINGBOT_VAE_T", "1")))
        out, caches = self._step(x[:, :, :1], None, first=True)
        outs, i, n = [out], 1, x.shape[2]
        while i < n:
            j = min(n, i + k)
            out, caches = self._step(x[:, :, i:j], caches, first=False)
            outs.append(out)
            i = j
        return torch.cat(outs, 2)
