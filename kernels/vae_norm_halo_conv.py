"""Norm + SiLU + halo + cache + causal 3x3x3 conv of the VAE decoder in one NKI kernel (LINGBOT_VAE_NKI_CONV,
wan/modules/vae2_1_fused.py _nh_conv: the head, the stage-3 ResidualBlocks and with LINGBOT_VAE_NKI_CONV_CMAX=192
the stage-2 ones).

  norm_halo_conv(x, left, right, up, down, cache, g, wt, bias) -> (y [CO, T, H, W] bf16, new cache [C, 2, Hp, Wp])
      x, left, right, up, down, cache, g   as kernels/vae_norm_halo.py (raw tile and raw neighbour edges)
      wt [C, 27 * CO] bf16, bias [CO, 1] fp32   as kernels/vae_head_conv.py
  = F.conv3d(buf, w, bias), buf[:, -2:] with buf = norm_halo(x, left, right, up, down, cache, g).

kernels/vae_norm_halo.py writes the normalized conv input to HBM and the compiler's conv reads it back, so the
norm (Vector / Scalar / one ones-matmul per 512 positions) and the conv (Tensor) run one after the other. Here the
tile is processed in bands of R output rows: each band's R + 2 padded input rows of every frame are loaded raw
(cache frames: already normalized), normalized in SBUF (the two overlap rows of neighbouring bands are normalized
twice) and convolved straight from SBUF (kernels/vae_head_conv.py's flat-shift matmuls), so one band's norm runs
under another band's matmuls. The last two frames' rows go out as the new cache. Per position the norm is the same
arithmetic as kernels/vae_norm_halo.py (fp32 squares and ones-matmul by default); the conv accumulates in fp32 PSUM
(taps outer, channel tiles inner). C, CO > 128 in partition tiles of 128.
LINGBOT_VAE_NKI_CONV_R: output rows per band (default 12, the largest divisor of H not above it; 20 and 24 were no
better: 1678 / 1735 vs 1692-1705 us for the 96 -> 96 conv unit, perf/vae_head_bench.py VARIANT=fused).
LINGBOT_VAE_NKI_CONV_LA=n: issue each output frame's matmuls after n more frames' norms (default 0: in order; 1 and 2
measured the same as 0 within 15 us). DO NOT use n >= 1 above 4 frames per call: in the 16-rank decode with 8 frames
per call (LINGBOT_VAE_T=2) LA=1 corrupted a burst of frames (from output frame 5 on) in about half the decodes,
run-dependent and more often under host load, while this kernel alone on one core was exact (perf/vae_conv_t_check.py,
perf/vae_repeat_check.sh). With LA=0, 8 and 12 frames per call were clean; 16 frames corrupted 3 of 12 decodes (from
output frame 2), so the race is not only the look-ahead; keep the calls at <= 12 frames.
LINGBOT_VAE_NKI_CONV_EDGE=copy: load the left / right halo columns contiguously and place them with a GpSimd copy
(kernels/vae_norm_halo.py's LINGBOT_VAE_NKI_HALO_EDGE=copy) instead of one 2-byte DMA element per row. REFUTED
(accuracy): bit-identical in the one-core benches (96 -> 96 conv unit 1580 -> 1530 us, head 1614 -> 1514 us with
SQ=bf16) but in the 16-rank decode vs_R fell to 52.67 dB (585 ms) and 52.86 dB with LINGBOT_VAE_NKI_CONV_RES (582 ms),
against 52.87 without it: run-dependent, so most likely a missed dependency between the GpSimd column copies and the
norm that reads the tile (the first and last bands' up / down row DMAs come after the copies in program order).
LINGBOT_VAE_NKI_CONV_SQ=bf16: bf16 squares and a bf16 ones-matmul (one PE pass instead of two; the sum of squares
to ~1e-4 relative, below the bf16 rounding of the normalized values) instead of fp32: ~2% of the normalized values
move by one bf16 ulp (conv output rel_l2 1.7e-3 vs the fp32 path), decode vs_R unchanged at 52.87 dB; 96 -> 96 conv
unit 1705 -> 1580 us. The settings go into the kernel as constant arguments: read inside the kernel as module
globals they are not part of the NKI / NEFF cache key, and a process with another setting reused the first one's
kernel.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki

_P = 128
_B = 512          # one PSUM bank of fp32 per partition
_R = int(os.environ.get("LINGBOT_VAE_NKI_CONV_R", "12"))
_SQ16 = os.environ.get("LINGBOT_VAE_NKI_CONV_SQ", "fp32") == "bf16"
_LA = int(os.environ.get("LINGBOT_VAE_NKI_CONV_LA", "0"))
_EDGE = os.environ.get("LINGBOT_VAE_NKI_CONV_EDGE", "") == "copy"


def _norm_tiles(xts, yts, n, cs, ones, gts, sq16):
    """yts[k][0:cs[k], 0:n] = silu(x * rsqrt(sum_c x^2 + 1e-24) * g) over the channel tiles xts[k] (flat SBUF)."""
    nct = len(cs)
    sdt = nl.bfloat16 if sq16 else nl.float32
    sqs = []
    for k in range(nct):
        sq = nl.ndarray((_P, n), dtype=sdt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq[0:cs[k], 0:n], data1=xts[k][0:cs[k], 0:n], data2=xts[k][0:cs[k], 0:n],
                           op=nl.multiply)
        sqs.append(sq)
    for co in range(nct):
        inv = nl.ndarray((_P, n), dtype=nl.float32, buffer=nl.sbuf)
        for b in range((n + _B - 1) // _B):
            b0 = b * _B
            bs = min(_B, n - b0)
            ps = nl.ndarray((_P, _B), dtype=nl.float32, buffer=nl.psum)
            for k in range(nct):
                nisa.nc_matmul(dst=ps[0:cs[co], 0:bs], stationary=ones[0:cs[k], 0:cs[co]],
                               moving=sqs[k][0:cs[k], b0:b0 + bs], accumulate=(k > 0))
            nisa.activation(dst=inv[0:cs[co], b0:b0 + bs], op=nl.rsqrt, data=ps[0:cs[co], 0:bs], scale=1.0,
                            bias=1e-24)
        t = nl.ndarray((_P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t[0:cs[co], 0:n], data1=xts[co][0:cs[co], 0:n], data2=inv[0:cs[co], 0:n],
                           op=nl.multiply)
        nisa.activation(dst=yts[co][0:cs[co], 0:n], op=nl.silu, data=t[0:cs[co], 0:n], scale=gts[co][0:cs[co], 0:1])


@nki.jit
def _norm_halo_conv(x, left, right, up, down, cache, g, wt, bias, rows=12, la=0, sq16=0, edge=0):
    return _body(x, left, right, up, down, cache, g, wt, bias, None, rows, la, sq16, edge)


@nki.jit
def _norm_halo_conv_res(x, left, right, up, down, cache, g, wt, bias, res, rows=12, la=0, sq16=0, edge=0):
    return _body(x, left, right, up, down, cache, g, wt, bias, res, rows, la, sq16, edge)


def _body(x, left, right, up, down, cache, g, wt, bias, res, rows, la, sq16, edge):
    C, T, H, W = x.shape
    Hp, Wp = H + 2, W + 2
    CO = bias.shape[0]
    nct, ncot = (C + _P - 1) // _P, (CO + _P - 1) // _P
    cs = [min(_P, C - k * _P) for k in range(nct)]
    cos = [min(_P, CO - k * _P) for k in range(ncot)]
    R = max(r for r in range(1, min(rows, H) + 1) if H % r == 0)
    NB = H // R
    FL = (R + 2) * Wp
    NQ = R * Wp - 2                    # the band's last valid output is at R * Wp - 3; its last input at FL - 1
    nq = (NQ + _B - 1) // _B
    y = nl.ndarray((CO, T, H, W), dtype=x.dtype, buffer=nl.shared_hbm)
    nc = nl.ndarray((C, 2, Hp, Wp), dtype=x.dtype, buffer=nl.shared_hbm)
    ones = nl.ndarray((_P, _P), dtype=nl.bfloat16 if sq16 else nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    gts, wss = [], []
    for k in range(nct):
        c0 = k * _P
        gt = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gt[0:cs[k], 0:1], src=g[c0:c0 + cs[k], 0:1])
        gts.append(gt)
        ws = nl.ndarray((_P, 27 * CO), dtype=wt.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ws[0:cs[k], 0:27 * CO], src=wt[c0:c0 + cs[k], 0:27 * CO])
        wss.append(ws)
    bss = []
    for m in range(ncot):
        bs = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bs[0:cos[m], 0:1], src=bias[m * _P:m * _P + cos[m], 0:1])
        bss.append(bs)

    def frame(j, f, nts):
        """Frame f's band j rows into SBUF (cache frames as they are, new frames normalized); the new cache rows."""
        h0 = j * R                                         # padded rows h0 .. h0 + R + 1
        nfs = [nl.ndarray((_P, R + 2, Wp), dtype=x.dtype, buffer=nl.sbuf) for k in range(nct)]
        if f < 2:
            for k in range(nct):
                c0 = k * _P
                nisa.dma_copy(dst=nfs[k][0:cs[k], 0:R + 2, 0:Wp], src=cache[c0:c0 + cs[k], f, h0:h0 + R + 2, 0:Wp])
        else:
            t = f - 2
            p0, p1 = max(h0, 1), min(h0 + R + 2, Hp - 1)  # interior padded rows (raw rows p - 1)
            d0, nr = p0 - h0, p1 - p0
            xts = []
            for k in range(nct):
                c0, c1 = k * _P, k * _P + cs[k]
                xt = nl.ndarray((_P, R + 2, Wp), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt[0:cs[k], d0:d0 + nr, 1:W + 1], src=x[c0:c1, t, p0 - 1:p1 - 1, 0:W])
                if edge:
                    for src, col in ((left, 0), (right, W + 1)):
                        et = nl.ndarray((_P, R + 2, 1), dtype=x.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=et[0:cs[k], 0:nr, 0:1], src=src[c0:c1, t, p0 - 1:p1 - 1, 0:1])
                        nisa.tensor_copy(dst=xt[0:cs[k], d0:d0 + nr, col:col + 1], src=et[0:cs[k], 0:nr, 0:1],
                                         engine=nisa.gpsimd_engine)
                else:
                    nisa.dma_copy(dst=xt[0:cs[k], d0:d0 + nr, 0:1], src=left[c0:c1, t, p0 - 1:p1 - 1, 0:1])
                    nisa.dma_copy(dst=xt[0:cs[k], d0:d0 + nr, W + 1:W + 2], src=right[c0:c1, t, p0 - 1:p1 - 1, 0:1])
                if h0 == 0:
                    nisa.dma_copy(dst=xt[0:cs[k], 0, 0:Wp], src=up[c0:c1, t, 0, 0:Wp])
                if h0 + R + 2 == Hp:
                    nisa.dma_copy(dst=xt[0:cs[k], R + 1, 0:Wp], src=down[c0:c1, t, 0, 0:Wp])
                xts.append(xt.reshape((_P, FL)))
            _norm_tiles(xts, [v.reshape((_P, FL)) for v in nfs], FL, cs, ones, gts, sq16)
        if f >= T:                                         # one of the last two frames: the new cache
            rr = R + 2 if j == NB - 1 else R
            for k in range(nct):
                c0 = k * _P
                nisa.dma_copy(dst=nc[c0:c0 + cs[k], f - T, h0:h0 + rr, 0:Wp], src=nfs[k][0:cs[k], 0:rr, 0:Wp])
        nts[f] = [v.reshape((_P, FL)) for v in nfs]

    def conv(j, t, nts):
        """Output frame t of band j from frames t, t + 1, t + 2."""
        h0 = j * R
        for m in range(ncot):
            m0, ms = m * _P, cos[m]
            ot = nl.ndarray((_P, R, Wp), dtype=x.dtype, buffer=nl.sbuf)
            otf = ot.reshape((_P, R * Wp))
            for b in range(nq):
                q0 = b * _B
                n = min(_B, NQ - q0)
                ps = nl.ndarray((_P, _B), dtype=nl.float32, buffer=nl.psum)
                i = 0
                for tap in range(27):
                    kt, kh, kw = tap // 9, (tap // 3) % 3, tap % 3
                    s = q0 + kh * Wp + kw
                    for k in range(nct):
                        nisa.nc_matmul(dst=ps[0:ms, 0:n], stationary=wss[k][0:cs[k], tap * CO + m0:tap * CO + m0 + ms],
                                       moving=nts[t + kt][k][0:cs[k], s:s + n], accumulate=(i > 0))
                        i += 1
                nisa.activation(dst=otf[0:ms, q0:q0 + n], op=nl.copy, data=ps[0:ms, 0:n], bias=bss[m][0:ms, 0:1],
                                scale=1.0)
            if res is not None:                            # + the residual, rounded like torch's bf16 y + h
                rt = nl.ndarray((_P, R, W), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rt[0:ms, 0:R, 0:W], src=res[m0:m0 + ms, t, h0:h0 + R, 0:W])
                o2 = nl.ndarray((_P, R, W), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=o2[0:ms, 0:R, 0:W], data1=ot[0:ms, 0:R, 0:W], data2=rt[0:ms, 0:R, 0:W],
                                   op=nl.add)
                nisa.dma_copy(dst=y[m0:m0 + ms, t, h0:h0 + R, 0:W], src=o2[0:ms, 0:R, 0:W])
            else:
                nisa.dma_copy(dst=y[m0:m0 + ms, t, h0:h0 + R, 0:W], src=ot[0:ms, 0:R, 0:W])

    # Program order: natural (band, frame) order, but each output frame's matmuls are issued la frames later, so
    # with la >= 1 the Tensor queue holds the next frame's norm ones-matmuls before this conv and the norm's
    # Scalar / Vector tail runs under the conv instead of stalling the next one.
    ops = []
    for j in range(NB):
        nts = [None] * (T + 2)
        for f in range(T + 2):
            ops.append((0, j, f, nts))
            if f >= 2:
                ops.append((1, j, f - 2, nts))
    pending, seen = [], 0
    for kind, j, f, nts in ops:
        if kind == 1:
            pending.append((j, f, nts, seen))
        else:
            frame(j, f, nts)
            seen += 1
        while pending and seen - pending[0][3] >= la:
            pj, pt, pn, _ = pending.pop(0)
            conv(pj, pt, pn)
    for pj, pt, pn, _ in pending:
        conv(pj, pt, pn)
    return y, nc


_kernel = wrap_nki(_norm_halo_conv)
_kernel_res = wrap_nki(_norm_halo_conv_res)


def norm_halo_conv(x, left, right, up, down, cache, g, wt, bias, res=None):
    """Traceable inside a fullgraph torch.compile region; res [CO, T, H, W] bf16: y = conv + res (the ResidualBlock's
    shortcut add, LINGBOT_VAE_NKI_CONV_RES). The variant settings go in as constant kernel arguments (part of the
    kernel's cache key; module globals read inside the kernel are not)."""
    kw = dict(rows=_R, la=_LA, sq16=int(_SQ16), edge=int(_EDGE))
    if res is not None:
        return _kernel_res(x, left, right, up, down, cache, g, wt, bias, res, **kw)
    return _kernel(x, left, right, up, down, cache, g, wt, bias, **kw)
