"""FFN epilogue between the two MXFP8 matmuls (LINGBOT_FFN_EPI=1): u [M, N] (ffn.0 output without its bias)
-> GELU_tanh(u + b) written into [M, n_out] with zero columns N..n_out (the down projection's K padding to 512).

The compiler's version of `gelu(u + b)` + `pad` at the 2-chip shape (M=2176, N=3456 -> 3584) ran ~200 us between
the two matmul kernels with PE transposes; this one loads whole rows (tokens on partitions, contiguous features),
adds the partition-broadcast bias on Vector, applies GELU on Scalar and stores rows. LNC2 splits the rows.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki


@nki.jit
def _bias_gelu_pad(u, b, n_out):
    """u [M, N] bf16 HBM, b [1, N] bf16 HBM, n_out >= N (constant) -> [M, n_out] bf16."""
    M, N = u.shape
    P = 128
    out = nl.ndarray((M, n_out), dtype=u.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs(axes=0)
    rows = M // nprog
    r_base = nl.program_id(axis=0) * rows
    b_sb = nl.ndarray((P, N), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sb, src=b.ap(pattern=[[0, P], [1, N]], offset=0))     # bias on every partition
    n_tiles = (rows + P - 1) // P
    for t in nl.affine_range(n_tiles):
        sz = min(P, rows - t * P)
        x = nl.ndarray((P, N), dtype=u.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:sz, 0:N], src=u[nl.ds(r_base + t * P, sz), 0:N])
        s = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=s[0:sz, 0:N], data1=x[0:sz, 0:N], data2=b_sb[0:sz, 0:N], op=nl.add)
        y = nl.ndarray((P, n_out), dtype=u.dtype, buffer=nl.sbuf)
        nisa.activation(dst=y[0:sz, 0:N], op=nl.gelu_apprx_tanh, data=s[0:sz, 0:N])
        if n_out > N:
            nisa.memset(dst=y[0:sz, N:n_out], value=0.0)
        nisa.dma_copy(dst=out[nl.ds(r_base + t * P, sz), 0:n_out], src=y[0:sz, 0:n_out])
    return out


bias_gelu_pad = wrap_nki(_bias_gelu_pad)
