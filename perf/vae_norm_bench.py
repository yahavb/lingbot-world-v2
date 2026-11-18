"""One-core A/B of the VAE decoder's F.silu(RMS_norm(x)).to(bf16): the compiled torch form (LINGBOT_VAE_NORM, default
mm) vs the fused NKI kernel (kernels/vae_rms_silu.py, LINGBOT_VAE_NKI_NORM=1). Error of both vs an fp64 CPU reference.
    NEURON_RT_VISIBLE_CORES=0 python3 perf/vae_norm_bench.py            SHAPE=1,96,4,120,192 (stage 3) REPS=20
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LINGBOT_VAE_NORM", "mm")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from wan.modules import vae2_1_fused as vf  # noqa: E402
from kernels.vae_rms_silu import rms_silu  # noqa: E402


def main():
    shape = [int(v) for v in os.environ.get("SHAPE", "1,96,4,120,192").split(",")]
    reps = int(os.environ.get("REPS", "20"))
    C = shape[1]
    scale = C ** 0.5
    torch.manual_seed(0)
    dev = torch.device("neuron:0")
    xc = (torch.randn(shape) * 2.5 + 0.3).to(torch.bfloat16)
    gc = (1 + 0.2 * torch.randn(C, 1, 1, 1)).float()
    x, g = xc.to(dev), gc.to(dev)

    def ref(x, g):
        return F.silu(vf._rms(x, g, scale)).to(torch.bfloat16)

    def nki_(x, g):
        c = x.shape[1]
        return rms_silu(x.reshape(c, -1), g.reshape(c, 1) * scale).view(x.shape)

    y64 = F.silu(F.normalize(xc.double(), dim=1) * scale * gc.double())
    for name, fn in (("torch_" + vf._NORM_MODE, ref), ("nki", nki_)):
        cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        y = cf(x, g)
        torch.neuron.synchronize()
        tc = time.perf_counter() - t0
        for _ in range(3):
            y = cf(x, g)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            y = cf(x, g)
        torch.neuron.synchronize()
        dt = (time.perf_counter() - t0) / reps * 1e6
        d = (y.cpu().double() - y64)
        rel = d.norm() / y64.norm()
        print(f"NORMBENCH {name} shape={shape} us/call={dt:.1f} first={tc:.1f}s max_abs={d.abs().max().item():.3e} "
              f"rel_l2={rel.item():.3e}", flush=True)


if __name__ == "__main__":
    main()
