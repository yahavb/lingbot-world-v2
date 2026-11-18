"""One-core A/B and exactness of the upsample conv (wan/modules/vae2_1_fused._up_core, nearest x2 + 3x3 conv) vs
kernels/vae_up_conv.py (_up_core_nki, LINGBOT_VAE_NKI_UP), fake halos (the tile's own opposite edges, the same on
both sides), at the 2-D 16-rank per-rank shapes.
    NEURON_RT_VISIBLE_CORES=0 SHAPE=1,192,4,60,96 CO=96 FOLD=0 python3 perf/vae_up_bench.py
SHAPE: up2d 1,192,4,60,96 (CO=96); upsample3d convs 1,384,4,30,48 and 1,384,2,15,24 (CO=192).
Prints UPBENCH (us/call of both) and UPCHECK (max_abs, rel_l2 of the kernel vs the compiler's conv).
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LINGBOT_VAE_2D", "1")
os.environ.setdefault("LINGBOT_VAE_NKI_UP", "all")
import torch  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from wan.modules import vae2_1_fused as vf  # noqa: E402
from kernels.vae_up_conv import up_conv_weights  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_rb_bench import fake_halo, fake_edges  # noqa: E402


def main():
    shape = [int(v) for v in os.environ.get("SHAPE", "1,192,4,60,96").split(",")]
    co = int(os.environ.get("CO", "96"))
    reps = int(os.environ.get("REPS", "20"))
    fold = os.environ.get("FOLD", "0") == "1"
    vf._NKI_UP_FOLD = fold
    b, C, T, H, W = shape
    torch.manual_seed(0)
    dev = torch.device("neuron:0")
    bf = torch.bfloat16
    x = (torch.randn(shape) * 2).to(bf).to(dev)
    w32 = torch.randn(co, C, 3, 3) * (1.0 / (9 * C) ** 0.5)
    wb = w32.to(bf).to(dev)
    bb = (torch.randn(co) * 0.1).to(bf).to(dev)
    wk = up_conv_weights(w32 if fold else w32.to(bf), fold=fold).to(bf).to(dev)
    vf._halo = fake_halo
    vf._raw_edges_2d = fake_edges

    def f_ref(x, w, bb):
        return vf._up_core(x, w, bb, None)

    def f_nki(x, w, bb):
        return vf._up_core_nki(x, w, bb, None)
    outs = {}
    for name, fn, w in (("ref", f_ref, wb), ("nki", f_nki, wk)):
        cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        outs[name] = cf(x, w, bb).cpu().float()
        tc = time.perf_counter() - t0
        for _ in range(3):
            cf(x, w, bb)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            cf(x, w, bb)
        torch.neuron.synchronize()
        print(f"UPBENCH {name} fold={int(fold)} shape={shape} co={co} us/call={(time.perf_counter() - t0) / reps * 1e6:.1f} "
              f"first={tc:.1f}s", flush=True)
    a, r = outs["nki"], outs["ref"]
    d = a - r
    print(f"UPCHECK fold={int(fold)} shape={shape} out={tuple(a.shape)} max_abs={d.abs().max().item():.3e} "
          f"rel_l2={(d.norm() / r.norm()).item():.3e} n_diff={(d != 0).sum().item()} ref_absmax={r.abs().max().item():.3e}",
          flush=True)


if __name__ == "__main__":
    main()
