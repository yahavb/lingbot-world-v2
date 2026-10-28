"""Precompute the job's prompt embeddings (T5-TP over the whole world, as generate.py's inline path
does) and save them in generate.py's --prompt_embeds_path format: {'prompt_embeds': [tensor]}.

    cd <repo> && PYTHONPATH=<repo> torchrun --nproc_per_node=64 kiro/encode_prompt.py <out.pt>

Used for the job-layout baseline: with the core order from core_order.py, large collectives over
the full 64-rank world abort in the runtime (rdh_scheduler::verify_read_safety) while TP and CP
group collectives run, so T5 -- the only world-wide large collective in generate.py -- is run
here with the default core order and its output passed in.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel, destroy_t5_parallel_group, init_t5_parallel_group

PROMPT = ("gentle ripples spreading across the calm lake water, tree leaves swaying softly in the "
          "breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, "
          "serene natural motion")
CKPT = "/work/ckpt/lingbot-world-v2-14b-causal-fast"


def main(out):
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.neuron.set_device(local)
    dist.init_process_group(backend="neuron", rank=rank, world_size=world)
    cfg = WAN_CONFIGS["i2v-A14B"]
    device = torch.device(f"neuron:{local}")
    init_t5_parallel_group(world)
    enc = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=torch.device("cpu"),
                         checkpoint_path=os.path.join(CKPT, cfg.t5_checkpoint),
                         tokenizer_path=os.path.join(CKPT, cfg.t5_tokenizer))
    enc.model.to(device)
    with torch.no_grad():
        embeds = [t.cpu() for t in enc([PROMPT], device)]
    destroy_t5_parallel_group()
    dist.barrier()
    if rank == 0:
        torch.save({"prompt_embeds": embeds, "prompt": PROMPT}, out)
        print(f"[encode_prompt] saved {out}: {[tuple(t.shape) for t in embeds]} "
              f"{embeds[0].dtype}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
