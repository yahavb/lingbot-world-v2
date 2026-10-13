import argparse
import logging
import math
import gc
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image


def _unmask_nki_tracebacks():
    """Stop nki's traceback filter from hiding the real kernel error under torch.compile.

    When a NKI kernel raises inside a compiled region, nki._traceback_util wraps it and
    calls filter_traceback -> include_frame -> os.path.abspath(f.tb_frame.f_code.co_filename).
    Dynamo cannot type a traceback-derived attribute, so its os polyfill raises
    "builtin isinstance() cannot determine type of argument" and REPLACES the original
    exception. The log then shows a dynamo complaint about isinstance while the actual
    kernel error -- shape, tile, collective -- is never printed.

    Only the error path is affected: filter_traceback runs when an exception already
    exists, so making it the identity changes nothing on a successful run. It costs a
    longer traceback (unfiltered nki frames) in exchange for the message actually being
    the failure. NKI_FILTER_TRACEBACK=1 restores nki's behaviour.
    """
    if os.environ.get("NKI_FILTER_TRACEBACK", "0") == "1":
        return
    try:
        from nki import _traceback_util as _tu
    except Exception:
        return
    _tu.filter_traceback = lambda tb: tb

    # Patching filter_traceback is not enough: the wrapper ALSO calls e.add_note(_note) at
    # _traceback_util.py:106, and add_note is a method on the exception object, not a module
    # global, so it cannot be swapped the same way. Dynamo cannot trace either call, so it
    # discards the kernel's real RuntimeError and reports a graph break instead.
    #
    # Neutralize the DECORATOR instead, which runs before any kernel is wrapped (this shim
    # executes before `import wan` -> kernels.const_max_ag). An unwrapped kernel raises its
    # RuntimeError directly, with nothing for dynamo to choke on. The decorator's name is not
    # documented, so find it by source inspection rather than guessing.
    import inspect
    for _name in dir(_tu):
        if _name.startswith("__"):
            continue
        _obj = getattr(_tu, _name, None)
        if not callable(_obj):
            continue
        try:
            _src = inspect.getsource(_obj)
        except Exception:
            continue
        if "add_note" in _src and "def wrapper" in _src:
            setattr(_tu, _name, lambda f: f)   # identity decorator
            print(f"[generate] nki._traceback_util.{_name} neutralized "
                  f"(it wraps kernels with add_note, which dynamo cannot trace)", flush=True)
    print("[generate] nki traceback filter disabled -- real kernel errors will surface "
          "(NKI_FILTER_TRACEBACK=1 to restore)", flush=True)


_unmask_nki_tracebacks()

import wan
from wan.configs import EXACT_GRID_SIZES, MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.modules.model_fast import destroy_dit_parallel_group
from wan.modules.t5 import (
    T5EncoderModel,
    destroy_t5_parallel_group,
    init_t5_parallel_group,
)
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.vae2_1_sharded import (
    _set_vae_group,
    gather_video_width,
    shard_vae_decoder_width,
    width_shard_latent,
)
from wan.utils.utils import save_video, str2bool


_I2V_EXAMPLE = {
    "prompt":
        "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
    "image":
        "examples/04/image.jpg",
}

EXAMPLE_PROMPT = {
    "i2v-A14B": _I2V_EXAMPLE,
    "i2v-1.3B": _I2V_EXAMPLE,
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task.startswith("i2v"):
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="causal_fast",
        choices=["causal_fast", "causal_pretrain"],
        help="Inference mode: 'causal_fast' for the distilled few-step model, "
             "'causal_pretrain' for the pretrained causal model with 40-step CFG sampling.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Denoise steps per chunk (default 4, the model's timesteps_index). "
             "the reference implementation's equivalent --steps is described there as the biggest single "
             "latency lever, ~linear, and their published 6.53 fps run used 2."),
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="The chunk size."),
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=None,
        help="Optional directory that holds shared T5 / VAE / tokenizer assets "
             "(used when the DiT checkpoint folder does not include them).")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor-parallel degree for the causal_fast DiT. With --sp_size, "
             "forms an sp_size x tp_degree rank mesh spanning world_size; 1:1 "
             "legacy is tp_degree == world_size.")
    parser.add_argument(
        "--sp_size",
        type=int,
        default=1,
        help="Sequence (Ulysses) parallel degree for the causal_fast DiT. "
             "sp_size * tp_degree == world_size; needs --frame_num aligned so "
             "max_seq_len is divisible by sp_size (<= num_heads).")
    parser.add_argument(
        "--cp_size",
        type=int,
        default=1,
        help="Context-parallel degree for the causal_fast DiT: shards the TOKEN "
             "axis (latent patch-grid height rows) and KEEPS it sharded through "
             "attention, unlike --sp_size (Ulysses), which all_to_alls the token "
             "shard into a HEAD shard. Consequence: Ulysses divides the 40 heads "
             "by sp_size*tp_degree, so world_size must divide 40 and 32 ranks are "
             "impossible (40 %% 32 = 8); CP divides heads by tp_degree ONLY, so "
             "cp_size multiplies the rank count freely -- cp_size=4 x tp_degree=8 "
             "= 32 ranks. Mutually exclusive with --sp_size. Needs the latent "
             "patch-grid height divisible by cp_size (30 at 480*832, 60 at "
             "960*1664).")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--save_latents_path",
        type=str,
        default=None,
        help="If set, skip VAE decode and dump the pre-decode latent to this "
             "path instead (for the width-sharded decode stage, see "
             "decode_latents_width_sharded.py).")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--prompt_embeds_path",
        type=str,
        default=None,
        help="Path to a prompt_embeds .pt saved by encode_prompt_sharded.py "
        "(Stage 1 of the two-stage T5-TP pipeline). When set, skips the "
        "inline text encoder and loads precomputed embeddings instead; "
        "--prompt is not required and is not re-encoded. See "
        "run_fast_neuron.sh.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help='The local size of kv cache during inference')
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help='The sink size of kv cache during inference')
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _encode_prompt_tp(args, cfg, device, rank, world_size):
    """Inline TP T5 encode for the single-torchrun e2e path (tp_degree>1, no
    --prompt_embeds_path given). Returns the prompt_embeds tensors directly
    -- T5-TP's all-reduce already makes them identical on every rank, so
    there's no cross-rank handoff to do; no disk round-trip needed since
    encode and consume happen in this same process."""
    init_t5_parallel_group(world_size)
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )
    text_encoder.model.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder([args.prompt], device)
    prompt_embeds = [t.cpu() for t in prompt_embeds]
    del text_encoder
    destroy_t5_parallel_group()
    torch.neuron.empty_cache()
    dist.barrier()  # all ranks done with the "t5-tp" group before it's torn down
    return prompt_embeds


def _decode_latents_width_sharded(args, cfg, device, rank, world_size, latent):
    """Inline width-sharded VAE decode for the single-torchrun e2e path
    (tp_degree>1, no --save_latents_path given). Same logic as
    decode_latents_width_sharded.py, run here instead of as a separate
    torchrun launch. `latent` is the tensor returned directly by
    WanI2VCausal.generate(return_latents=True) -- no disk round-trip."""
    # The width shard needs lat_w % participants == 0, and lat_w is set by the
    # IMAGE aspect, not by the rank count: examples/03 at 480*832 gives 104 =
    # 2^3 * 13, which divides 8 but not 32, so a 32-rank run asserted
    # "width=104 not divisible by world=32" AFTER a successful 671 s denoise.
    # Decode on the largest divisor of lat_w that fits the world and let the
    # surplus ranks sit out; they rejoin at the caller's barrier. The VAE stage
    # is ~23% of steady-state time and is not the CP subject, so trading some of
    # its parallelism to keep the run alive is the right way round.
    # Two constraints, not one:
    #   lat_w % vae_world == 0        -- the width shard itself
    #   world_size % vae_world == 0   -- Neuron's collective lowering, which
    #                                    rejected 26 of 32 with "Cannot create
    #                                    global replica groups: world_size (32)
    #                                    not divisible by group_size (26)"
    # The largest number satisfying both is gcd. For lat_w=104 (2^3*13) and
    # world=32 that is 8, decoding 13 px per rank -- which is exactly the shape
    # the working 8-rank baseline used, so it is a proven configuration.
    # BUILD ONCE, DECODE MANY. This used to construct the VAE, call dist.new_group and
    # apply shard_vae_decoder_width on EVERY call, which was invisible while the decode
    # ran once. With a second timed pass it would have rebuilt the model, created a
    # duplicate process group, and handed _compile_pointwise 66 brand-new module objects
    # to compile again -- so pass 2 would have measured a fresh compile and proved
    # nothing. Memoised on (vae_world, ckpt).
    # WIDTH PADDING TO USE THE WHOLE MACHINE.
    #
    # gcd alone leaves most of the machine idle: lat_w=104 is 2^3*13, so the largest
    # divisor it shares with 32 is 8 and 24 of 32 cores sat out the decode while the DiT
    # used all 32. Padding the latent width up to the next multiple of world_size makes
    # every rank a participant (104 -> 128, 4 px per rank instead of 13 on 8 ranks).
    #
    # NOT numerically neutral, and this is the cost, stated rather than buried: the
    # decoder's right-hand columns now see replicated latent content where they
    # previously saw the tensor edge, so the rightmost output pixels change. Replicate
    # (not zeros) keeps the perturbation as small as possible, and the padded region is
    # cropped after the gather, but the true edge is still affected within the decoder's
    # receptive field. Compare the mp4 against an unpadded run before trusting it.
    lat_w = latent.shape[-1]
    _pad_w = ((lat_w + world_size - 1) // world_size) * world_size
    if _pad_w != lat_w and world_size > math.gcd(lat_w, world_size):
        # Explicit edge replicate. F.pad(mode="replicate") rejects a 2-tuple at 4-D or
        # 5-D ("Padding size 2 is not supported for 4D input tensor") and would have
        # crashed in the pod; a cat of the repeated last column is rank-agnostic.
        latent = torch.cat(
            [latent, latent[..., -1:].expand(*latent.shape[:-1], _pad_w - lat_w)],
            dim=-1)
        vae_world = world_size
    else:
        _pad_w = lat_w
        vae_world = math.gcd(lat_w, world_size)
    _key = (vae_world, args.ckpt_dir, cfg.vae_checkpoint)
    _cached = getattr(_decode_latents_width_sharded, "_built", None)
    if _cached is None or _cached[0] != _key:
        group = None
        if world_size > 1:
            # dist.new_group is COLLECTIVE: every rank must call it, member or not.
            group = dist.new_group(ranks=list(range(vae_world)))
        if rank == 0:
            if _pad_w != lat_w:
                logging.info(
                    "VAE width-shard: latent width %d padded to %d (replicate) so all "
                    "%d ranks participate, %d px each; cropped after gather.",
                    lat_w, _pad_w, vae_world, _pad_w // vae_world)
            elif vae_world != world_size:
                logging.info(
                    "VAE width-shard: latent width %d does not divide world %d; "
                    "decoding on %d ranks (%d px each), %d idle.",
                    lat_w, world_size, vae_world, lat_w // vae_world,
                    world_size - vae_world)
        if rank >= vae_world:
            _decode_latents_width_sharded._built = (_key, None, None)
            return None
        _set_vae_group(group)
        vae = Wan2_1_VAE(
            vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=device,
            dtype=torch.bfloat16)
        if vae_world > 1:
            shard_vae_decoder_width(vae.model.decoder)
        _decode_latents_width_sharded._built = (_key, vae, group)
    else:
        _, vae, group = _cached
        if vae is None:
            return None
        _set_vae_group(group)

    latent = latent.to(device)
    latent_local = width_shard_latent(latent, rank=rank, world=vae_world)
    video_local = vae.decode([latent_local])[0]
    video_full = gather_video_width(video_local)
    if rank == 0 and _pad_w != lat_w:
        # Crop the padded columns. cfg.vae_stride is (t, h, w); width scale is [2].
        video_full = video_full[..., :lat_w * cfg.vae_stride[2]].contiguous()
    return video_full if rank == 0 else None


# Exceptions prewarm must NOT swallow: control flow, not failure. Named by class so
# capture_dit_inputs.py does not have to be imported here.
_PREWARM_PASSTHROUGH = (KeyboardInterrupt, SystemExit)


def run_causal(args, cfg, img, device, rank, mode="causal_fast",
               prompt_embeds=None, return_latents=False):
    """Run inference with the distilled few-step model (infer_mode='causal_fast')."""
    _load_start = time.perf_counter()
    logging.info(f"Creating WanI2VCausal pipeline (infer_mode={mode}).")
    wan_i2v = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode=mode,
        assets_dir=args.assets_dir,
        skip_text_encoder=(prompt_embeds is not None or args.prompt_embeds_path is not None),
        tp_degree=args.tp_degree,
        sp_size=args.sp_size,
        seq_mode=getattr(args, 'seq_mode', 'ulysses'),
    )
    # Split one-time setup from steady-state generation. The [BENCH] DiT marker
    # used to bracket ALL of run_causal from generate(), i.e. pipeline
    # construction (37 GB of DiT weights, VAE, T5) plus generation -- measured
    # 88.7 s where the denoise loop alone is 48.0 s, so ~41 s of "DiT time" was
    # weight loading. Warm and cold runs matched to within 0.6%, confirming it
    # is loading and not compilation. Anyone reading that field as DiT compute
    # was off by ~85%.
    #
    # Load cost is paid once; a streaming deployment cares about the
    # steady-state rate. Report them separately.
    # PREWARM BEFORE THE TIMED WINDOW. prewarm() existed but nothing ever called
    # it -- documented and dead. It matters now: model_fast passes the KV write
    # window as plain ints, so there is one compiled graph PER chunk position, and
    # compiling them lazily puts the compile time inside GEN. Measured, cp4 with
    # compiled blocks reported GEN 1119.9 s against 705.4 s eager while the
    # DiT-only replay (chunk 0 only, one graph) was 108.0 ms/forward.
    #
    # Counted inside LOAD, which is where one-time cost belongs and which the
    # frame-rate block already excludes from steady state.
    if mode == "causal_fast" and os.environ.get("LINGBOT_PREWARM", "1") == "1":
        try:
            _pw = time.perf_counter()
            wan_i2v.prewarm(img, max_area=MAX_AREA_CONFIGS[args.size],
                            target_hw=EXACT_GRID_SIZES.get(args.size),
                            frame_num=args.frame_num, chunk_size=args.chunk_size,
                            text_seq_len=cfg.text_len)
            if rank == 0:
                print(f"[BENCH] PREWARM time: {(time.perf_counter() - _pw) * 1000:.1f} ms")
        except _PREWARM_PASSTHROUGH:
            # capture_dit_inputs.py raises a control-flow exception from inside the DiT
            # forward to stop as soon as it has the inputs. A bare `except Exception`
            # swallowed it: the capture marked itself done, prewarm reported "failed",
            # and generation carried on to completion -- at 768*1152 that exhausted the
            # device ("lazy::AllocBind: NRT_RESOURCE on retry after drain") and the
            # inputs that HAD been saved came from prewarm's DUMMY tensors, not real
            # generation. Let it through.
            raise
        except Exception as e:
            logging.warning("prewarm failed (%s); the first chunks will compile "
                            "inside the timed window instead", e)

    if rank == 0:
        print(f"[BENCH] LOAD time: {(time.perf_counter() - _load_start) * 1000:.1f} ms")
    logging.info("Generating video ...")
    # GENERATE TWICE, REPORT THE SECOND. the reference implementation the reference benchmark does exactly this:
    # "[run1] ... (compile+warmup; NOT latency)" and "RUN 2 -- steady state, fully
    # compiled and warm. THIS is the number."
    #
    # I built prewarm and assumed it removed the need for a second pass, then kept
    # calling pass 1's fps "steady-state" even after the evidence said otherwise: with
    # 1417.1 s of prewarm, the FIRST generation forward still cost 62.6 s and the worst
    # 82.4 s. Those are compile-shaped, so prewarm misses guard states that generation
    # visits, and every fps I have quoted was measured on the run that pays for them.
    #
    # LINGBOT_BENCH_RUNS=1 restores the single-pass behaviour.
    _runs = max(1, int(os.environ.get("LINGBOT_BENCH_RUNS", "2")))
    _gen_ms = []
    for _r in range(_runs):
        _gen_start = time.perf_counter()
        _result = wan_i2v.generate(
            args.prompt,
            img,
            action_path=args.action_path,
            chunk_size=args.chunk_size,
            max_area=MAX_AREA_CONFIGS[args.size],
            target_hw=EXACT_GRID_SIZES.get(args.size),
            denoise_steps=args.denoise_steps,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            seed=args.base_seed,
            offload_model=args.offload_model,
            max_attention_size=args.max_attention_size,
            save_latents_path=args.save_latents_path,
            prompt_embeds_path=args.prompt_embeds_path,
            prompt_embeds=prompt_embeds,
            return_latents=return_latents)
        if dist.is_initialized():
            torch.neuron.synchronize()
            dist.barrier()
        _gen_ms.append((time.perf_counter() - _gen_start) * 1000)
        if rank == 0:
            _tag = ("compile+warmup, NOT latency" if _r == 0 and _runs > 1
                    else "steady state" if _runs > 1 else "single pass")
            print(f"[BENCH] GEN run{_r + 1} time: {_gen_ms[-1]:.1f} ms  ({_tag})")
    # RELEASE THE DiT. Both GEN passes are done and run_causal returns latents, so the
    # DiT is never needed again -- yet its weights (5.15 GB/rank) and every NEFF stayed
    # resident through the VAE decode, and at 768*1152 that ran the device out:
    #   Failed to allocate 1363148800 bytes on DEVICE
    #   NRT Execution error occurred on Neuron for operation=allgather
    # The DiT had finished; only the width-sharded decode was still allocating. 2.29x the
    # pixels of 480p leaves no headroom to keep 5 GB of dead weights on the core.
    if getattr(wan_i2v, "model", None) is not None:
        _t_free = time.perf_counter()
        # Moving parameters to host is NOT enough, and the first attempt proved it: the
        # OOM simply moved from nrt_tensor_allocate (1.27 GiB) to
        #   NRT_RESOURCE in nrt_load_util / Failed to allocate resource
        # which is NEFF LOADING. model.cpu() moves parameters and empty_cache() drops the
        # allocator's blocks, but the DiT's compiled NEFFs stay loaded on the device --
        # the memory table showed 1300+ resident. They are reachable only through the
        # compiled callables, so drop those and reset Dynamo before collecting.
        _st = getattr(wan_i2v.model, "_stacks", None)
        if _st is not None:
            wan_i2v.model._stacks = None
            wan_i2v.model._stack_tokens = None
            wan_i2v.model._stacks_built = None
        wan_i2v.model.cpu()
        try:
            torch._dynamo.reset()
        except Exception as _e:
            print(f"[BENCH] dynamo reset failed ({_e})")
        gc.collect()
        torch.neuron.empty_cache()
        if dist.is_initialized():
            torch.neuron.synchronize()
            dist.barrier()
        if rank == 0:
            print(f"[BENCH] DiT released before VAE: "
                  f"{(time.perf_counter() - _t_free) * 1000:.0f} ms")
    if rank == 0:
        # The headline GEN is the LAST pass, so every fps derived from it is warm.
        print(f"[BENCH] GEN time: {_gen_ms[-1]:.1f} ms")
        if len(_gen_ms) > 1:
            print(f"[BENCH] GEN compile overhead in pass 1: "
                  f"{(_gen_ms[0] - _gen_ms[-1]) / 1000:.1f} s "
                  f"({_gen_ms[0] / max(_gen_ms[-1], 1e-9):.2f}x)")
    return _result


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    cfg = WAN_CONFIGS[args.task]

    if world_size > 1:
        torch.neuron.set_device(local_rank)
        dist.init_process_group(
            backend="neuron",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not args.dit_fsdp, \
            "dit_fsdp is not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.sp_size > 1
        ), "sequence parallel is not supported in non-distributed environments."
        assert not (
            args.tp_degree > 1
        ), "tensor parallel is not supported in non-distributed environments."

    assert not (args.sp_size > 1 and args.cp_size > 1), (
        "--sp_size (Ulysses, head-sharding inside attention) and --cp_size "
        "(token-axis, stays token-sharded) are two different uses of the same "
        "rank axis; pick one.")
    # CP reuses the sp mesh geometry, so everything below that sizes the mesh
    # treats cp_size as sp_size. Only the attention math differs, selected by
    # set_seq_parallel_mode() in image2video.py.
    seq_mode = "cp" if args.cp_size > 1 else "ulysses"
    args.seq_mode = seq_mode
    if args.cp_size > 1:
        args.sp_size = args.cp_size

    use_sp_mesh = args.sp_size > 1
    if args.ulysses_size > 1 or use_sp_mesh:
        if args.ulysses_size > 1:
            assert args.ulysses_size == world_size, "ulysses_size must equal world size."
            assert cfg.num_heads % args.ulysses_size == 0
        init_distributed_group()

    if args.tp_degree > 1 or use_sp_mesh:
        assert args.ulysses_size == 1, "--ulysses_size (legacy SP) cannot mix with --sp_size/--tp_degree."
        if use_sp_mesh:
            assert args.sp_size * args.tp_degree == world_size, (
                f"sp_size({args.sp_size}) * tp_degree({args.tp_degree}) must equal world_size({world_size}).")
            if seq_mode == "cp":
                # CP never converts the token shard into a head shard, so only
                # tp_degree consumes heads. THIS is what lifts the world-size
                # ceiling: with 40 heads, Ulysses needs sp*tp to divide 40 (so
                # 8 ranks max), while CP needs only tp to divide 40 -- 40/8 = 5
                # heads per rank at any cp_size.
                assert cfg.num_heads % args.tp_degree == 0, (
                    f"num_heads={cfg.num_heads} not divisible by tp_degree={args.tp_degree}.")
            else:
                assert cfg.num_heads % (args.sp_size * args.tp_degree) == 0, (
                    f"num_heads={cfg.num_heads} not divisible by sp*tp={args.sp_size * args.tp_degree}.")
        else:
            assert args.tp_degree == world_size, "tp_degree must equal world_size when --sp_size is not used."
            assert cfg.num_heads % args.tp_degree == 0

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Single-torchrun e2e mode: tp_degree>1 and the user hasn't opted into
    # the separate-stage debug path (encode_prompt_sharded.py /
    # decode_latents_width_sharded.py + run_fast_neuron.sh) via
    # --prompt_embeds_path / --save_latents_path. Those two flags keep
    # working as explicit opt-outs into the standalone per-stage scripts.
    user_wants_latents_only = args.save_latents_path is not None
    e2e_t5 = args.tp_degree > 1 and args.prompt_embeds_path is None
    e2e_vae = args.tp_degree > 1 and args.save_latents_path is None
    prompt_embeds = None
    if e2e_t5:
        prompt_embeds = _encode_prompt_tp(
            args, cfg, torch.device(f"neuron:{local_rank}"), rank, world_size)

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    dit_start = time.perf_counter()
    video = run_causal(args, cfg, img, device, rank, mode=args.infer_mode,
                        prompt_embeds=prompt_embeds, return_latents=e2e_vae)
    if dist.is_initialized():
        torch.neuron.synchronize()
        dist.barrier()
    if rank == 0:
        print(f"[BENCH] DiT time: {(time.perf_counter() - dit_start) * 1000:.1f} ms")

    if e2e_vae:
        destroy_dit_parallel_group()
        torch.neuron.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        video_in = video
        # DECODE TWICE, REPORT THE SECOND -- the same fix already applied to GEN.
        # I applied two-pass timing to generation and left the VAE single-pass, so when
        # the VAE gained 66 compiled regions its compilation landed inside the ONE timed
        # decode and read as a 12x regression (22.4 s -> 274.1 s). Whether the warm VAE
        # is faster or slower than eager was unmeasurable, which is the same mistake as
        # reporting cold GEN as steady state.
        _vae_runs = max(1, int(os.environ.get("LINGBOT_BENCH_RUNS", "2")))
        _vae_ms = []
        for _vr in range(_vae_runs):
            if _vr == _vae_runs - 1:
                # Zero the collective counters before the pass we actually report.
                # They accumulated across BOTH passes while the share was divided by one
                # pass, so the printed figure read 44.7% where the truth was ~22%.
                try:
                    from wan.modules.vae2_1_sharded import reset_vae_collective_stats
                    reset_vae_collective_stats()
                except Exception:
                    pass
            _t = time.perf_counter()
            video = _decode_latents_width_sharded(
                args, cfg, torch.device(f"neuron:{local_rank}"), rank, world_size,
                video_in)
            if dist.is_initialized():
                torch.neuron.synchronize()
                dist.barrier()
            _vae_ms.append((time.perf_counter() - _t) * 1000)
            if rank == 0 and _vae_runs > 1:
                _tag = ("compile+warmup, NOT latency" if _vr == 0 else "steady state")
                print(f"[BENCH] VAE run{_vr + 1} time: {_vae_ms[-1]:.1f} ms  ({_tag})")
        if rank == 0 and len(_vae_ms) > 1:
            print(f"[BENCH] VAE compile overhead in pass 1: "
                  f"{(_vae_ms[0] - _vae_ms[-1]) / 1000:.1f} s "
                  f"({_vae_ms[0] / max(_vae_ms[-1], 1e-9):.2f}x)")
        if rank == 0:
            # The LAST pass, so the frame-rate block downstream is warm like GEN's.
            _vae_s = _vae_ms[-1] / 1000.0
            print(f"[BENCH] VAE time: {_vae_ms[-1]:.1f} ms")
            # The VAE is 10x the whole DiT (192.6 s vs 19.3 s) and I treated it as a
            # fixed cost for the entire bring-up. Every collective in the width-sharded
            # decode is OUTSIDE the compiled leaves -- bare dist.all_gather_into_tensor
            # on _VAE_GROUP -- so the decoder is ~30 small NEFFs per latent frame glued
            # together with ~30 eager all-gathers. This says how much of the VAE that
            # accounts for, instead of me guessing at it.
            try:
                from wan.modules.vae2_1_sharded import (
                    vae_collective_stats, vae_compile_report)
                print(f"[BENCH] {vae_compile_report()}")
                _c = vae_collective_stats()
                _coll_s = _c["halo_s"] + _c["gather_s"]
                print(f"[BENCH] VAE collectives (outside every compiled region): "
                      f"halo={_c['halo']} gather={_c['gather']} "
                      f"total={_coll_s:.1f} s = {100 * _coll_s / max(_vae_s, 1e-9):.1f}% of VAE")
            except Exception as _e:
                print(f"[BENCH] VAE collectives: unavailable ({_e})")

    if rank == 0 and user_wants_latents_only:
        logging.info(f"Latents saved to {args.save_latents_path}; "
                     "skipping video save (run decode_latents_width_sharded.py next).")
    elif rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.mp4'
            args.save_file = f"lingbot-world-v2-{args.infer_mode}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    del video

    torch.neuron.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
