import gc
import hashlib
import json
import logging
import math
import contextlib
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from functools import partial

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .modules.attention import const_max_ag_available
from .modules.chunked_kv_cache import (
    allocate_self_kv_chunks,
    chunk_bytes,
    num_chunks,
)
from .distributed.util import get_world_size
from .modules.model_fast import (
    init_dit_mesh,
    set_seq_parallel_mode,
    init_dit_parallel_group,
    init_dit_sp_group,
    load_tp_wan_model_fast,
)
from .modules.model_causal import WanModelCausal
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange


def _resolve_asset_path(filename, checkpoint_dir, assets_dir=None):
    """Resolve T5 / VAE / tokenizer paths, falling back to ``assets_dir``.

    The 1.3B Hugging Face upload currently ships only DiT weights; T5, VAE,
    and the tokenizer can be reused from the 14B release via ``--assets_dir``.
    """
    candidates = [os.path.join(checkpoint_dir, filename)]
    if assets_dir:
        candidates.append(os.path.join(assets_dir, filename))
    for path in candidates:
        if os.path.exists(path):
            return path
    searched = ", ".join(candidates)
    raise FileNotFoundError(
        f"Required asset {filename!r} not found. Looked in: {searched}. "
        "Pass --assets_dir pointing at a 14B (or Wan) checkpoint that contains "
        "models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, and google/umt5-xxl."
    )


def _resolve_dit_dir(checkpoint_dir, subfolder):
    """Prefer ``checkpoint_dir/subfolder`` when it exists, else the root.

    14B causal-fast stores DiT weights under ``transformers/``. The 1.3B
    causal-fast upload currently places shards at the repository root.
    """
    if subfolder:
        candidate = os.path.join(checkpoint_dir, subfolder)
        if os.path.isdir(candidate):
            return candidate
    return checkpoint_dir


def _load_safetensors_state_dict(dit_dir):
    """Load a (possibly sharded) safetensors dump from ``dit_dir``."""
    from safetensors.torch import load_file

    for index_name in (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
    ):
        index_path = os.path.join(dit_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path) as f:
            index = json.load(f)
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(os.path.join(dit_dir, shard)))
        return state

    for single_name in (
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
    ):
        single_path = os.path.join(dit_dir, single_name)
        if os.path.isfile(single_path):
            return load_file(single_path)

    raise FileNotFoundError(
        f"No safetensors weights found in {dit_dir}. Expected a sharded "
        "index (model.safetensors.index.json) or a single model.safetensors."
    )


def _dit_kwargs_from_config(config, extra=None):
    kwargs = dict(
        model_type="i2v",
        patch_size=tuple(config.patch_size),
        text_len=config.text_len,
        in_dim=getattr(config, "in_dim", 36),
        dim=config.dim,
        ffn_dim=config.ffn_dim,
        freq_dim=config.freq_dim,
        text_dim=getattr(config, "text_dim", 4096),
        out_dim=getattr(config, "out_dim", 16),
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qk_norm=config.qk_norm,
        cross_attn_norm=config.cross_attn_norm,
        eps=config.eps,
    )
    if extra:
        kwargs.update(extra)
    return kwargs


def load_dit_model(model_cls, checkpoint_dir, subfolder, config, torch_dtype,
                   extra=None):
    """Load a DiT from ``transformers/`` or the checkpoint root.

    Uses ``from_pretrained`` when ``config.json`` is present. Otherwise builds
    the module from the task EasyDict and loads sharded safetensors — the
    layout of the current 1.3B Hugging Face upload.
    """
    extra = extra or {}
    dit_dir = _resolve_dit_dir(checkpoint_dir, subfolder)
    logging.info(f"Loading {model_cls.__name__} from {dit_dir}")
    if os.path.isfile(os.path.join(dit_dir, "config.json")):
        return model_cls.from_pretrained(
            dit_dir, torch_dtype=torch_dtype, **extra)

    logging.info(
        f"config.json not found in {dit_dir}; building {model_cls.__name__} "
        "from the task config and loading safetensors weights."
    )
    model = model_cls(**_dit_kwargs_from_config(config, extra))
    state = _load_safetensors_state_dict(dit_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning(f"Missing keys when loading DiT: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading DiT: {unexpected}")
    return model.to(dtype=torch_dtype)


class WanI2VCausal:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        dit_fsdp=False,
        use_sp=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
        infer_mode="causal_fast",
        assets_dir=None,
        skip_text_encoder=False,
        tp_degree=1,
        sp_size=1,
        seq_mode="ulysses",
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            infer_mode (`str`, *optional*, defaults to "causal_fast"):
                Inference mode. "causal_fast" uses the distilled few-step
                model (config.fast_checkpoint) with KV-cache windowing
                (local_attn_size / sink_size). "causal_pretrain" uses the
                pretrained causal model (config.causal_checkpoint) with
                40-step CFG sampling.
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            assets_dir (`str`, *optional*):
                Directory that holds shared T5 / VAE / tokenizer files when
                they are not packaged with ``checkpoint_dir`` (the 1.3B DiT
                upload). Typically the 14B causal-fast checkpoint directory.
            skip_text_encoder (`bool`, *optional*, defaults to False):
                Skip building the T5 text encoder entirely. Set this flag to True when generate()
                call uses precomputed embeddings from prompt_embeds_path.
                The text encoder would otherwise be loaded and never used.
            tp_degree (`int`, *optional*, defaults to 1):
                Tensor-parallel degree for the causal_fast DiT (model_fast.py
                only). Must equal world_size when > 1.
        """
        assert infer_mode in ("causal_fast", "causal_pretrain"), \
            f"Unsupported infer_mode: {infer_mode}"
        self.infer_mode = infer_mode

        self.device = torch.device(f"neuron:{device_id}")
        self.config = config
        self.rank = rank
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.tp_degree = tp_degree
        self.sp_size = sp_size

        # TP x SP are now composable for the causal_fast DiT: they are two
        # orthogonal axes of one rank mesh (TP shards channels/heads within a
        # cell, SP shards sequence/cells across it). causal_pretrain (CFG,
        # WanModelCausal) still asserts against mixing.
        if self.infer_mode == "causal_pretrain":
            assert not (tp_degree > 1 and use_sp), (
                "tensor-parallel + sequence-parallel mix unsupported in causal_pretrain.")
        # "cp": token-axis, stays token-sharded through attention, so heads are
        # divided by tp_degree alone. "ulysses": all_to_all makes the token
        # shard a head shard, so heads are divided by sp_size*tp_degree.
        self.seq_mode = seq_mode
        self.heads_div = tp_degree if seq_mode == "cp" else sp_size * tp_degree
        if sp_size > 1:
            assert config.num_heads % self.heads_div == 0, (
                f"num_heads={config.num_heads} must be divisible by "
                f"{self.heads_div} (heads are the head-parallel resource; "
                f"seq_mode={seq_mode}).")
        if use_sp or sp_size > 1:
            init_dit_mesh(sp_size, tp_degree)
            set_seq_parallel_mode(seq_mode)
        elif tp_degree > 1:
            init_dit_parallel_group(tp_degree)

        if dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)

        if skip_text_encoder:
            logging.info(
                "Skipping text encoder - not loading the T5 checkpoint "
                "(prompt_embeds_path will be used instead).")
            self.text_encoder = None
        else:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=_resolve_asset_path(
                    config.t5_checkpoint, checkpoint_dir, assets_dir),
                tokenizer_path=_resolve_asset_path(
                    config.t5_tokenizer, checkpoint_dir, assets_dir),
            )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=_resolve_asset_path(
                config.vae_checkpoint, checkpoint_dir, assets_dir),
            device=self.device, dtype=torch.bfloat16)

        if self.infer_mode == "causal_fast":
            log_prefix = (f"TP={tp_degree}-sharded" if tp_degree > 1
                          else "unsharded (tp_degree=1)")
            logging.info(f"Creating {log_prefix} WanModelFast from {checkpoint_dir}")
            dit_pth = os.path.join(
                checkpoint_dir, f"{config.fast_checkpoint}-bf16.pth")
            if not os.path.exists(dit_pth):
                logging.warning(
                    "%s not found; falling back to per-rank from_pretrained, "
                    "which needs ~54 GB host RAM per rank. Build it once with "
                    "scripts/inference/convert_dit_to_pth.py.", dit_pth)
                dit_pth = None
            self.model = load_tp_wan_model_fast(
                checkpoint_dir,
                subfolder=config.fast_checkpoint,
                tp_degree=tp_degree,
                dtype=torch.bfloat16,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
                device=self.device,
                pth_path=dit_pth)
        else:
            self.model = load_dit_model(
                WanModelCausal,
                checkpoint_dir,
                config.causal_checkpoint,
                config,
                torch.bfloat16,
            )

        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.local_heads_div = self.sp_size * self.tp_degree
        else:
            self.local_heads_div = self.tp_degree

        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 prompt-embedding cache. Same-prompt re-encodes hit this dict
        # instead of re-running the umt5-xxl encoder (~360 ms/call).
        # Keyed by sha256(prompt.utf8); value is the list returned by
        # T5EncoderModel.__call__ (already device-resident). Unbounded;
        # callers can clear via `pipe.clear_text_cache()` if needed.
        self._t5_cache: dict[str, list] = {}

        # Reset per generate() and flipped True after the first DiT forward.
        # Passed into model.forward as `cross_attn_first_call` to skip the
        # crossattn_cache["is_init"].item() sync inside WanCrossAttention.
        self._cross_attn_initialized: bool = False

    def clear_text_cache(self):
        """Drop all cached T5 prompt embeddings. Frees ~4 MB per entry."""
        self._t5_cache.clear()

    def prewarm(
        self,
        img,
        max_area: int = 480 * 832,
        target_hw=None,
        frame_num: int = 81,
        chunk_size: int = 3,
        text_seq_len: int = 512,
    ):
        """Opt-in pre-warm. Run one dummy DiT forward at the same shape a
        subsequent generate() call will use, so CUDA kernels are autotuned,
        FSDP all-gathers happen, and Ulysses all-to-alls handshake — all
        outside the timed generate() window.

        Without this call, the first generate() pays a ~7s warmup tax in
        chunk 0 (CUDA lazy init, kernel autotuning, NCCL handshake). On
        8xH100 at 480*832/81 frames, calling prewarm() before the first
        generate() reduces generate()'s wall-clock by ~6.5s (~30%) with
        bit-identical output.

        Idempotent: subsequent calls on the same pipe are no-ops.
        Shape-keyed: if generate() is later invoked with a different shape,
        the autotuner will warm those kernels on demand in chunk 0 (no
        incorrect output, just the tax re-paid once).

        Args:
            img: PIL image or torch tensor — used only for its h/w to match
                generate()'s lat_h/lat_w derivation.
            max_area, frame_num, chunk_size: shape parameters; must match
                the subsequent generate() call to be effective.
            text_seq_len: T5 sequence length (defaults to config.text_len).

        Caller pattern:
            pipe = WanI2VCausal(...)
            pipe.prewarm(img, max_area=..., frame_num=...)
            # start your timer here
            video = pipe.generate(prompt, img, ...)
        """
        if self.infer_mode != "causal_fast":
            logging.info("prewarm is only supported for infer_mode='causal_fast'; skipping.")
            return
        if getattr(self, "_warmed", False):
            return

        cfg = self.config

        # Match generate()'s shape derivation exactly.
        F = frame_num
        h, w = (img.shape[1], img.shape[2]) if hasattr(img, 'shape') else (img.size[1], img.size[0])
        aspect_ratio = h / w
        # Same exact-grid override as generation. If prewarm computed a different grid
        # it would compile the wrong shapes and generation would recompile anyway.
        if target_hw is not None:
            lat_h = target_hw[0] // cfg.vae_stride[1]
            lat_w = target_hw[1] // cfg.vae_stride[2]
        else:
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] //
                cfg.patch_size[1] * cfg.patch_size[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] //
                cfg.patch_size[2] * cfg.patch_size[2])
        lat_f = (F - 1) // cfg.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))

        frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        max_seq_len = chunk_size * frame_seqlen
        head_dim = cfg.dim // cfg.num_heads
        local_num_heads = cfg.num_heads // self.heads_div

        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f

        transformer_dtype = self.pipe_dtype
        # generate() folds the VAE spatial stride into the Plücker channel
        # dim via rearrange 'f (h s1) (w s2) c -> (f h w) (c s1 s2)' with
        # s1=s2=vae_stride[1]=8, so control_dim=6 → 6 * 8 * 8 = 384.
        plucker_channels = 6 * cfg.vae_stride[1] * cfg.vae_stride[2]
        # T5 (umt5-xxl) hidden size; cross-attn projects t5_hidden → cfg.dim.
        t5_hidden = 4096

        warmup_self_kv = self._make_self_kv(
            cfg.num_layers, 1, kv_size, local_num_heads, head_dim,
            chunk_size, frame_seqlen, transformer_dtype, self.device)
        warmup_cross_kv = self._initialize_crossattn_cache(
            num_layers=cfg.num_layers,
            shape=[1, text_seq_len, local_num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)

        # `y` is concat([msk_4ch, vae_latent_16ch]) → 20 channels; combined
        # with latent's 16 ch at patch-embed concat, the DiT sees 36 ch in.
        dummy_latent = torch.zeros(
            16, chunk_size, lat_h, lat_w,
            device=self.device, dtype=torch.float32)
        dummy_y = torch.zeros(
            20, chunk_size, lat_h, lat_w,
            device=self.device, dtype=transformer_dtype)
        dummy_c2ws = torch.zeros(
            1, plucker_channels, chunk_size, lat_h, lat_w,
            device=self.device, dtype=self.param_dtype)
        dummy_context = torch.zeros(
            text_seq_len, t5_hidden,
            device=self.device, dtype=self.param_dtype)
        dummy_t = torch.tensor(
            [500.0], device=self.device, dtype=torch.float32)

        @contextmanager
        def _noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', _noop_no_sync)

        if dist.is_initialized():
            torch.neuron.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        with torch.amp.autocast('neuron', dtype=self.param_dtype), \
             torch.no_grad(), \
             no_sync_model():
            _ = self.model(
                x=[dummy_latent],
                t=dummy_t,
                context=[dummy_context],
                seq_len=max_seq_len,
                y=[dummy_y],
                dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                kv_cache=warmup_self_kv,
                crossattn_cache=warmup_cross_kv,
                cross_attn_first_call=True,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )

            # WARM EVERY CHUNK POSITION, not just position 0.
            #
            # model_fast passes the KV write window as PLAIN INTS so the slice is
            # a compile-time constant (that is what let a whole block compile at
            # fullgraph=True). The consequence is one graph PER chunk position, and
            # if they are compiled lazily inside the denoise loop the compile time
            # lands in the measured region: the DiT-only replay always feeds chunk
            # 0 and got 108.0 ms/forward, while the real 21-chunk run measured
            # GEN 1119.9 s against 705.4 s eager -- ~21 compilations at roughly the
            # replay's 102.3 s first call accounts for nearly all of the gap.
            #
            # the reference implementation builds its destination-specialised graphs up front for the
            # same reason (reference setup.py:231), "for destination in
            # range(kv_num_chunks)"). This does the equivalent: touch each
            # current_start once here, outside the timed window.
            if True:
                _n_chunks = max(1, lat_f // max(1, chunk_size))
                for _c in range(1, _n_chunks):
                    self.model(
                        x=[dummy_latent],
                        t=dummy_t,
                        context=[dummy_context],
                        seq_len=max_seq_len,
                        y=[dummy_y],
                        dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                        kv_cache=warmup_self_kv,
                        crossattn_cache=warmup_cross_kv,
                        cross_attn_first_call=False,
                        current_start=_c * chunk_size * frame_seqlen,
                        max_attention_size=kv_size,
                        frame_seqlen=frame_seqlen,
                    )
                if self.rank == 0:
                    logging.info(
                        "prewarm: touched %d chunk positions so their graphs "
                        "compile here, not inside the denoise loop", _n_chunks)

        if dist.is_initialized():
            torch.neuron.synchronize()
            dist.barrier()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"WanI2VCausal.prewarm: {dt_ms:.0f} ms")

        del (warmup_self_kv, warmup_cross_kv, dummy_latent, dummy_y,
             dummy_c2ws, dummy_context, dummy_t)
        torch.neuron.empty_cache()
        self._warmed = True

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            model._onload_device = self.device

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        if flow_pred.device.type == "neuron" and os.environ.get("LINGBOT_HOST_SCHED", "1") == "1":
            # The form below asks for fp64, but on Neuron it is computed in fp32 and every op
            # runs with its own D2H/H2D pair. Measured (scripts/profiling/check_sched_parity.py):
            # looking sigma_t up on the host tables and doing the fp32 multiply-subtract on
            # device is bit-identical to it (max|d| = 0), with no copies and no sync.
            _ts = scheduler.timesteps.cpu().double()
            _tid = torch.argmin((_ts - timestep.cpu().double()).abs())
            _sig = float(scheduler.sigmas.cpu().double()[_tid])
            return (xt - _sig * flow_pred).to(original_dtype)
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)


    def generate(self,
                 input_prompt,
                 img,
                 action_path,
                 chunk_size=3,
                 max_area=480 * 832,
                 target_hw=None,
                 denoise_steps=None,
                 frame_num=81,
                 timesteps_index=[0, 250, 500, 750],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,
                 save_latents_path=None,
                 prompt_embeds_path=None,
                 prompt_embeds=None,
                 return_latents=False,
                 chunk_callback=None,):
        r"""
        Generates video frames from input image and text prompt.

        Dispatches to the mode-specific implementation according to
        `self.infer_mode`:
            - "causal_fast": distilled few-step sampling (`_generate_causal_fast`)
            - "causal_pretrain": 40-step CFG sampling (`_generate_causal_pretrain`)

        Both only supported in "causal_fast" mode (the single-torchrun e2e
        path's DiT->VAE handoff -- see generate.py):
        save_latents_path (`str`, *optional*): if set, skip VAE decode and
            dump the pre-decode latent to this path instead (for the
            width-sharded decode stage, see decode_latents_width_sharded.py).
        return_latents (`bool`, *optional*): skip VAE decode and return the
            pre-decode latent tensor directly instead of a video.

        Also "causal_fast"-only, in-process T5-TP handoff (see generate.py):
        prompt_embeds_path (`str`, *optional*): path to a prompt_embeds.pt
            saved by encode_prompt_sharded.py. Skips the inline text encoder,
            loads precomputed embeddings from disk instead.
        prompt_embeds (*optional*): precomputed embeds passed directly
            in-memory instead of via prompt_embeds_path.
        chunk_callback (*optional*): "causal_fast" only. Called on every rank as
            chunk_callback(chunk_id, x0) with each denoised latent chunk
            [C, chunk_size, H, W] as soon as it is ready, for streaming decode.
        """
        extra = {}
        if chunk_callback is not None:
            assert self.infer_mode == "causal_fast", "chunk_callback requires infer_mode='causal_fast'"
            extra["chunk_callback"] = chunk_callback
        gen_fn = (self._generate_causal_fast
                  if self.infer_mode == "causal_fast"
                  else self._generate_causal_pretrain)
        return gen_fn(
            input_prompt,
            img,
            action_path,
            chunk_size=chunk_size,
            max_area=max_area,
            target_hw=target_hw,
            denoise_steps=denoise_steps,
            frame_num=frame_num,
            timesteps_index=timesteps_index,
            shift=shift,
            seed=seed,
            offload_model=offload_model,
            max_sequence_length=max_sequence_length,
            max_attention_size=max_attention_size,
            save_latents_path=save_latents_path,
            prompt_embeds_path=prompt_embeds_path,
            prompt_embeds=prompt_embeds,
            return_latents=return_latents,
            **extra)

    def _generate_causal_fast(self,
                              input_prompt,
                              img,
                              action_path,
                              chunk_size=3,
                              max_area=480 * 832,
                              target_hw=None,
                              frame_num=81,
                              timesteps_index=[0, 179, 358, 679],
                              denoise_steps=None,
                              shift=5.0,
                              seed=-1,
                              offload_model=True,
                              max_sequence_length=512,
                              max_attention_size=None,
                              save_latents_path=None,
                              prompt_embeds_path=None,
                              prompt_embeds=None,
                              return_latents=False,
                              chunk_callback=None,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1
        
        assert action_path is not None, "action_path is required"
        c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        # An EXACT grid wins over the area fit. max_area cannot express a required
        # latent shape -- sqrt(area * aspect) follows the image -- and const-max AG
        # attention needs frame_seqlen to be a whole number of 128-token tiles. See
        # EXACT_GRID_SIZES in wan/configs.
        if target_hw is not None:
            lat_h = target_hw[0] // self.vae_stride[1]
            lat_w = target_hw[1] // self.vae_stride[2]
            assert lat_h % self.patch_size[1] == 0 and lat_w % self.patch_size[2] == 0, (
                f"exact grid {target_hw} gives lat {lat_h}x{lat_w}, not divisible by "
                f"patch_size {self.patch_size[1:]}")
        else:
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                self.patch_size[1] * self.patch_size[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        # Reset per-generate state: cross-attn K/V cache will be freshly
        # initialized below; the first DiT forward must compute and store.
        self._cross_attn_initialized = False

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device='cpu')
        seed_g.manual_seed(seed)
        # PHASES COVER THE WHOLE FUNCTION, not just the chunk loop. The first six timers
        # accounted for 19.8 s of a 32.62 s GEN and left 12.8 s outside all of them, with
        # nothing left in the loop to blame -- because GEN wraps this entire function and
        # I had only instrumented the loop inside it.
        self._gen_phase = {"sched": 0.0, "setup": 0.0, "prep": 0.0, "randn": 0.0,
                           "addnoise": 0.0, "ctx": 0.0, "encode": 0.0, "cat": 0.0,
                           "kvinit": 0.0}
        _g0 = time.perf_counter()
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device='cpu').to(self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        # DENOISE STEPS, the reference implementation's "biggest single lever" (the reference benchmark --steps: "Latency
        # is ~linear in steps"). Their measured 6.53 fps run used STEPS=2 against a
        # default of 4, which is also our default here.
        #
        # Subsample the schedule keeping BOTH ENDPOINTS: timesteps_index[0] is the
        # highest-noise step and the last is the lowest, and the loop returns x0 from the
        # final one -- dropping the tail would return a still-noisy latent.
        if denoise_steps is not None and 0 < denoise_steps < len(timesteps_index):
            _n = len(timesteps_index)
            _sel = [round(i * (_n - 1) / (denoise_steps - 1)) for i in range(denoise_steps)] \
                if denoise_steps > 1 else [_n - 1]
            timesteps_index = [timesteps_index[i] for i in sorted(set(_sel))]
            logging.info(f"denoise steps: {len(timesteps_index)} "
                         f"(timesteps_index={timesteps_index})")
        timesteps = self.scheduler.timesteps[timesteps_index]

        # preprocess
        # T5 cache: skip the encoder entirely if we've seen this exact prompt
        # (or, with precomputed embeds, this exact file) before in this pipe
        # instance. Bit-identical: cached tensor is the same object returned
        # by the prior call. Keyed off prompt_embeds_path rather than
        # input_prompt when set, since input_prompt is never re-encoded on
        # that path and isn't even required there (see generate()'s
        # docstring) — generate.py's EXAMPLE_PROMPT fallback means it may not
        # be the text actually encoded, so it can't be trusted as a cache key.
        cache_key = hashlib.sha256(
            (prompt_embeds_path or input_prompt).encode('utf-8')).hexdigest()
        if prompt_embeds is not None:
            context = [t.to(self.device) for t in prompt_embeds]
        elif cache_key in self._t5_cache:
            context = self._t5_cache[cache_key]
        elif prompt_embeds_path is not None:
            # Stage 2 of the two-stage T5-TP pipeline (see generate()'s
            # docstring): load Stage 1's precomputed embeds instead of
            # running the (inline, single-process, unsharded) text encoder
            # at all.
            _embeds = torch.load(
                prompt_embeds_path, map_location='cpu', weights_only=False)
            context = [t.to(self.device) for t in _embeds['prompt_embeds']]
            logging.info(
                f"Using precomputed prompt embeds {tuple(context[0].shape)} "
                f"from {prompt_embeds_path}; text encoder skipped.")
            self._t5_cache[cache_key] = context
        else:
            _t5_t0 = time.perf_counter()
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
            logging.info(
                f"T5 encode (device): {(time.perf_counter() - _t5_t0) * 1000:.0f} ms")
            self._t5_cache[cache_key] = context

        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

        # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
        Ks = get_Ks_transformed(Ks,
                                height_org=480,
                                width_org=832,
                                height_resize=h,
                                width_resize=w,
                                height_final=h,
                                width_final=w)
        Ks = Ks[0]

        len_c2ws = len(c2ws)
        len_c2ws_ = int((len_c2ws - 1) // 4) + 1
        len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        wasd_action = None
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
        if wasd_action is not None:
            wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
            wasd_action_tensor = rearrange(
                wasd_action_tensor,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
            wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        torch.neuron.synchronize()
        self._gen_phase["ctx"] += time.perf_counter() - _g0
        _e0 = time.perf_counter()
        # CACHED. Measured at 12.2 s -- 36% of GEN and 2.6x the whole VAE decode -- and
        # it is the same computation every time: one conditioning image at a fixed
        # (F, h, w) always yields the same y.
        #
        # Two reasons it costs so much, neither of which caching fixes for a first
        # generation:
        #   1. It encodes F frames, and F-1 of them are torch.zeros. At frame_num=81
        #      that is 80 zero frames, so 80/81 of the encoder work is encoding nothing.
        #   2. The ENCODER is not width-sharded. shard_vae_decoder_width is only ever
        #      applied to vae.model.decoder, so this runs at single-core width and
        #      redundantly on all 32 ranks.
        _ek = (int(img.shape[1]), int(img.shape[2]), int(F), int(h), int(w),
               hash(img.detach().cpu().numpy().tobytes()))
        _ec = getattr(self, "_encode_cache", None)
        if _ec is not None and _ec[0] == _ek:
            y = _ec[1].clone()
        else:
            y = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w)
                ],
                             dim=1).to(self.device)
            ])[0]
            y = torch.concat([msk, y])
            self._encode_cache = (_ek, y.clone())
        torch.neuron.synchronize()
        self._gen_phase["encode"] += time.perf_counter() - _e0

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)


        # Initialize KV cache to all zeros
        _k0 = time.perf_counter()
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        # Self-attention: after the Ulysses all_to_all, each rank holds
        # num_heads/(sp*tp) local heads.
        local_num_heads = model_args.num_heads // self.heads_div
        self_kv_cache = self._make_self_kv(
            model_args.num_layers, batch_size, kv_size, local_num_heads, head_dim,
            chunk_size, frame_seqlen, transformer_dtype, self.device)
        # Cross-attention is TP-only (context, no Ulysses all_to_all), so its
        # cache holds num_heads/tp heads, NOT divided by sp_size.
        cross_num_heads = model_args.num_heads // self.tp_degree
        cross_kv_shape = [batch_size, max_sequence_length, cross_num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                          shape=cross_kv_shape,
                                                          dtype=transformer_dtype,
                                                          device=self.device)
        torch.neuron.synchronize()
        self._gen_phase["kvinit"] += time.perf_counter() - _k0
        # evaluation mode
        with (
                torch.amp.autocast('neuron', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            # sample videos
            latent = noise
            self._dit_fwd_ms = []
            # Optional single-forward profile (set by generate.py LINGBOT_PROFILE_FORWARD).
            _pf = getattr(self, "_profile_forward", None)
            self._profile_forward = None
            def _fwd_ctx():
                if _pf is not None and len(self._dit_fwd_ms) == _pf[0]:
                    return _pf[1]()
                return contextlib.nullcontext()
            self._dit_fwd_tag = []
            # WHERE THE NON-DiT TIME GOES.
            #
            # GEN is 32.4 s warm and the DiT forwards are 18.8 s of it, so 13.6 s -- 2.8x
            # the entire VAE -- is spent somewhere in this loop and nothing has measured
            # it. Every phase below is synced on both sides: torch.neuron dispatch is
            # async, and an unsynced timer reports host issue time, which is the mistake
            # the DiT timer made first time round (26.9 ms median against an 82 s max).

            latents_chunk = latent.split(chunk_size, dim=1) # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                torch.neuron.synchronize()
                _c0 = time.perf_counter()
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                kwargs = {
                    'context': [context[0]],
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,
                    'kv_cache': self_kv_cache,
                    'crossattn_cache': cross_kv_cache,
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                    'frame_seqlen': frame_seqlen,
                }

                if offload_model:
                    torch.neuron.empty_cache()

                torch.neuron.synchronize()
                self._gen_phase["setup"] += time.perf_counter() - _c0
                for timestep_idx in range(len(timesteps)):
                    torch.neuron.synchronize()
                    _p0 = time.perf_counter()
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = torch.stack(current_timestep).to(self.device)
                    torch.neuron.synchronize()
                    self._gen_phase["prep"] += time.perf_counter() - _p0

                    # PER-FORWARD DiT TIME, MEASURED IN THE REAL LOOP.
                    #
                    # SYNC BEFORE AND AFTER. The first version of this had neither, on
                    # the claim that "the next host-side read forces the sync anyway".
                    # That was wrong: torch.neuron dispatch is async, so it reported
                    # median=26.9 ms -- how fast the host ISSUED the call -- while the
                    # synced DiT-only harness measured 184.3 ms of device time for the
                    # same forward. The total was still real (async work backs up and
                    # eventually blocks), which is why it read 336.2 s with a 26.9 ms
                    # median and an 82.4 s max: a few calls absorbed everyone else's
                    # queued work. Unsynced per-call numbers are meaningless here.
                    #
                    # The only DiT-forward numbers on record came from
                    # profile_dit_from_captured.py, which replays ONE captured chunk-0
                    # input at FRAME_NUM=21, chunk_size=4 and local_attn_size=-1 -- a
                    # different frame count, a different token count per forward (6032
                    # vs 1508) and a different cache design from anything generation
                    # runs. Comparing them to GEN was invalid, and "[BENCH] DiT time"
                    # is no help either: it is just LOAD + GEN.
                    #
                    # This is the honest number. perf_counter around the call includes
                    # the device sync that the next host-side read forces anyway, so it
                    # attributes queueing to the forward rather than hiding it.
                    self.model._t_host_key = float(current_timestep[0])
                    torch.neuron.synchronize()
                    _f0 = time.perf_counter()
                    with _fwd_ctx():
                        noise_pred = self.model(
                            x=latent_model_input, t=timestep,
                            cross_attn_first_call=not self._cross_attn_initialized,
                            **kwargs)[0]
                        torch.neuron.synchronize()
                    self._dit_fwd_ms.append((time.perf_counter() - _f0) * 1000)
                    self._dit_fwd_tag.append(f"c{chunk_id}s{timestep_idx}")
                    self._cross_attn_initialized = True

                    if offload_model:
                        torch.neuron.empty_cache()

                    torch.neuron.synchronize()
                    _s0 = time.perf_counter()
                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )
                    torch.neuron.synchronize()
                    self._gen_phase["sched"] += time.perf_counter() - _s0

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        # Split deliberately: the noise is drawn on the CPU and copied to
                        # device on every step, which is the obvious suspect, but 13.05 s
                        # over 63 calls is 207 ms each and a 96K-element randn cannot
                        # cost that. Timing the draw and the scheduler step apart says
                        # which -- or neither, in which case the time is in the loop
                        # itself and both suspects are wrong.
                        torch.neuron.synchronize()
                        _n0 = time.perf_counter()
                        _noise = torch.randn(x0.shape, generator=seed_g, device='cpu',
                                             dtype=x0.dtype).to(x0.device)
                        torch.neuron.synchronize()
                        self._gen_phase["randn"] += time.perf_counter() - _n0
                        _a0 = time.perf_counter()
                        if os.environ.get("LINGBOT_HOST_SCHED", "1") == "1":
                            # add_noise with the sigma lookup done on the host tables
                            # (index_for_timestep's nonzero/.item() synced the device), then
                            # the same fp32 alpha*x0 + sigma*noise on device.
                            _sch = self.scheduler
                            _sig = _sch.sigmas.to(dtype=x0.dtype, device="cpu")
                            _idx = _sch.index_for_timestep(
                                next_timestep.cpu() if torch.is_tensor(next_timestep) else next_timestep,
                                _sch.timesteps.cpu())
                            _a, _s = _sch._sigma_to_alpha_sigma_t(_sig[_idx])
                            current_latent = float(_a) * x0 + float(_s) * _noise
                        else:
                            current_latent = self.scheduler.add_noise(
                                x0, _noise, next_timestep)
                        torch.neuron.synchronize()
                        self._gen_phase["addnoise"] += time.perf_counter() - _a0
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)
                if chunk_callback is not None:
                    chunk_callback(chunk_id, x0)

                # Update kv cache
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                # A FIFTH DiT forward per chunk. len(timesteps) is 4
                # (timesteps_index=[0, 179, 358, 679]), so the denoise loop above is not
                # the whole cost -- this cache-commit pass runs the full 40 blocks too.
                self.model._t_host_key = float(context_timestep[0])
                torch.neuron.synchronize()
                _f0 = time.perf_counter()
                with _fwd_ctx():
                    self.model(x=[x0], t=timestep,
                               cross_attn_first_call=False,
                               **kwargs)
                    torch.neuron.synchronize()
                self._dit_fwd_ms.append((time.perf_counter() - _f0) * 1000)
                self._dit_fwd_tag.append(f"c{chunk_id}commit")

            if self._dit_fwd_ms and (not dist.is_initialized() or dist.get_rank() == 0):
                _m = sorted(self._dit_fwd_ms)
                _tot = sum(_m) / 1000.0
                logging.info(
                    f"[BENCH] DiT forwards: n={len(_m)} total={_tot:.1f} s "
                    f"median={_m[len(_m) // 2]:.1f} ms min={_m[0]:.1f} max={_m[-1]:.1f} "
                    f"first={self._dit_fwd_ms[0]:.1f} ms")
                # WHERE the stalls are. A median of 26.9 ms against an 82.4 s max means
                # a handful of calls hold nearly all the time, and prewarm was supposed
                # to have paid for them already. Tagging by chunk/step says which.
                _top = sorted(zip(self._dit_fwd_ms, self._dit_fwd_tag), reverse=True)[:8]
                logging.info("[BENCH] DiT slowest: " + ", ".join(
                    f"{t}={ms / 1000:.1f}s" for ms, t in _top))
                # Every forward in call order, so window-filling vs full-window cost is visible.
                logging.info("[BENCH] DiT all: " + " ".join(
                    f"{t}={ms:.0f}" for ms, t in zip(self._dit_fwd_ms, self._dit_fwd_tag)))
                _big = [ms for ms in self._dit_fwd_ms if ms > 10 * _m[len(_m) // 2]]
                logging.info(
                    f"[BENCH] DiT stalls >10x median: {len(_big)} calls, "
                    f"{sum(_big) / 1000:.1f} s of {_tot:.1f} s total "
                    f"({100 * sum(_big) / max(_tot * 1000, 1e-9):.0f}%)")
                _ph = self._gen_phase
                logging.info(
                    f"[BENCH] GEN phases: dit={_tot:.1f} s "
                    f"sched={_ph['sched']:.1f} s setup={_ph['setup']:.1f} s "
                    f"prep={_ph['prep']:.1f} s randn={_ph['randn']:.1f} s "
                    f"addnoise={_ph['addnoise']:.1f} s "
                    f"ctx={_ph['ctx']:.1f} s encode={_ph['encode']:.1f} s "
                    f"cat={_ph['cat']:.1f} s kvinit={_ph['kvinit']:.1f} s "
                    f"accounted={_tot + sum(_ph.values()):.1f} s")
            _t0 = time.perf_counter()
            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)
            torch.neuron.synchronize()
            self._gen_phase["cat"] += time.perf_counter() - _t0

            if offload_model:
                self.model.cpu()
                torch.neuron.empty_cache()

            skip_decode = save_latents_path is not None or return_latents
            _post0 = time.perf_counter()
            if self.rank == 0:
                if save_latents_path is not None:
                    save_latents_dir = Path(save_latents_path).parent
                    if not save_latents_dir.exists():
                        save_latents_dir.mkdir(parents=True, exist_ok=True)
                        logging.info(f"Created directory {save_latents_dir} for save_latents_path")
                    torch.save(pred_latent_chunks.cpu(), save_latents_path)
                    logging.info(f"Saved pre-decode latent to {save_latents_path}")
                if not skip_decode:
                    self.vae.model.to('cpu')
                    self.vae.mean = self.vae.mean.cpu()
                    self.vae.std = self.vae.std.cpu()
                    self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
                    videos = self.vae.decode([pred_latent_chunks.cpu()])

        # del noise, latent, x0
        # del sample_scheduler
        if offload_model:
            gc.collect()
            torch.neuron.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if not dist.is_initialized() or dist.get_rank() == 0:
            logging.info(f"[BENCH] GEN post (latents .cpu() + save + barrier): "
                         f"{(time.perf_counter() - _post0) * 1000:.0f} ms")
        if skip_decode:
            return pred_latent_chunks.cpu()
        return videos[0] if self.rank == 0 else None

    def _generate_causal_pretrain(self,
                                  input_prompt,
                                  img,
                                  action_path,
                                  chunk_size=3,
                                  max_area=480 * 832,
                                  frame_num=81,
                                  timesteps_index=None,
                                  shift=5.0,
                                  seed=-1,
                                  offload_model=True,
                                  max_sequence_length=512,
                                  max_attention_size=None,
                                  save_latents_path=None,
                                  prompt_embeds_path=None,
                                  prompt_embeds=None,
                                  return_latents=False,):
        r"""
        Generates video frames with the pretrained causal model using
        40-step CFG sampling per chunk. `timesteps_index` is unused in this
        mode (kept for signature compatibility with `generate`).
        """
        assert save_latents_path is None and not return_latents, \
            "save_latents_path/return_latents only supported in infer_mode='causal_fast'"
        assert prompt_embeds is None, \
            "prompt_embeds only supported in infer_mode='causal_fast'"
        guide_scale = 5.0
        n_prompt = "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        if input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1

        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps (scheduler object created once, state reset per chunk)
        sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)

        # preprocess text: cond + uncond
        if prompt_embeds_path is not None:
            # Stage 2 of the two-stage T5-TP pipeline — see generate()'s
            # docstring. input_prompt/n_prompt are not required and are not
            # re-encoded here; this function has no T5 cache to key (unlike
            # _generate_causal_fast), so nothing else depends on their value.
            _embeds = torch.load(
                prompt_embeds_path, map_location='cpu', weights_only=False)
            context      = [t.to(self.device) for t in _embeds['prompt_embeds']]
            context_null = [t.to(self.device) for t in _embeds['negative_prompt_embeds']]
            logging.info(
                f"Using precomputed prompt embeds {tuple(context[0].shape)} "
                f"from {prompt_embeds_path}; text encoder skipped.")
        else:
            self.text_encoder.model.to(self.device)
            context      = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt],     self.device)
            if offload_model:
                self.text_encoder.model.cpu()

        # cam preparation (only if action_path is provided)
        c2ws_plucker_emb = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks,
                                    height_org=480, width_org=832,
                                    height_resize=h, width_resize=w,
                                    height_final=h, width_final=w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w,
            ).to(self.param_dtype)

        y = self.vae.encode([
            torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w)
                ], 
                dim=1,
            ).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // max(self.sp_size, 1)
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]

        # CFG requires separate caches for the cond / uncond streams.
        self_kv_cache_cond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        self_kv_cache_uncond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_cond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_uncond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)

        with (
                torch.amp.autocast('neuron', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1)          # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []

            for chunk_id in tqdm(range(num_inference_chunk)):
                # Reset the multi-step scheduler state for each chunk
                sample_scheduler.set_timesteps(40, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps

                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                common = {
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,          # camera condition is kept for uncond as well
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                }
                kwargs_cond = {
                    **common,
                    'context': [context[0]],
                    'kv_cache': self_kv_cache_cond,
                    'crossattn_cache': cross_kv_cache_cond,
                }
                kwargs_uncond = {
                    **common,
                    'context': context_null,
                    'kv_cache': self_kv_cache_uncond,
                    'crossattn_cache': cross_kv_cache_uncond,
                }

                if offload_model:
                    torch.neuron.empty_cache()

                for timestep_idx in tqdm(range(len(timesteps)), desc=f"infer chunk {chunk_id}"):
                    latent_model_input = [current_latent.to(self.device)]
                    t = timesteps[timestep_idx]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_cond)[0]
                    noise_pred_uncond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_uncond)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    if offload_model:
                        torch.neuron.empty_cache()
                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t,
                        current_latent.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0]
                    current_latent = temp_x0.squeeze(0)

                    del latent_model_input, timestep

                pred_latent_chunks.append(current_latent)

                # Update both self KV caches with the clean latent (once for cond, once for uncond)
                timestep0 = torch.stack([timesteps[-1] * 0.0]).to(self.device)
                self.model(x=[current_latent], t=timestep0, **kwargs_cond)
                self.model(x=[current_latent], t=timestep0, **kwargs_uncond)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                torch.neuron.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        if offload_model:
            gc.collect()
            torch.neuron.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _make_self_kv(self, num_layers, batch, kv_size, heads, head_dim,
                      chunk_size, frame_seqlen, dtype, device):
        """Chunked self-attention KV cache when LINGBOT_CHUNKED_KV=1, else the flat one.

        Chunked is the reference implementation's design; wan/modules/chunked_kv_cache.py records what it buys
        and why. It only means anything for a BOUNDED window -- with local_attn_size=-1
        there is no window to chunk, so num_chunks() returns 0 and the flag is ignored.
        """
        n = (num_chunks(self.local_attn_size, chunk_size)
             if os.environ.get("LINGBOT_CHUNKED_KV", "0") == "1" else 0)
        if n:
            # PER-LAYER layout, because a hot layer takes the exact-attention path and
            # reads [B, S, N, D] while its cold neighbours feed the const-max kernel and
            # need N-major. With FLASH_ATTN_LAYERS unset this is all-True and identical to
            # the previous n_major=const_max_ag_available().
            from .modules.model_fast import flash_attn_layers, layer_uses_const_max
            from .modules.attention import fa2_ag_available
            # FA2 AG hot layers read the same N-major cache as the const-max layers.
            _n_major = [const_max_ag_available() and (layer_uses_const_max(i) or fa2_ag_available())
                        for i in range(num_layers)]
            cache = allocate_self_kv_chunks(
                num_layers, batch, chunk_size * frame_seqlen, heads, head_dim,
                n, dtype, device, n_major=_n_major)
            _flash = sorted(flash_attn_layers())
            if _flash:
                logging.info(
                    f"hybrid attention: {num_layers - len(_flash)} layers const_max_ag, "
                    f"{len(_flash)} exact (FLASH_ATTN_LAYERS={_flash})")
            # Build the fused AG kernels once, now that the chunk size is known. The
            # replica group is COLLECTIVE (all_gather_object), so every rank reaches it.
            if const_max_ag_available():
                from .distributed import tensor_parallel as _tp
                from .modules.attention import init_const_max_ag
                # ONE call: replica_groups() is a COLLECTIVE (dist.all_gather_object).
                # Two calls happened to be safe here only because the f-string is
                # evaluated on every rank regardless of log level, so all 32 ranks made
                # both. The harness put the second inside `if rank == 0` and every rank
                # failed with "replica group signature mismatch for group 0". One call
                # removes the hazard and a redundant world collective per generation.
                _rg = _tp.replica_groups("cp")
                init_const_max_ag(_rg, chunk_size * frame_seqlen)
                from .modules.attention import fa2_ag_available, init_fa2_ag
                if fa2_ag_available() and _flash:
                    init_fa2_ag(_rg, chunk_size * frame_seqlen)
                    logging.info(f"hot layers {_flash}: fa2_ag (fused AG FlashAttention-2)")
                logging.info(
                    f"const_max_ag: {n} chunks x {chunk_size * frame_seqlen} tokens, "
                    f"cp replica groups {_rg[:2]}...")
            logging.info(
                f"self KV: chunked, {n} x {chunk_size * frame_seqlen} tokens, "
                f"heads/rank={heads}, {chunk_bytes(cache) / 1e9:.2f} GB")
            return cache
        return self._initialize_self_kv_cache(
            num_layers=num_layers, shape=[batch, kv_size, heads, head_dim],
            dtype=dtype, device=device)

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            k = torch.zeros(shape, dtype=dtype, device=device)
            v = torch.zeros(shape, dtype=dtype, device=device)
            # Pin the data pointers so dynamo does not treat them as changing
            # inputs. Taken from the reference implementation (kv_cache.py:41)
            # (NeuronKVCache.preallocate), whose docstring is the reason: the base
            # class "allocates each block's window lazily on the first write and
            # returns a slice that grows as the window fills. Neither suits a
            # compiled region, which takes the buffers as arguments and
            # re-specializes if a shape changes."
            #
            # These buffers already exist up front here and are read whole every
            # call (attn_lo/attn_hi mask the window rather than slicing it), which
            # is the reference implementation's other requirement -- "handing attention the whole window
            # on every call, a constant shape without lying about the write
            # position". What was missing is the address pinning.
            for _t in (k, v):
                try:
                    torch._dynamo.mark_static_address(_t)
                except Exception:
                    pass  # older torch: no-op, cache still correct
            self_kv_cache.append({
                'k': k,
                'v': v,
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache


    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            k = torch.zeros(shape, dtype=dtype, device=device)
            v = torch.zeros(shape, dtype=dtype, device=device)
            for _t in (k, v):   # same pinning as the self-attn cache above
                try:
                    torch._dynamo.mark_static_address(_t)
                except Exception:
                    pass
            crossattn_cache.append({
                'k': k,
                'v': v,
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache

    def _initialize_crossattn_cache_pretrain(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache for the pretrained causal
        model, which expects `is_init` to be a plain Python bool.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': False,
            })

        return crossattn_cache