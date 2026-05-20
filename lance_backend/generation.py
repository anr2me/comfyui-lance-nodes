# lance_backend/generation.py
# Generation helpers for the ComfyUI Lance custom nodes.
#
# Lance source lives in the sibling Lance/ git submodule.
# These functions wrap Lance's internal validation pipeline so ComfyUI
# nodes can call them without managing distributed state or DataLoaders.

from __future__ import annotations

import gc
import json
import os
import tempfile
from typing import List

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import set_seed


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _video_tensor_to_frames(video: torch.Tensor) -> List[torch.Tensor]:
    """
    (T,H,W,C) or (C,T,H,W) float tensor in [0,1]
    → list of (1,H,W,C) ComfyUI IMAGE tensors.
    """
    if video.dim() == 4:
        frames = video if video.shape[-1] == 3 else video.permute(1, 2, 3, 0)
    else:
        raise ValueError(f"Unexpected video tensor shape: {video.shape}")
    return [frames[i].unsqueeze(0).float().clamp(0, 1) for i in range(frames.shape[0])]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_t2i(
    *, model, vae_model, tokenizer, new_token_ids, image_token_id,
    prompt, width, height, num_steps, cfg_scale, timestep_shift, seed,
    device, dtype,
) -> torch.Tensor:
    """Text-to-Image → (1,H,W,3) ComfyUI IMAGE tensor."""
    set_seed(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = _write_json(tmpdir, {
            "sample_0": {
                "interleave_array":    [prompt],
                "element_dtype_array": ["text"],
                "task":                "t2i",
            }
        })
        frames = _run_generation(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            new_token_ids=new_token_ids, image_token_id=image_token_id,
            json_path=json_path, task="t2i",
            width=width, height=height, num_frames=1,
            num_steps=num_steps, cfg_scale=cfg_scale,
            timestep_shift=timestep_shift, seed=seed,
            device=device, dtype=dtype,
        )
    return frames[0] if frames else torch.zeros(1, height, width, 3)


def run_image_edit(
    *, model, vae_model, tokenizer, new_token_ids, image_token_id,
    source_image: PILImage.Image, instruction,
    num_steps, cfg_scale, timestep_shift, seed, device, dtype,
) -> torch.Tensor:
    """Image editing → (1,H,W,3) ComfyUI IMAGE tensor."""
    set_seed(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "source.png")
        source_image.save(img_path)
        json_path = _write_json(tmpdir, {
            "sample_0": {
                "interleave_array":    [img_path, ["", instruction]],
                "element_dtype_array": ["image", "text"],
                "task":                "image_edit",
            }
        })
        frames = _run_generation(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            new_token_ids=new_token_ids, image_token_id=image_token_id,
            json_path=json_path, task="image_edit",
            width=source_image.width, height=source_image.height, num_frames=1,
            num_steps=num_steps, cfg_scale=cfg_scale,
            timestep_shift=timestep_shift, seed=seed,
            device=device, dtype=dtype,
        )
    return frames[0] if frames else torch.zeros(1, source_image.height, source_image.width, 3)


def run_t2v(
    *, model, vae_model, tokenizer, new_token_ids, image_token_id,
    prompt, width, height, num_frames, num_steps, cfg_scale, timestep_shift,
    seed, device, dtype,
) -> List[torch.Tensor]:
    """Text-to-Video → list of (1,H,W,3) ComfyUI IMAGE tensors."""
    set_seed(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = _write_json(tmpdir, {
            "sample_0": {
                "interleave_array":    [prompt],
                "element_dtype_array": ["text"],
                "task":                "t2v",
            }
        })
        frames = _run_generation(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            new_token_ids=new_token_ids, image_token_id=image_token_id,
            json_path=json_path, task="t2v",
            width=width, height=height, num_frames=num_frames,
            num_steps=num_steps, cfg_scale=cfg_scale,
            timestep_shift=timestep_shift, seed=seed,
            device=device, dtype=dtype,
        )
    return frames or [torch.zeros(1, height, width, 3)]


def run_video_edit(
    *, model, vae_model, tokenizer, new_token_ids, image_token_id,
    source_frames: List[PILImage.Image], instruction,
    num_steps, cfg_scale, timestep_shift, seed, device, dtype,
) -> List[torch.Tensor]:
    """Video editing → list of (1,H,W,3) ComfyUI IMAGE tensors."""
    set_seed(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(source_frames):
            p = os.path.join(tmpdir, f"frame_{i:05d}.png")
            frame.save(p)
            frame_paths.append(p)
        h, w = source_frames[0].height, source_frames[0].width
        json_path = _write_json(tmpdir, {
            "sample_0": {
                "interleave_array":    [frame_paths, ["", instruction]],
                "element_dtype_array": ["video", "text"],
                "task":                "video_edit",
            }
        })
        result = _run_generation(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            new_token_ids=new_token_ids, image_token_id=image_token_id,
            json_path=json_path, task="video_edit",
            width=w, height=h, num_frames=len(source_frames),
            num_steps=num_steps, cfg_scale=cfg_scale,
            timestep_shift=timestep_shift, seed=seed,
            device=device, dtype=dtype,
        )
    return result or [
        torch.from_numpy(np.array(f).astype(np.float32) / 255.0).unsqueeze(0)
        for f in source_frames
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(tmpdir: str, data: dict) -> str:
    path = os.path.join(tmpdir, "prompt.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _run_generation(
    *, model, vae_model, tokenizer, new_token_ids, image_token_id,
    json_path, task, width, height, num_frames,
    num_steps, cfg_scale, timestep_shift, seed, device, dtype,
) -> List[torch.Tensor]:
    """
    Core driver: build a single-item ValidationDataset batch, run Lance
    generation, VAE-decode, and return ComfyUI frame tensors.
    """
    from torch.utils.data import DataLoader
    from data.datasets_custom import ValidationDataset
    from data.dataset_base import DataConfig, simple_custom_collate
    from common.val.utils import make_padded_latent
    from common.utils.misc import tuple_mul
    from config.config_factory import ModelArguments, DataArguments, InferenceArguments

    # ---- minimal arg objects ----
    model_args = ModelArguments()
    model_args.latent_patch_size           = [1, 2, 2]
    model_args.max_num_frames              = 121
    model_args.max_latent_size             = [32, 96, 96]
    model_args.vit_max_num_patch_per_side  = 32
    model_args.vit_patch_size              = 14
    model_args.vit_patch_size_temporal     = 2
    model_args.cfg_text_scale              = cfg_scale

    data_args = DataArguments()
    data_args.val_dataset_config_file = json_path

    is_image_task = task in ("t2i", "image_edit")
    inf_args = InferenceArguments()
    inf_args.task                      = task
    inf_args.video_height              = height
    inf_args.video_width               = width
    inf_args.num_frames                = num_frames
    inf_args.validation_num_timesteps  = num_steps
    inf_args.validation_timestep_shift = timestep_shift
    inf_args.validation_max_samples    = 1
    inf_args.validation_noise_seed     = seed
    inf_args.validation_data_seed      = seed
    inf_args.apply_chat_template       = True
    inf_args.apply_qwen_2_5_vl_pos_emb = True
    inf_args.cfg_type                  = "full"
    inf_args.cfg_uncond_token_id       = None
    inf_args.cfg_interval              = [0.0, 1.0]
    inf_args.cfg_renorm_min            = 1.0
    inf_args.cfg_renorm_type           = "global"
    inf_args.use_KVcache               = False
    inf_args.resolution                = "image_768res" if is_image_task else "video_480p"
    inf_args.text_template             = False

    vae_cfg = vae_model.vae_config
    vae_downsample = tuple_mul(
        model_args.latent_patch_size,
        (vae_cfg.downsample_temporal, vae_cfg.downsample_spatial, vae_cfg.downsample_spatial),
    )

    dataset_config = DataConfig.from_yaml(json_path)
    dataset_config.vit_patch_size             = model_args.vit_patch_size
    dataset_config.vit_patch_size_temporal    = model_args.vit_patch_size_temporal
    dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    dataset_config.latent_patch_size          = model_args.latent_patch_size
    dataset_config.vae_downsample             = vae_downsample
    dataset_config.max_latent_size            = model_args.max_latent_size
    dataset_config.max_num_frames             = model_args.max_num_frames
    dataset_config.text_cond_dropout_prob     = 0.0
    dataset_config.vae_cond_dropout_prob      = 0.0
    dataset_config.vit_cond_dropout_prob      = 0.0
    dataset_config.num_frames                 = num_frames
    dataset_config.H                          = height
    dataset_config.W                          = width
    dataset_config.task                       = task
    dataset_config.resolution                 = inf_args.resolution
    dataset_config.text_template              = inf_args.text_template

    val_dataset = ValidationDataset(
        jsonl_path=json_path, tokenizer=tokenizer,
        data_args=data_args, model_args=model_args, training_args=inf_args,
        new_token_ids=new_token_ids, dataset_config=dataset_config,
        local_rank=0, world_size=1,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=0, pin_memory=True,
        collate_fn=simple_custom_collate, drop_last=False,
    )

    result_frames: List[torch.Tensor] = []
    cuda_idx = device.index if hasattr(device, "index") and device.index is not None else 0

    for val_data_cpu in val_loader:
        val_data  = val_data_cpu.cuda(cuda_idx).to_dict()
        model_dev = model.to(device=device, dtype=dtype)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            if "padded_videos" in val_data:
                val_data["padded_latent"] = make_padded_latent(
                    val_data["padded_videos"], val_data["vae_data_mode"], vae_model
                )

            denoise_latent, captions, _, index = model_dev.validation_gen(
                val_packed_text_ids           = val_data["packed_text_ids"],
                val_packed_text_indexes       = val_data["packed_text_indexes"],
                val_sample_lens               = val_data["sample_lens"],
                val_packed_position_ids       = val_data["packed_position_ids"],
                val_split_lens                = val_data["split_lens"],
                val_attn_modes                = val_data["attn_modes"],
                val_sample_N_target           = val_data["sample_N_target"],
                val_packed_vae_token_indexes  = val_data["packed_vae_token_indexes"],
                timestep_shift                = timestep_shift,
                num_timesteps                 = num_steps,
                val_mse_loss_indexes          = val_data.get("mse_loss_indexes"),
                val_padded_latent             = val_data["padded_latent"],
                video_sizes                   = val_data["video_sizes"],
                cfg_text_scale                = cfg_scale,
                cfg_interval                  = inf_args.cfg_interval,
                cfg_renorm_min                = inf_args.cfg_renorm_min,
                cfg_renorm_type               = inf_args.cfg_renorm_type,
                device                        = device,
                dtype                         = dtype,
                new_token_ids                 = new_token_ids,
                max_samples                   = 1,
                validation_noise_seed         = seed,
                apply_chat_template           = True,
                apply_qwen_2_5_vl_pos_emb     = True,
                image_token_id                = image_token_id,
                val_packed_vit_token_indexes  = val_data.get("packed_vit_token_indexes"),
                val_packed_vit_tokens         = val_data.get("packed_vit_tokens"),
                vit_video_grid_thw            = val_data.get("vit_video_grid_thw"),
                vae_video_grid_thw            = val_data["vae_video_grid_thw"],
                video_grid_thw                = val_data.get("video_grid_thw"),
                caption                       = val_data.get("caption"),
                sample_task                   = val_data["sample_task"],
                sample_modality               = val_data["sample_modality"],
                cfg_type                      = inf_args.cfg_type,
                cfg_uncond_token_id           = inf_args.cfg_uncond_token_id,
                index                         = val_data["index"],
                val_padded_videos             = None,
            )

            for latent_list in denoise_latent:
                targets = [latent_list[-1]] if task in ("image_edit", "video_edit") else latent_list
                for lat in targets:
                    decoded = vae_model.vae_decode([lat])[0]
                    result_frames.extend(_video_tensor_to_frames(decoded))

        _clean()
        break  # single-sample inference

    return result_frames
