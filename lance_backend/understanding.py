# lance_backend/understanding.py
# Visual understanding helpers for the ComfyUI Lance custom nodes.
# Handles image VQA / captioning (x2t_image) and video VQA / captioning (x2t_video).
#
# Lance source lives in the sibling Lance/ git submodule; it must be on
# sys.path before these functions are called (install.py / __init__.py handle this).

from __future__ import annotations

import gc
import json
import os
import tempfile
from typing import List

import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_json(tmpdir: str, data: dict) -> str:
    path = os.path.join(tmpdir, "prompt.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _normalize_answer(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("<|im_end|>", "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_image_understanding(
    *,
    model,
    tokenizer,
    new_token_ids: dict,
    image_token_id: int,
    image: PILImage.Image,
    question: str,
    max_new_tokens: int,
    device,
    dtype,
) -> str:
    """
    Run Lance image understanding (x2t_image) and return the answer string.

    Parameters
    ----------
    model           : Lance model (already on device)
    tokenizer       : Qwen2Tokenizer
    new_token_ids   : dict of special token ids
    image_token_id  : id of the image/video pad token
    image           : PIL Image to query
    question        : natural-language question or captioning instruction
    max_new_tokens  : maximum number of tokens to generate
    device          : torch.device
    dtype           : torch.dtype
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "image.png")
        image.save(img_path)
        json_path = _write_json(tmpdir, {
            "sample_0": {
                "interleave_array":    [img_path, ["", question]],
                "element_dtype_array": ["image", "text"],
                "task":                "x2t_image",
            }
        })
        return _run_understanding(
            model=model, tokenizer=tokenizer,
            new_token_ids=new_token_ids, image_token_id=image_token_id,
            json_path=json_path, task="x2t_image",
            max_new_tokens=max_new_tokens, device=device, dtype=dtype,
        )


def run_video_understanding(
    *,
    model,
    tokenizer,
    new_token_ids: dict,
    image_token_id: int,
    frames: List[PILImage.Image],
    question: str,
    max_new_tokens: int,
    device,
    dtype,
) -> str:
    """
    Run Lance video understanding (x2t_video) and return the answer string.

    Parameters
    ----------
    frames : list of evenly-sampled PIL Images from the video
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(frames):
            p = os.path.join(tmpdir, f"frame_{i:05d}.png")
            frame.save(p)
            frame_paths.append(p)
        json_path = _write_json(tmpdir, {
            "sample_0": {
                "interleave_array":    [frame_paths, ["", question]],
                "element_dtype_array": ["video", "text"],
                "task":                "x2t_video",
            }
        })
        return _run_understanding(
            model=model, tokenizer=tokenizer,
            new_token_ids=new_token_ids, image_token_id=image_token_id,
            json_path=json_path, task="x2t_video",
            max_new_tokens=max_new_tokens, device=device, dtype=dtype,
        )


# ---------------------------------------------------------------------------
# Internal driver
# ---------------------------------------------------------------------------

def _run_understanding(
    *,
    model,
    tokenizer,
    new_token_ids: dict,
    image_token_id: int,
    json_path: str,
    task: str,               # "x2t_image" | "x2t_video"
    max_new_tokens: int,
    device,
    dtype,
) -> str:
    """
    Build a single-item ValidationDataset batch, run Lance video-to-text
    autoregressive decoding, and return the decoded answer string.
    """
    from torch.utils.data import DataLoader
    from data.datasets_custom import ValidationDataset
    from data.dataset_base import DataConfig, simple_custom_collate
    from config.config_factory import ModelArguments, DataArguments, InferenceArguments

    # ---- minimal arg objects (mirror what inference_lance.py builds) ----
    model_args = ModelArguments()
    model_args.latent_patch_size          = [1, 2, 2]
    model_args.max_num_frames             = 121
    model_args.max_latent_size            = [32, 96, 96]
    model_args.vit_max_num_patch_per_side = 32
    model_args.vit_patch_size             = 14
    model_args.vit_patch_size_temporal    = 2

    data_args = DataArguments()
    data_args.val_dataset_config_file = json_path

    inf_args = InferenceArguments()
    inf_args.task                      = task
    inf_args.apply_chat_template       = True
    inf_args.apply_qwen_2_5_vl_pos_emb = True
    inf_args.validation_max_samples    = 1
    inf_args.resolution                = "image_768res" if task == "x2t_image" else "video_480p"
    inf_args.text_template             = False
    inf_args.visual_gen                = True
    inf_args.visual_und                = True

    dataset_config = DataConfig.from_yaml(json_path)
    dataset_config.vit_patch_size             = model_args.vit_patch_size
    dataset_config.vit_patch_size_temporal    = model_args.vit_patch_size_temporal
    dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    dataset_config.latent_patch_size          = model_args.latent_patch_size
    dataset_config.max_latent_size            = model_args.max_latent_size
    dataset_config.max_num_frames             = model_args.max_num_frames
    dataset_config.text_cond_dropout_prob     = 0.0
    dataset_config.vae_cond_dropout_prob      = 0.0
    dataset_config.vit_cond_dropout_prob      = 0.0
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

    answer = ""
    cuda_idx = device.index if hasattr(device, "index") and device.index is not None else 0

    for val_data_cpu in val_loader:
        val_data  = val_data_cpu.cuda(cuda_idx).to_dict()
        model_dev = model.to(device=device, dtype=dtype)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            generated_all, _captions, _index = model_dev.validation_video_to_text(
                val_packed_text_ids        = val_data["packed_text_ids"],
                val_packed_text_indexes    = val_data["packed_text_indexes"],
                val_packed_position_ids    = val_data["packed_position_ids"],
                val_sample_N_target        = val_data["sample_N_target"],
                val_split_lens             = val_data["split_lens"],
                val_attn_modes             = val_data["attn_modes"],
                val_sample_lens            = val_data["sample_lens"],
                val_sample_type            = val_data["sample_type"],
                val_packed_vit_tokens      = val_data["packed_vit_tokens"],
                val_vit_video_grid_thw     = val_data["vit_video_grid_thw"],
                val_ce_loss_indexes        = val_data["ce_loss_indexes"],
                max_samples                = 1,
                max_length                 = max_new_tokens,
                device                     = device,
                dtype                      = dtype,
                new_token_ids              = new_token_ids,
                pad_token_id               = tokenizer.pad_token_id,
                vocab_size                 = len(tokenizer),
                caption                    = val_data.get("caption_cn"),
                tokenizer                  = tokenizer,
                apply_chat_template        = True,
                apply_qwen_2_5_vl_pos_emb  = True,
                do_sample                  = False,
                image_token_id             = image_token_id,
                index                      = val_data["index"],
            )

            for seq in generated_all:
                raw    = tokenizer.decode(seq[:, 0])
                answer = _normalize_answer(raw)
                break  # single sample

        _clean()
        break  # single batch

    return answer
