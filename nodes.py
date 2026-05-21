# ComfyUI Custom Nodes for Lance (bytedance/Lance)
# A lightweight native unified multimodal model for image and video
# understanding, generation, and editing.
#
# Source model: https://github.com/bytedance/Lance  (included as git submodule)
# License: Apache-2.0

import os
import gc
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

import folder_paths

# ---------------------------------------------------------------------------
# Ensure the Lance submodule is on sys.path before any Lance imports.
# install.py does this at startup, but we guard here too so the nodes work
# even when loaded without going through ComfyUI Manager's installer.
# ---------------------------------------------------------------------------

_NODE_ROOT = Path(__file__).parent.resolve()
_LANCE_ROOT = _NODE_ROOT / "Lance"

def _ensure_lance_on_path():
    lance_str = str(_LANCE_ROOT)
    if lance_str not in sys.path:
        sys.path.insert(0, lance_str)

_ensure_lance_on_path()


# ---------------------------------------------------------------------------
# ComfyUI model-directory helpers
#
#   LLM checkpoint + tokenizer  → ComfyUI/models/LLM/
#   ViT (vision encoder)        → ComfyUI/models/clip_vision/
#   VAE weights                 → ComfyUI/models/vae/
#
# Each of the LLM and ViT entries is a SINGLE .safetensors file whose
# companion config/tokenizer JSON files may either sit beside it on disk
# OR be embedded in the safetensors metadata header under keys such as
# "llm_config", "tokenizer_config", "config", etc.
# (see https://huggingface.co/anr2me/bytedance_lance)
# ---------------------------------------------------------------------------

def _list_safetensors(base: Path) -> list[str]:
    """Return *.safetensors filenames directly inside *base*."""
    if not base.exists():
        return ["<not found>"]
    files = sorted(f.name for f in base.iterdir() if f.suffix == ".safetensors")
    return files or ["<empty>"]

def _llm_dir() -> Path:
    """models/LLM/ — main Lance checkpoint + tokenizer."""
    try:
        # ComfyUI does not register "LLM" by default; use models_dir fallback.
        paths = folder_paths.get_folder_paths("LLM")
        return Path(paths[0])
    except Exception:
        return Path(folder_paths.models_dir) / "LLM"

def _clip_vision_dir() -> Path:
    """models/clip_vision/ — ViT (vision encoder) weights."""
    try:
        paths = folder_paths.get_folder_paths("clip_vision")
        return Path(paths[0])
    except Exception:
        return Path(folder_paths.models_dir) / "clip_vision"

def _vae_dir() -> Path:
    try:
        paths = folder_paths.get_folder_paths("vae")
        return Path(paths[0])
    except Exception:
        return Path(folder_paths.models_dir) / "vae"


# ---------------------------------------------------------------------------
# Embedded-metadata helpers
#
# The anr2me/bytedance_lance HF files store every companion JSON directly
# inside the safetensors header so no sidecar files are needed.
# Metadata keys used:
#
#   LLM file   : "llm_config"        → llm_config.json  (Qwen2Config)
#                "tokenizer_config"  → tokenizer_config.json
#                "tokenizer"         → tokenizer.json  (vocab + merges)
#                "special_tokens_map"→ special_tokens_map.json
#
#   ViT file   : "config"            → config.json  (Qwen2_5_VLVisionConfig)
# ---------------------------------------------------------------------------

def _read_safetensors_metadata(path: str) -> dict:
    """
    Return the metadata dict from a safetensors file header without loading
    any tensors.  Falls back to {} on any error.
    """
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            return dict(f.metadata()) if f.metadata() else {}
    except Exception as e:
        print(f"[LanceNodes] Could not read metadata from {path}: {e}")
        return {}


def _resolve_json(
    safetensors_path: str,
    sidecar_path: str,
    metadata_key: str,
) -> dict:
    """
    Return a parsed JSON dict, preferring the sidecar file if it exists,
    and falling back to the value stored under *metadata_key* in the
    safetensors header.

    Raises FileNotFoundError if neither source is available.
    """
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as fh:
            return json.load(fh)

    meta = _read_safetensors_metadata(safetensors_path)
    if metadata_key in meta:
        raw = meta[metadata_key]
        return json.loads(raw)

    raise FileNotFoundError(
        f"Companion file '{os.path.basename(sidecar_path)}' not found beside "
        f"'{safetensors_path}' and metadata key '{metadata_key}' is absent from "
        "the safetensors header.\n"
        "Download a self-contained file from "
        "https://huggingface.co/anr2me/bytedance_lance or place the companion "
        "JSON files in the same directory."
    )


def _resolve_text(
    safetensors_path: str,
    sidecar_path: str,
    metadata_key: str,
) -> str:
    """Like _resolve_json but returns the raw string (for tokenizer vocab etc.)."""
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as fh:
            return fh.read()

    meta = _read_safetensors_metadata(safetensors_path)
    if metadata_key in meta:
        return meta[metadata_key]

    raise FileNotFoundError(
        f"Companion file '{os.path.basename(sidecar_path)}' not found beside "
        f"'{safetensors_path}' and metadata key '{metadata_key}' is absent from "
        "the safetensors header."
    )


def _materialise_companion_files(safetensors_path: str, tmpdir: str) -> str:
    """
    Ensure every companion file that a HuggingFace *from_pretrained* call
    expects actually exists on disk, materialising any that are missing by
    extracting them from the safetensors header.

    Returns the directory to pass to from_pretrained / from_json_file.
    """
    model_dir = os.path.dirname(safetensors_path)
    meta = _read_safetensors_metadata(safetensors_path)

    # Map: filename on disk → metadata key (used when the file is absent)
    companion_map = {
        "generation_config.json": "generation_config_json",
        "llm_config.json":        "llm_config_json",
        "tokenizer_config.json":  "tokenizer_config_json",
        "tokenizer.json":         "tokenizer_json",
        "special_tokens_map.json":"special_tokens_map_json",
        "vocab.json":             "vocab_json",
        "merges.txt":             "merges_txt",
        "config.json":            "config",         # ViT
    }

    # Check whether every companion already exists alongside the weights
    all_present = all(
        os.path.isfile(os.path.join(model_dir, fname))
        for fname in companion_map
        if fname in (os.listdir(model_dir) or []) or fname in meta
    )

    # If any required companion is missing from model_dir, work in tmpdir
    missing = [
        fname for fname, key in companion_map.items()
        if key in meta and not os.path.isfile(os.path.join(model_dir, fname))
    ]

    if not missing:
        return model_dir  # everything already on disk beside the weights

    # Write missing companions to tmpdir, copy the weights path there too
    # so that from_pretrained(tmpdir) can find everything in one place.
    import shutil
    out_dir = tmpdir

    # Symlink / copy the safetensors itself so tokenizer loading can find it
    dst_weights = os.path.join(out_dir, os.path.basename(safetensors_path))
    if not os.path.exists(dst_weights):
        try:
            os.symlink(os.path.abspath(safetensors_path), dst_weights)
        except OSError:
            shutil.copy2(safetensors_path, dst_weights)

    # Copy any sidecar files that DO exist
    for fname in companion_map:
        src = os.path.join(model_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Write companions extracted from metadata
    for fname, key in companion_map.items():
        dst = os.path.join(out_dir, fname)
        if os.path.exists(dst):
            continue
        if key not in meta:
            continue
        with open(dst, "w") as fh:
            fh.write(meta[key])

    return out_dir


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _clean_memory(*objects):
    for obj in objects:
        if isinstance(obj, (dict, list, set)):
            obj.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Lazy pipeline cache   key → (model, vae, tokenizer, new_token_ids,
#                               image_token_id, device, dtype)
# ---------------------------------------------------------------------------

_LANCE_CACHE: Dict[str, Any] = {}


def _load_lance_pipeline(
    llm_file: str,
    vit_file: str,
    vae_file: str,
    device: str,
    dtype: str,
):
    """
    Load and cache the full Lance pipeline.

    Parameters
    ----------
    llm_file  : absolute path to the LLM .safetensors inside models/LLM/
                Companion files (llm_config.json, tokenizer files) are read
                from the same directory OR extracted from the file's metadata
                header when absent (self-contained HF format).
    vit_file  : absolute path to the ViT .safetensors inside models/clip_vision/
                config.json is read from the same directory OR from metadata.
    vae_file  : absolute path to the VAE .safetensors inside models/vae/
    device    : 'cuda' or 'cpu'
    dtype     : 'bf16' or 'fp16'
    """
    cache_key = f"{llm_file}|{vit_file}|{vae_file}|{device}|{dtype}"
    if cache_key in _LANCE_CACHE:
        return _LANCE_CACHE[cache_key]

    # ---- delayed imports so ComfyUI loads even if Lance submodule is absent --
    try:
        import warnings
        import tempfile
        warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

        from copy import deepcopy
        from safetensors.torch import load_file
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLVisionConfig,
        )

        # Lance internals (available once submodule is on sys.path)
        from modeling.lance import LanceConfig, Lance, Qwen2ForCausalLM
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.qwen2.modeling_qwen2 import Qwen2Config
        from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
        from modeling.vae.wan.model import WanVideoVAE
        from data.data_utils import add_special_tokens
        from common.model.hacks import hack_qwen2_5_vl_config

    except ImportError as exc:
        raise ImportError(
            f"Lance submodule import failed: {exc}\n\n"
            "Make sure the Lance submodule has been initialised:\n"
            "  cd comfyui-lance-nodes && git submodule update --init --recursive\n"
            f"Expected submodule root: {_LANCE_ROOT}"
        ) from exc

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    t0 = time.perf_counter()
    print(
        f"[LanceNodes] Loading pipeline\n"
        f"  LLM : {llm_file}\n"
        f"  ViT : {vit_file}\n"
        f"  VAE : {vae_file}"
    )

    # We may need to materialise companion JSONs from metadata into a temp dir.
    # Keep it alive for the whole function by opening it here.
    _tmpdir_ctx = tempfile.TemporaryDirectory(prefix="lance_nodes_")
    tmpdir = _tmpdir_ctx.name

    # ------------------------------------------------------------------
    # LLM + main checkpoint  (models/LLM/<file>.safetensors)
    #
    # Companion files required by Qwen2Config / Qwen2Tokenizer:
    #   llm_config.json, tokenizer_config.json, tokenizer.json,
    #   special_tokens_map.json, vocab.json, merges.txt
    # ------------------------------------------------------------------
    llm_tmpdir = os.path.join(tmpdir, "llm")
    os.makedirs(llm_tmpdir, exist_ok=True)
    llm_dir = _materialise_companion_files(llm_file, llm_tmpdir)

    llm_config_path = os.path.join(llm_dir, "generation_config.json")
    llm_config: Qwen2Config = Qwen2Config.from_json_file(llm_config_path)
    llm_config.apply_qwen_2_5_vl_pos_emb = True
    language_model = Qwen2ForCausalLM(llm_config)

    # ------------------------------------------------------------------
    # ViT  (models/clip_vision/<file>.safetensors)
    #
    # Companion file required: config.json  (Qwen2_5_VLVisionConfig)
    # ------------------------------------------------------------------
    vit_tmpdir = os.path.join(tmpdir, "vit")
    os.makedirs(vit_tmpdir, exist_ok=True)
    vit_dir = _materialise_companion_files(vit_file, vit_tmpdir)

    vit_config = Qwen2_5_VLVisionConfig.from_pretrained(vit_dir)
    vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
    vit_weights = load_file(vit_file)   # always load weights from the original file
    vit_model.load_state_dict(vit_weights, strict=True)
    _clean_memory(vit_weights)

    # ------------------------------------------------------------------
    # VAE  (models/vae/<file>.safetensors)
    # ------------------------------------------------------------------
    vae_model = WanVideoVAE()
    if os.path.isfile(vae_file):
        vae_weights = load_file(vae_file, device="cpu")
        vae_model.load_state_dict(vae_weights, strict=True)
        _clean_memory(vae_weights)
        print(f"[LanceNodes] VAE weights loaded from {vae_file}")
    else:
        print(
            f"[LanceNodes] WARNING: VAE file not found at '{vae_file}'. "
            "Using randomly initialised weights – outputs will be nonsense."
        )
    vae_config = deepcopy(vae_model.vae_config)

    # ------------------------------------------------------------------
    # Assemble Lance
    # ------------------------------------------------------------------
    config = LanceConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=[1, 2, 2],
        max_num_frames=121,
        max_latent_size=[32, 96, 96],
        vit_max_num_patch_per_side=32,
        connector_act="silu",
        interpolate_pos=True,
        timestep_shift=3.5,
    )

    class _InferenceArgs:
        visual_gen = True
        visual_und = True
        copy_init_moe = False
        freeze_und = False
        apply_qwen_2_5_vl_pos_emb = True

    model = Lance(
        language_model=language_model,
        vit_model=vit_model,
        vit_type="qwen2_5_vl",
        config=config,
        training_args=_InferenceArgs(),
    )

    # Load main model checkpoint — the LLM .safetensors IS the checkpoint
    print(f"[LanceNodes] Loading checkpoint: {llm_file}")
    state_dict = load_file(llm_file, device="cpu")
    state_dict.pop("latent_pos_embed.pos_embed", None)
    model.load_state_dict(state_dict, strict=False)
    _clean_memory(state_dict)

    # Tokenizer — use materialised companion dir so from_pretrained finds all files
    tokenizer = Qwen2Tokenizer.from_pretrained(llm_dir)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    language_model = hack_qwen2_5_vl_config(language_model)
    image_token_id = language_model.config.video_token_id
    new_token_ids["image_token_id"] = image_token_id
    model.update_tokenizer(tokenizer=tokenizer)

    dev = torch.device(device)
    model     = model.to(device=dev, dtype=torch_dtype).eval()
    vae_model = vae_model.to(device=dev).eval()

    # Keep the temp dir alive as long as the pipeline is cached
    result = (model, vae_model, tokenizer, new_token_ids, image_token_id, dev, torch_dtype)
    _LANCE_CACHE[cache_key] = result
    _LANCE_CACHE[cache_key + "__tmpdir"] = _tmpdir_ctx   # prevents GC cleanup
    print(f"[LanceNodes] Pipeline ready in {time.perf_counter() - t0:.1f}s")
    return result


# ===========================================================================
# Node definitions
# ===========================================================================

# ---------------------------------------------------------------------------
# Lance Model Loader
# ---------------------------------------------------------------------------

class LanceModelLoader:
    """
    Load the Lance-3B pipeline.

    Model components use three standard ComfyUI model directories:

    • Main checkpoint + tokenizer  →  models/LLM/<file>.safetensors
      (companion JSON files can be beside the .safetensors OR embedded
       in its metadata header — compatible with anr2me/bytedance_lance HF files)

    • ViT (vision encoder)  →  models/clip_vision/<file>.safetensors
      (config.json can be beside the file OR embedded in its metadata header)

    • VAE  →  models/vae/<file>.safetensors
    """

    CATEGORY = "Lance"
    FUNCTION = "load_model"
    RETURN_TYPES  = ("LANCE_PIPELINE",)
    RETURN_NAMES  = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        llm_files = _list_safetensors(_llm_dir())
        vit_files = _list_safetensors(_clip_vision_dir())
        vae_files = _list_safetensors(_vae_dir())
        return {
            "required": {
                "llm_file": (llm_files, {"default": llm_files[0]}),
                "vit_file": (vit_files, {"default": vit_files[0]}),
                "vae_file": (vae_files, {"default": vae_files[0]}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype":  (["bf16", "fp16"], {"default": "bf16"}),
            }
        }

    def load_model(
        self,
        llm_file: str,
        vit_file: str,
        vae_file: str,
        device: str,
        dtype: str,
    ):
        llm_path = str(_llm_dir()         / llm_file)
        vit_path = str(_clip_vision_dir() / vit_file)
        vae_path = str(_vae_dir()         / vae_file)
        pipeline = _load_lance_pipeline(llm_path, vit_path, vae_path, device, dtype)
        return (pipeline,)


# ---------------------------------------------------------------------------
# Lance Text-to-Image
# ---------------------------------------------------------------------------

class LanceTextToImage:
    """Generate an image from a text prompt using Lance."""

    CATEGORY = "Lance"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":        ("LANCE_PIPELINE",),
                "prompt":          ("STRING", {"multiline": True,
                                               "default": "A beautiful sunset over the ocean."}),
                "width":           ("INT",   {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "height":          ("INT",   {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "num_steps":       ("INT",   {"default": 30,  "min": 10,  "max": 100}),
                "cfg_scale":       ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "timestep_shift":  ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed":            ("INT",   {"default": 42,  "min": 0,   "max": 2**31 - 1}),
            }
        }

    def generate(self, pipeline, prompt, width, height,
                 num_steps, cfg_scale, timestep_shift, seed):
        model, vae_model, tokenizer, new_token_ids, image_token_id, device, dtype = pipeline
        try:
            from lance_backend.generation import run_t2i
            result = run_t2i(
                model=model, vae_model=vae_model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_token_id=image_token_id,
                prompt=prompt, width=width, height=height,
                num_steps=num_steps, cfg_scale=cfg_scale,
                timestep_shift=timestep_shift, seed=seed,
                device=device, dtype=dtype,
            )
        except Exception as exc:
            print(f"[LanceNodes] run_t2i error: {exc}")
            result = torch.zeros(1, height, width, 3)
        return (result,)


# ---------------------------------------------------------------------------
# Lance Image Edit
# ---------------------------------------------------------------------------

class LanceImageEdit:
    """Edit an existing image using a natural-language instruction."""

    CATEGORY = "Lance"
    FUNCTION = "edit"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":       ("LANCE_PIPELINE",),
                "image":          ("IMAGE",),
                "instruction":    ("STRING", {"multiline": True,
                                              "default": "Make the sky more dramatic and stormy."}),
                "num_steps":      ("INT",   {"default": 30,  "min": 10, "max": 100}),
                "cfg_scale":      ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "timestep_shift": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 42,  "min": 0,   "max": 2**31 - 1}),
            }
        }

    def edit(self, pipeline, image, instruction,
             num_steps, cfg_scale, timestep_shift, seed):
        model, vae_model, tokenizer, new_token_ids, image_token_id, device, dtype = pipeline
        from PIL import Image as PILImage
        img_np    = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(img_np)
        try:
            from lance_backend.generation import run_image_edit
            result = run_image_edit(
                model=model, vae_model=vae_model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_token_id=image_token_id,
                source_image=pil_image, instruction=instruction,
                num_steps=num_steps, cfg_scale=cfg_scale,
                timestep_shift=timestep_shift, seed=seed,
                device=device, dtype=dtype,
            )
        except Exception as exc:
            print(f"[LanceNodes] run_image_edit error: {exc}")
            result = image
        return (result,)


# ---------------------------------------------------------------------------
# Lance Text-to-Video
# ---------------------------------------------------------------------------

class LanceTextToVideo:
    """Generate a video from a text prompt using Lance."""

    CATEGORY = "Lance"
    FUNCTION = "generate"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":        ("LANCE_PIPELINE",),
                "prompt":          ("STRING", {"multiline": True,
                                               "default": "A cat playfully chasing butterflies in a sunny garden."}),
                "width":           ("INT",   {"default": 832, "min": 256, "max": 1920, "step": 64}),
                "height":          ("INT",   {"default": 480, "min": 256, "max": 1080, "step": 64}),
                "num_frames":      ("INT",   {"default": 50,  "min": 1,   "max": 121}),
                "num_steps":       ("INT",   {"default": 30,  "min": 10,  "max": 100}),
                "cfg_scale":       ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "timestep_shift":  ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed":            ("INT",   {"default": 42,  "min": 0,   "max": 2**31 - 1}),
            }
        }

    def generate(self, pipeline, prompt, width, height, num_frames,
                 num_steps, cfg_scale, timestep_shift, seed):
        model, vae_model, tokenizer, new_token_ids, image_token_id, device, dtype = pipeline
        try:
            from lance_backend.generation import run_t2v
            frames = run_t2v(
                model=model, vae_model=vae_model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_token_id=image_token_id,
                prompt=prompt, width=width, height=height, num_frames=num_frames,
                num_steps=num_steps, cfg_scale=cfg_scale,
                timestep_shift=timestep_shift, seed=seed,
                device=device, dtype=dtype,
            )
        except Exception as exc:
            print(f"[LanceNodes] run_t2v error: {exc}")
            frames = [torch.zeros(1, height, width, 3) for _ in range(num_frames)]
        return (frames,)


# ---------------------------------------------------------------------------
# Lance Video Edit
# ---------------------------------------------------------------------------

class LanceVideoEdit:
    """Edit a video clip using a natural-language instruction."""

    CATEGORY = "Lance"
    FUNCTION = "edit"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    OUTPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":       ("LANCE_PIPELINE",),
                "frames":         ("IMAGE",),
                "instruction":    ("STRING", {"multiline": True,
                                              "default": "Apply a cinematic color grade."}),
                "num_steps":      ("INT",   {"default": 30,  "min": 10, "max": 100}),
                "cfg_scale":      ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "timestep_shift": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed":           ("INT",   {"default": 42,  "min": 0,   "max": 2**31 - 1}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 50, "min": 1, "max": 121}),
            }
        }

    def edit(self, pipeline, frames, instruction,
             num_steps, cfg_scale, timestep_shift, seed, max_frames=50):
        model, vae_model, tokenizer, new_token_ids, image_token_id, device, dtype = pipeline
        from PIL import Image as PILImage
        pil_frames = []
        for i in range(min(len(frames), max_frames)):
            img_np = (frames[i].cpu().numpy() * 255).astype(np.uint8)
            pil_frames.append(PILImage.fromarray(img_np))
        try:
            from lance_backend.generation import run_video_edit
            result = run_video_edit(
                model=model, vae_model=vae_model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_token_id=image_token_id,
                source_frames=pil_frames, instruction=instruction,
                num_steps=num_steps, cfg_scale=cfg_scale,
                timestep_shift=timestep_shift, seed=seed,
                device=device, dtype=dtype,
            )
        except Exception as exc:
            print(f"[LanceNodes] run_video_edit error: {exc}")
            result = [frames[i].unsqueeze(0) if frames[i].dim() == 3 else frames[i]
                      for i in range(len(pil_frames))]
        return (result,)


# ---------------------------------------------------------------------------
# Lance Image Understanding
# ---------------------------------------------------------------------------

class LanceImageUnderstanding:
    """Answer a question about an image, or generate a caption."""

    CATEGORY = "Lance"
    FUNCTION = "understand"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":       ("LANCE_PIPELINE",),
                "image":          ("IMAGE",),
                "question":       ("STRING", {"multiline": True,
                                              "default": "Describe this image in detail."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 1024}),
            }
        }

    def understand(self, pipeline, image, question, max_new_tokens):
        model, vae_model, tokenizer, new_token_ids, image_token_id, device, dtype = pipeline
        from PIL import Image as PILImage
        img_np    = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(img_np)
        try:
            from lance_backend.understanding import run_image_understanding
            answer = run_image_understanding(
                model=model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_token_id=image_token_id,
                image=pil_image, question=question,
                max_new_tokens=max_new_tokens, device=device, dtype=dtype,
            )
        except Exception as exc:
            answer = f"[LanceNodes error: {exc}]"
        return (answer,)


# ---------------------------------------------------------------------------
# Lance Video Understanding
# ---------------------------------------------------------------------------

class LanceVideoUnderstanding:
    """Answer a question about a video clip, or generate a video caption."""

    CATEGORY = "Lance"
    FUNCTION = "understand"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":       ("LANCE_PIPELINE",),
                "frames":         ("IMAGE",),
                "question":       ("STRING", {"multiline": True,
                                              "default": "Describe what happens in this video."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 1024}),
                "sample_frames":  ("INT", {"default": 8,   "min": 1,  "max": 32}),
            }
        }

    def understand(self, pipeline, frames, question, max_new_tokens, sample_frames):
        model, vae_model, tokenizer, new_token_ids, image_token_id, device, dtype = pipeline
        from PIL import Image as PILImage
        total   = frames.shape[0] if isinstance(frames, torch.Tensor) else len(frames)
        indices = np.linspace(0, total - 1, min(sample_frames, total), dtype=int)
        pil_frames = []
        for idx in indices:
            f = frames[idx]
            if isinstance(f, torch.Tensor) and f.dim() == 4:
                f = f[0]
            img_np = (f.cpu().numpy() * 255).astype(np.uint8)
            pil_frames.append(PILImage.fromarray(img_np))
        try:
            from lance_backend.understanding import run_video_understanding
            answer = run_video_understanding(
                model=model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_token_id=image_token_id,
                frames=pil_frames, question=question,
                max_new_tokens=max_new_tokens, device=device, dtype=dtype,
            )
        except Exception as exc:
            answer = f"[LanceNodes error: {exc}]"
        return (answer,)


# ---------------------------------------------------------------------------
# Lance Unload Model
# ---------------------------------------------------------------------------

class LanceUnloadModel:
    """Remove a cached Lance pipeline from memory to free VRAM."""

    CATEGORY = "Lance"
    FUNCTION = "unload"
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipeline": ("LANCE_PIPELINE",)}}

    def unload(self, pipeline):
        global _LANCE_CACHE
        keys_to_del = [k for k, v in _LANCE_CACHE.items() if v is pipeline]
        for k in keys_to_del:
            del _LANCE_CACHE[k]
            # Also clean up the temp-dir context object kept alive alongside it
            tmpdir_key = k + "__tmpdir"
            if tmpdir_key in _LANCE_CACHE:
                _LANCE_CACHE[tmpdir_key].cleanup()
                del _LANCE_CACHE[tmpdir_key]
        del pipeline
        _clean_memory()
        print("[LanceNodes] Pipeline unloaded.")
        return ()


# ===========================================================================
# Registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "LanceModelLoader":        LanceModelLoader,
    "LanceTextToImage":        LanceTextToImage,
    "LanceImageEdit":          LanceImageEdit,
    "LanceTextToVideo":        LanceTextToVideo,
    "LanceVideoEdit":          LanceVideoEdit,
    "LanceImageUnderstanding": LanceImageUnderstanding,
    "LanceVideoUnderstanding": LanceVideoUnderstanding,
    "LanceUnloadModel":        LanceUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LanceModelLoader":        "Lance: Load Model",
    "LanceTextToImage":        "Lance: Text → Image",
    "LanceImageEdit":          "Lance: Edit Image",
    "LanceTextToVideo":        "Lance: Text → Video",
    "LanceVideoEdit":          "Lance: Edit Video",
    "LanceImageUnderstanding": "Lance: Image Understanding",
    "LanceVideoUnderstanding": "Lance: Video Understanding",
    "LanceUnloadModel":        "Lance: Unload Model",
}
