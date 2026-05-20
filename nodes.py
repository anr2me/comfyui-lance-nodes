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
#   model.safetensors (main checkpoint + tokenizer) → ComfyUI/models/diffusion_models/
#   vit.safetensors (vision encoder)                → ComfyUI/models/text_encoders/
#   VAE weights  → ComfyUI/models/vae/
# ---------------------------------------------------------------------------

def _list_subdirs(base: Path) -> list[str]:
    """Return immediate child directory names, or a placeholder if empty."""
    if not base.exists():
        return ["<not found>"]
    dirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return dirs or ["<empty>"]

def _list_safetensors(base: Path) -> list[str]:
    """Return *.safetensors filenames directly inside *base*."""
    if not base.exists():
        return ["<not found>"]
    files = sorted(f.name for f in base.iterdir() if f.suffix == ".safetensors")
    return files or ["<empty>"]

def _diffusion_models_dir() -> Path:
    # folder_paths registers 'diffusion_models' in recent ComfyUI versions.
    # Fall back gracefully for older installs.
    try:
        paths = folder_paths.get_folder_paths("diffusion_models")
        return Path(paths[0])
    except Exception:
        return Path(folder_paths.models_dir) / "diffusion_models"

def _text_encoders_dir() -> Path:
    try:
        paths = folder_paths.get_folder_paths("text_encoders")
        return Path(paths[0])
    except Exception:
        return Path(folder_paths.models_dir) / "text_encoders"

def _vae_dir() -> Path:
    try:
        paths = folder_paths.get_folder_paths("vae")
        return Path(paths[0])
    except Exception:
        return Path(folder_paths.models_dir) / "vae"


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
    model_dir: str,
    vit_dir: str,
    vae_file: str,
    device: str,
    dtype: str,
):
    """
    Load and cache the full Lance pipeline.

    Parameters
    ----------
    model_dir : absolute path to the model folder inside models/diffusion_models/
                Must contain llm_config.json + tokenizer files +
                model.safetensors (or ema.safetensors)
    vit_dir   : absolute path to the ViT folder inside models/text_encoders/
                Must contain config.json + vit.safetensors
    vae_file  : absolute path to the VAE .safetensors inside models/vae/
                (Lance's WanVideoVAE loads its own architecture; the file
                 provides the learned weights)
    device    : 'cuda' or 'cpu'
    dtype     : 'bf16' or 'fp16'
    """
    cache_key = f"{model_dir}|{vit_dir}|{vae_file}|{device}|{dtype}"
    if cache_key in _LANCE_CACHE:
        return _LANCE_CACHE[cache_key]

    # ---- delayed imports so ComfyUI loads even if Lance submodule is absent --
    try:
        import warnings
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
        f"  Model : {model_dir}\n"
        f"  ViT   : {vit_dir}\n"
        f"  VAE   : {vae_file}"
    )

    # ------------------------------------------------------------------
    # LLM + main checkpoint  (diffusion_models/<folder>/)
    # ------------------------------------------------------------------
    llm_config: Qwen2Config = Qwen2Config.from_json_file(
        os.path.join(model_dir, "llm_config.json")
    )
    llm_config.apply_qwen_2_5_vl_pos_emb = True
    language_model = Qwen2ForCausalLM(llm_config)

    # ------------------------------------------------------------------
    # ViT  (text_encoders/<folder>/)
    # ------------------------------------------------------------------
    vit_config = Qwen2_5_VLVisionConfig.from_pretrained(vit_dir)
    vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
    vit_weights = load_file(os.path.join(vit_dir, "vit.safetensors"))
    vit_model.load_state_dict(vit_weights, strict=True)
    _clean_memory(vit_weights)

    # ------------------------------------------------------------------
    # VAE  (vae/<file>.safetensors)
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

    # Load main model checkpoint from model_dir
    ema_path  = os.path.join(model_dir, "ema.safetensors")
    ckpt_path = os.path.join(model_dir, "model.safetensors")
    load_path = ckpt_path if os.path.exists(ckpt_path) else (
                ema_path  if os.path.exists(ema_path)  else None)
    if load_path is None:
        raise FileNotFoundError(
            f"No Lance checkpoint found in '{model_dir}'.\n"
            "Expected 'model.safetensors' or 'ema.safetensors'."
        )
    print(f"[LanceNodes] Loading checkpoint: {load_path}")
    state_dict = load_file(load_path, device="cpu")
    state_dict.pop("latent_pos_embed.pos_embed", None)
    model.load_state_dict(state_dict, strict=False)
    _clean_memory(state_dict)

    # Tokenizer (lives alongside the main model weights in diffusion_models/)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
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

    result = (model, vae_model, tokenizer, new_token_ids, image_token_id, dev, torch_dtype)
    _LANCE_CACHE[cache_key] = result
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

    Model components are split across three standard ComfyUI model directories:

    • Main model checkpoint + tokenizer
        models/diffusion_models/<folder>/
            llm_config.json
            model.safetensors  (or ema.safetensors)
            tokenizer.json / tokenizer_config.json / …

    • ViT (vision encoder) weights
        models/text_encoders/<folder>/
            config.json
            vit.safetensors

    • VAE weights
        models/vae/
            <file>.safetensors
    """

    CATEGORY = "Lance"
    FUNCTION = "load_model"
    RETURN_TYPES  = ("LANCE_PIPELINE",)
    RETURN_NAMES  = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        model_dirs = _list_subdirs(_diffusion_models_dir())
        vit_dirs   = _list_subdirs(_text_encoders_dir())
        vae_files  = _list_safetensors(_vae_dir())
        return {
            "required": {
                "model_folder": (model_dirs, {"default": model_dirs[0]}),
                "vit_folder":   (vit_dirs,   {"default": vit_dirs[0]}),
                "vae_file":     (vae_files,  {"default": vae_files[0]}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype":  (["bf16", "fp16"], {"default": "bf16"}),
            }
        }

    def load_model(
        self,
        model_folder: str,
        vit_folder: str,
        vae_file: str,
        device: str,
        dtype: str,
    ):
        model_path = str(_diffusion_models_dir() / model_folder)
        vit_path   = str(_text_encoders_dir()    / vit_folder)
        vae_path   = str(_vae_dir()              / vae_file)
        pipeline = _load_lance_pipeline(model_path, vit_path, vae_path, device, dtype)
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
        for k in [k for k, v in _LANCE_CACHE.items() if v is pipeline]:
            del _LANCE_CACHE[k]
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
