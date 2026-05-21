# ComfyUI Lance Nodes

> ComfyUI custom nodes for **[Lance](https://github.com/bytedance/Lance)** —
> ByteDance's lightweight 3B unified multimodal model supporting image & video
> understanding, generation, and editing in a single framework.

The Lance repository is embedded as a **git submodule** at `Lance/` — no
separate clone or `PYTHONPATH` export is required.

---

## Features

| Node | Task |
|------|------|
| **Lance: Load Model** | Load LLM + ViT + VAE into GPU memory |
| **Lance: Text → Image** | Generate an image from a text prompt |
| **Lance: Edit Image** | Edit an image with a text instruction |
| **Lance: Text → Video** | Generate up to 121 frames from a text prompt |
| **Lance: Edit Video** | Edit a video clip with a text instruction |
| **Lance: Image Understanding** | VQA / captioning for images |
| **Lance: Video Understanding** | VQA / captioning for video clips |
| **Lance: Unload Model** | Free GPU VRAM by removing the cached pipeline |

---

## Requirements

### Hardware
- GPU with **≥ 40 GB VRAM** (required by Lance)
- CUDA 12.4+

### Software
- Python 3.10+
- ComfyUI (latest)
- Git (for submodule init)

---

## Installation

### 1. Clone with submodule

```bash
cd ComfyUI/custom_nodes

git clone --recurse-submodules https://github.com/your-repo/comfyui-lance-nodes.git
```

If you already cloned without `--recurse-submodules`:

```bash
cd comfyui-lance-nodes
git submodule update --init --recursive
```

ComfyUI Manager will also call `install.py` automatically on first load, which
runs `git submodule update --init --recursive` and installs Lance's Python
dependencies.

### 2. Place model weights

Model components sit in three ComfyUI model directories:

| Component | Directory | File(s) |
|-----------|-----------|---------|
| Main checkpoint + tokenizer | `models/LLM/` | `*.safetensors` |
| ViT (vision encoder) | `models/clip_vision/` | `*.safetensors` |
| VAE | `models/vae/` | `*.safetensors` |

#### Option A — Self-contained single files (recommended)

Download the pre-packed files from
**[anr2me/bytedance_lance](https://huggingface.co/anr2me/bytedance_lance)**
on HuggingFace. Every companion JSON (config, tokenizer, etc.) is embedded
in the safetensors header — no sidecar files needed:

```
ComfyUI/models/LLM/
  lance_3b_comfyui.safetensors          ← image tasks
  lance_3b_video_comfyui.safetensors    ← video tasks

ComfyUI/models/clip_vision/
  qwen2_5_vl_vit_comfyui.safetensors

ComfyUI/models/vae/
  wan2.2_vae.safetensors
```

#### Option B — Original files + sidecar JSONs

Download from
[bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)
and place companion files beside the weights:

```
ComfyUI/models/LLM/lance_3b/
  llm_config.json
  model.safetensors        ← (or ema.safetensors)
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.json  /  merges.txt

ComfyUI/models/clip_vision/lance_vit/
  config.json
  vit.safetensors

ComfyUI/models/vae/
  wan_vae.safetensors
```

> The nodes automatically detect whether companion files are present on disk
> or need to be extracted from the safetensors metadata header.

---

## Node Reference

### Lance: Load Model

| Input | Type | Description |
|-------|------|-------------|
| `llm_file` | dropdown | `.safetensors` in `models/LLM/` — main checkpoint + tokenizer |
| `vit_file` | dropdown | `.safetensors` in `models/clip_vision/` — ViT weights |
| `vae_file` | dropdown | `.safetensors` in `models/vae/` |
| `device` | `cuda` / `cpu` | Inference device |
| `dtype` | `bf16` / `fp16` | Compute precision (bf16 recommended) |

**Output:** `LANCE_PIPELINE` — pass to every other Lance node.

---

### Lance: Text → Image

| Input | Default | Description |
|-------|---------|-------------|
| `pipeline` | — | From *Load Model* |
| `prompt` | — | Text description of the desired image |
| `width` / `height` | 768 | Output resolution (multiples of 64) |
| `num_steps` | 30 | Denoising steps |
| `cfg_scale` | 4.0 | Classifier-Free Guidance strength |
| `timestep_shift` | 3.5 | Flow-matching schedule shift |
| `seed` | 42 | Reproducibility seed |

**Output:** `IMAGE`

---

### Lance: Edit Image

| Input | Default | Description |
|-------|---------|-------------|
| `pipeline` | — | From *Load Model* |
| `image` | — | Source image (ComfyUI IMAGE) |
| `instruction` | — | Natural-language edit instruction |
| `num_steps` | 30 | Denoising steps |
| `cfg_scale` | 4.0 | CFG scale |
| `timestep_shift` | 3.5 | Schedule shift |
| `seed` | 42 | Seed |

**Output:** `IMAGE`

---

### Lance: Text → Video

| Input | Default | Description |
|-------|---------|-------------|
| `pipeline` | — | From *Load Model* |
| `prompt` | — | Text description of the video |
| `width` / `height` | 832 / 480 | Frame resolution |
| `num_frames` | 50 | Frames to generate (max 121) |
| `num_steps` | 30 | Denoising steps |
| `cfg_scale` | 4.0 | CFG scale |
| `timestep_shift` | 3.5 | Schedule shift |
| `seed` | 42 | Seed |

**Output:** `IMAGE` list (one tensor per frame). Connect to a *Video Combine*
or *Preview Video* node.

---

### Lance: Edit Video

| Input | Default | Description |
|-------|---------|-------------|
| `pipeline` | — | From *Load Model* |
| `frames` | — | Input video frames (IMAGE list or batch) |
| `instruction` | — | Edit instruction |
| `max_frames` | 50 | Max frames to process |
| `num_steps` | 30 | Denoising steps |
| `cfg_scale` | 4.0 | CFG scale |
| `timestep_shift` | 3.5 | Schedule shift |
| `seed` | 42 | Seed |

**Output:** `IMAGE` list

---

### Lance: Image Understanding

| Input | Default | Description |
|-------|---------|-------------|
| `pipeline` | — | From *Load Model* |
| `image` | — | Input image |
| `question` | — | Question or captioning prompt |
| `max_new_tokens` | 256 | Max answer length |

**Output:** `STRING`

---

### Lance: Video Understanding

| Input | Default | Description |
|-------|---------|-------------|
| `pipeline` | — | From *Load Model* |
| `frames` | — | Video frames (IMAGE list or batch) |
| `question` | — | Question or captioning prompt |
| `max_new_tokens` | 256 | Max answer length |
| `sample_frames` | 8 | Evenly-spaced frames to sample from the clip |

**Output:** `STRING`

---

### Lance: Unload Model

Removes the pipeline from the in-memory cache and calls `torch.cuda.empty_cache()`.

---

## Repository layout

```
comfyui-lance-nodes/
├── .gitmodules             ← points to bytedance/Lance as submodule at Lance/
├── Lance/                  ← git submodule (bytedance/Lance)
├── lance_backend/
│   ├── __init__.py
│   ├── generation.py       ← T2I / image-edit / T2V / video-edit wrappers
│   └── understanding.py    ← image + video VQA / captioning wrappers
├── __init__.py             ← ComfyUI entry point; injects Lance/ onto sys.path
├── install.py              ← run by ComfyUI Manager; inits submodule + pip deps
├── nodes.py                ← all 8 ComfyUI node class definitions
└── README.md
```

## Example workflows

### Text-to-Image
```
[Lance: Load Model] ──pipeline──► [Lance: Text → Image] ──image──► [Preview Image]
```

### Image Editing
```
[Load Image] ──image──┐
                       ├──► [Lance: Edit Image] ──image──► [Preview Image]
[Lance: Load Model] ───┘
```

### Text-to-Video
```
[Lance: Load Model] ──pipeline──► [Lance: Text → Video] ──frames──► [Video Combine] ──► [Preview Video]
```

### Video VQA
```
[Load Video] ──frames──┐
                        ├──► [Lance: Video Understanding] ──answer──► [Show Text]
[Lance: Load Model] ────┘
```

---

## License

Apache-2.0 — same license as the upstream [Lance](https://github.com/bytedance/Lance) project.
