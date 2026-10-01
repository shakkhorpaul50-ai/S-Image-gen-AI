---
title: MicroDiffusion 0.15B Inference
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
pinned: false
license: mit
short_description: Q3 tiny DiT txt2img + img2img API, streaming runtime under 220MB
---

# MicroDiffusion 0.15B inference Space

Custom text-to-image + image-to-image API for S-Image-gen-AI.
Runs the from-scratch 0.15B tied-DiT model (Q3 weights, ~45MB) with PyTorch.

## Endpoints (served by the FastAPI `app` in `app.py`)

- `GET /healthz` → `{ok, model, device, streaming, unique_m}`
- `POST /generate` `{prompt, seed?, steps?, cfg?}` → `{image_b64, seed, model, size, ms}`
- `POST /edit` `{image_b64, prompt, strength?, steps?, cfg?, seed?}` → `{...}`
- `/ui` → minimal Gradio demo for human testing

If `MODEL_API_KEY` is set, clients must send it as the `X-Api-Key` header.

## Memory modes (`STREAMING` env, default `"1"`)

- `"1"` (default): packed Q3 resident (~45MB), one fp32 layer at a time → **~100–130MB total**. For tiny hosts and the <220MB budget.
- `"0"`: full materialisation → ~200MB+. Only with RAM to spare.

## Setup (website, ~5 min)

1. Create a Space: **New Space** → name (e.g. `microdiffusion-015b`) → SDK **Gradio** → **Create**.
2. Upload `app.py`, `requirements.txt`, `microdiffusion_model.py` (from this repo's `training/`), and this `README.md`.
3. Space **Settings → Variables and secrets**: add
   - `WEIGHTS_URL` = `https://huggingface.co/<you>/microdiffusion-015b-q3/resolve/main/microdiffusion150m_q3.safetensors`
   - `TOKENIZER_URL` = `https://huggingface.co/<you>/microdiffusion-015b-q3/resolve/main/tokenizer.json`
   - `STREAMING` = `1`
   - (optional) `MODEL_API_KEY` = a shared secret (same value goes in the web app's `INFERENCE_KEY`).
4. Wait for build → **Running**. Open `<space-url>/healthz` → `{"ok":true,...}`.
5. Optional speed: **Settings → Hardware → ZeroGPU** (free quota; needs verified email + 30+ day old account) for seconds-per-image instead of minutes on CPU.

## Files

- `app.py` — FastAPI + Gradio mount + `@spaces.GPU` inference core (CPU fallback included).
- `requirements.txt` — torch (plain CUDA-capable build), safetensors, fastapi, uvicorn, pillow, numpy, huggingface_hub.
- `microdiffusion_model.py` — copy from `../training/` (canonical arch + Q3 dequant + `Q3Streamer` + samplers).

## Weights

Upload once from the Kaggle packaging cell (Cell 14) to a Hub model repo, e.g.
`<you>/microdiffusion-015b-q3` containing `microdiffusion150m_q3.safetensors`,
`tokenizer.json`, `microdiffusion_model.py`. The Space downloads them at startup
and (in streaming mode) never materialises the full fp32 model.
