# S-Image-gen-AI (MicroDiffusion 0.15B)

Free AI image generator with a ChatGPT-style threads UI: ASP.NET Core MVC +
own 0.15B diffusion model run in-process via ONNX Runtime INT8
(~90MB weights, ~350–400MB total on Render free) + Neon Postgres + Cloudinary
storage. 50 free generations per user per day.

## Layout

- `training/` — from-scratch Kaggle pipeline (Flickr30k + WikiArt-30k, ~3h on T4):
  `kaggle_train_150m.py`/`.ipynb`, canonical `microdiffusion_model.py`
  (arch + Q3 quant/dequant + `Q3Streamer` + samplers), packaging Cell 14,
  export Cell 15 (ONNX quartet + static INT8 + Hub upload).
- `inference/` — HuggingFace Space code (FastAPI + Gradio, `@spaces.GPU`),
  usable as the `space` backend alternative.
- `src/SImageGen/` — the web app (chat UI, backends, Dockerfile, `render.yaml`).

## How it works

- **Chat** (`/Chat`): every message generates an image (text prompt, or attached
  photo = img2img) inside a persistent thread (sidebar history, quota meter).
- **Backend selection** (`IMAGE_BACKEND`, default `auto`): in-process ONNX model,
  with silent failover to the Pollinations gateway on timeout/failure
  (`LOCAL_TIMEOUT_SECONDS`, default 360). `local` = ONNX only,
  `space` = remote custom endpoint, `pollinations` = gateway only.
- Prompt-hash cache serves repeats instantly without consuming quota.

## Free-tier setup (Render + Neon + Cloudinary + Pollinations)

1. **Neon** (https://neon.tech): create project → copy the connection string → set as `DATABASE_URL`. Tables auto-migrate on app startup (fail-open if Neon naps).
2. **Cloudinary** (https://cloudinary.com, free 25 GB): Dashboard → copy `CLOUDINARY_URL` (`cloudinary://key:secret@cloud`).
3. **Pollinations** (https://enter.pollinations.ai/keys): free API key → `POLLINATIONS_API_KEY` (fallback backend).
4. **ONNX weights**: already released at `model-150m-v1` on this repo's Releases page
   (INT8 quartet ~88MB + tokenizer). The app downloads them by default; override with
   `ONNX_BASE_URL` to use another source. Downloaded once into the container on first generation.
5. **Render** (https://render.com): New → Web Service (manual, Docker, free, root dir `src/SImageGen`) → set env vars → Deploy. Free plan sleeps after 15 min idle; first request wakes it (~30–60 s). `/healthz` reports backend mode + build SHA.

## Speed notes (honest)

- Local ONNX on 0.1 CPU: minutes per image (queue + polling absorb it); past `LOCAL_TIMEOUT_SECONDS` it fails over to the gateway in seconds.
- Repeat of same prompt+seed: instant (cache).
- This app generates images (diffusion); "tokens/sec" is an LLM metric and doesn't apply. The 77-token prompt encodes in milliseconds.

## Roadmap

- Style presets, public gallery, admin panel.
- Register the custom model as a Pollinations community model (alternative to self-hosting).
