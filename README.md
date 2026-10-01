# S-Image-gen-AI (MicroDiffusion 0.15B)

Free AI image generator: ASP.NET Core MVC + own 0.15B diffusion model
(Q3 weights ~45MB, streaming runtime ~100–130MB) + Neon Postgres + Cloudinary
storage. 50 free generations per user per day.

## Layout

- `training/` — from-scratch Kaggle pipeline (Flickr30k + WikiArt-30k, ~3h on T4):
  `kaggle_train_150m.py`/`.ipynb`, canonical `microdiffusion_model.py`
  (arch + Q3 quant/dequant + `Q3Streamer` + samplers), packaging Cell 14.
- `inference/` — HuggingFace Space code (FastAPI `/generate` + `/edit` + `/healthz`,
  Gradio demo at `/ui`, `@spaces.GPU` with CPU fallback, `STREAMING=1` default).
- `src/SImageGen/` — the web app (controllers, services, views, Dockerfile, `render.yaml`).

## How it works

- **Generate** (`/Generate`): prompt (+ optional source image for img2img) → quota check (50/day, UTC) → prompt-hash cache lookup → background queue → image backend → PNG uploaded to Cloudinary → gallery.
- **Backend selection** (`IMAGE_BACKEND`, default `auto`): own 0.15B model via `INFERENCE_URL` when set, otherwise the Pollinations gateway. In `auto` mode a dead/unreachable custom host **silently fails over** to the gateway (`local` = custom only, `pollinations` = gateway only).
- **Runtime weights <220MB**: the inference host keeps Q3 packed (~45MB) resident and materialises one fp32 layer at a time (`Q3Streamer`).

## Free-tier setup (Render + Neon + Cloudinary + Pollinations)

1. **Neon** (https://neon.tech): create project → copy the connection string → set as `DATABASE_URL`. Tables auto-migrate on app startup (fail-open if Neon naps).
2. **Cloudinary** (https://cloudinary.com, free 25 GB): Dashboard → copy `CLOUDINARY_URL` (`cloudinary://key:secret@cloud`).
3. **Pollinations** (https://enter.pollinations.ai/keys): free API key → `POLLINATIONS_API_KEY` (fallback backend).
4. **Custom model** (optional, recommended): run `training` Cell 14 with `HF_TOKEN`+`HF_REPO` → create the Gradio Space from `inference/` → set Space `WEIGHTS_URL`/`TOKENIZER_URL` (+ optional `MODEL_API_KEY`) → set web app `INFERENCE_URL` (+ `INFERENCE_KEY`) → generations serve `microdiffusion-0.15b-q3`.
5. **Render** (https://render.com): New → Web Service (manual, Docker, free) → select this repo → set env vars → Deploy. Free plan sleeps after 15 min idle; first request wakes it (~30–60 s). `/healthz` reports backend mode + build SHA.

## Speed notes (honest)

- Custom model on free CPU: minutes per image (queue + polling absorb it); on ZeroGPU Space: seconds.
- Repeat of same prompt+seed: instant (cache).
- This app generates images (diffusion); "tokens/sec" is an LLM metric and doesn't apply. The 77-token prompt encodes in milliseconds.

## Roadmap

- Chat-style threaded UI (conversations sidebar + composer).
- Style presets, public gallery, admin panel.
