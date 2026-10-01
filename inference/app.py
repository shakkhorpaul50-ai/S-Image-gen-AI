"""MicroDiffusion 0.15B inference Space (Gradio SDK, ZeroGPU-ready).

Serves the custom Q3 model over plain HTTP for the S-Image-gen-AI web app:
  GET  /healthz
  POST /generate  {prompt, seed?, steps?, cfg?} -> {image_b64, seed, model, size, ms}
  POST /edit      {image_b64, prompt, strength?, steps?, cfg?, seed?} -> {...}

Memory modes (STREAMING env, default "1"):
  1 = streaming: packed Q3 resident (~45MB), one fp32 layer materialised at a
      time  -> ~100-130MB total. Use on tiny hosts (<220MB budget).
  0 = full fp16/fp32 materialisation -> ~200MB+ total. Use with RAM to spare.

A minimal Gradio UI is mounted at /ui for human testing.
GPU: inference core is @spaces.GPU-decorated (no-op outside ZeroGPU).
Auth: if MODEL_API_KEY env is set, clients must send it as X-Api-Key header.
Weights: WEIGHTS_URL + TOKENIZER_URL (e.g. HuggingFace Hub resolve URLs),
         downloaded once at startup; or WEIGHTS_PATH / TOKENIZER_PATH for local files.
"""
import base64
import io
import os
import random
import threading
import time
import urllib.request

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ---- optional ZeroGPU (no-op anywhere else) -------------------------------
try:
    import spaces

    def _gpu(fn):
        return spaces.GPU(fn)

    _HAS_SPACES = True
except Exception:  # local dev / plain CPU: decorator does nothing
    def _gpu(fn):
        return fn

    _HAS_SPACES = False

# ---- optional Gradio UI (platform provides it; skip locally) -------------
try:
    import gradio as gr

    _HAS_GRADIO = True
except Exception:
    _HAS_GRADIO = False

from microdiffusion_model import (
    Q3Streamer,
    apply_rest_state,
    build_models,
    load_packed_bundle,
    load_service_bundle,
    sample_img2img,
    sample_txt,
)

MODEL_NAME = "microdiffusion-0.15b-q3"
OUT_SIZE = 256
STREAMING = os.environ.get("STREAMING", "1") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_lock = threading.Lock()
B = None       # bundle dict (full mode) or packed-bundle dict (streaming mode)
STREAM = None  # Q3Streamer attached per model in streaming mode


def _download(url, dest):
    if os.path.exists(dest):
        print(f"cache hit: {dest}", flush=True)
        return dest
    print(f"downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)", flush=True)
    return dest


def _resolve_paths():
    os.makedirs("./models", exist_ok=True)
    w = os.environ.get("WEIGHTS_PATH")
    t = os.environ.get("TOKENIZER_PATH")
    if not w:
        url = os.environ.get("WEIGHTS_URL", "")
        if not url:
            raise RuntimeError("Set WEIGHTS_URL (or WEIGHTS_PATH) to the Q3 .safetensors file.")
        w = _download(url, "./models/microdiffusion150m_q3.safetensors")
    if not t:
        url = os.environ.get("TOKENIZER_URL", "")
        if not url:
            raise RuntimeError("Set TOKENIZER_URL (or TOKENIZER_PATH) to tokenizer.json.")
        t = _download(url, "./models/tokenizer.json")
    return w, t


def _new_seed():
    return random.SystemRandom().randint(1, 1_000_000_000)


print("loading bundle ...", flush=True)
_WPATH, _TPATH = _resolve_paths()
if STREAMING:
    B = load_packed_bundle(_WPATH, _TPATH)
    dit, te, vae = build_models(B["cfg"], len(B["vocab"]), device="cpu")
    apply_rest_state({"dit": dit, "te": te, "vae": vae},
                     {k: v for k, v in B["rest"].items()})
    _STREAMERS = {}
    for mname, model in (("dit", dit), ("te", te), ("vae", vae)):
        st = Q3Streamer(
            {k: v for k, v in B["packed"].items() if k.startswith(mname + ".")},
            {"tensors": {k: v for k, v in B["meta"]["tensors"].items() if k.startswith(mname + ".")}},
            device="cpu")
        st.attach(model)
        _STREAMERS[mname] = st
    B.update({"dit": dit, "te": te, "vae": vae, "unique_m": (
        sum(v.numel() for v in B["rest"].values()) +
        sum(i["n"] for i in B["meta"]["tensors"].values() if i["q"] == "q3")) / 1e6})
    print(f"streaming mode: packed resident, per-layer fp32 materialisation", flush=True)
else:
    B = load_service_bundle(_WPATH, _TPATH, device="cpu")


def _to_device():
    # HF ZeroGPU guidance: place models on cuda at module level (emulation
    # handles it before a real GPU attaches); plain-CPU boxes fall back.
    global DEVICE
    try:
        B["dit"].to("cuda")
        B["te"].to("cuda")
        B["vae"].to("cuda")
        DEVICE = "cuda"
    except Exception as e:
        print(f"cuda placement failed ({e}); using cpu", flush=True)
        DEVICE = "cpu"


_to_device()
print(f"ready: {B['unique_m']:.1f}M unique params on {DEVICE} "
      f"(streaming={STREAMING}, spaces={_HAS_SPACES}, gradio={_HAS_GRADIO})", flush=True)


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


@_gpu
def _run_txt(prompt, seed, steps, cfg):
    imgs = sample_txt(B["dit"], B["te"], B["vae"], B["stoi"], [prompt],
                      B["scale"], steps=steps, cfg=cfg, seed=seed, device=DEVICE)
    return imgs[0]


@_gpu
def _run_img(pil_img, prompt, strength, steps, cfg, seed):
    imgs = sample_img2img(B["dit"], B["te"], B["vae"], B["stoi"], pil_img, prompt,
                          B["scale"], strength=strength, steps=steps, cfg=cfg,
                          seed=seed, device=DEVICE)
    return imgs[0]


def _check_key(x_api_key: str | None):
    need = os.environ.get("MODEL_API_KEY", "")
    if need and x_api_key != need:
        raise HTTPException(status_code=401, detail="bad api key")


app = FastAPI(title="MicroDiffusion 0.15B inference")


class GenReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int = 0
    steps: int = 24
    cfg: float = 5.0


class EditReq(BaseModel):
    image_b64: str
    prompt: str = Field(min_length=1, max_length=2000)
    strength: float = 0.6
    steps: int = 24
    cfg: float = 5.0
    seed: int = 0


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE,
            "streaming": STREAMING,
            "unique_m": round(B["unique_m"], 1), "spaces_gpu": _HAS_SPACES}


@app.post("/generate")
def generate(r: GenReq, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    seed = r.seed or _new_seed()
    t0 = time.time()
    with _lock:
        try:
            img = _run_txt(r.prompt, int(seed), max(4, min(r.steps, 64)), float(r.cfg))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}")
    ms = int((time.time() - t0) * 1000)
    return JSONResponse({"image_b64": _pil_to_b64(img), "seed": int(seed),
                         "model": MODEL_NAME, "size": f"{OUT_SIZE}x{OUT_SIZE}", "ms": ms})


@app.post("/edit")
def edit(r: EditReq, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    try:
        pil_img = _b64_to_pil(r.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="image_b64 is not a valid image")
    seed = r.seed or _new_seed()
    t0 = time.time()
    with _lock:
        try:
            img = _run_img(pil_img, r.prompt, float(min(max(r.strength, 0.02), 1.0)),
                           max(4, min(r.steps, 64)), float(r.cfg), int(seed))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}")
    ms = int((time.time() - t0) * 1000)
    return JSONResponse({"image_b64": _pil_to_b64(img), "seed": int(seed),
                         "model": MODEL_NAME, "size": f"{OUT_SIZE}x{OUT_SIZE}", "ms": ms})


if _HAS_GRADIO:
    with gr.Blocks(title="MicroDiffusion 0.15B") as demo:
        gr.Markdown("# MicroDiffusion 0.15B (custom model API demo)")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="a cat, ukiyo-e woodblock print")
            seed = gr.Number(label="Seed (0 = random)", value=0, precision=0)
        btn = gr.Button("Generate")
        out = gr.Image(label="Output")
        btn.click(lambda p, s: _run_txt(p, int(s or 0), 24, 5.0),
                  inputs=[prompt, seed], outputs=out)
    app = gr.mount_gradio_app(app, demo, path="/ui")
