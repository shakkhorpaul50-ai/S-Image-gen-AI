# Training: MicroDiffusion 0.15B (Kaggle, ~3h on T4)

From-scratch text-to-image + image-to-image training, targeting **~150M nominal
params / ~90M stored params** with **runtime weights under 220MB** (via Q3 +
streaming dequant).

## Files

- `kaggle_train_150m.py` — the notebook cells (`# %%` separated; paste into Kaggle or open as a notebook in VS Code).
- `kaggle_train_150m.ipynb` — same content, upload directly to Kaggle via File → Import Notebook.
- `make_ipynb.py` — regenerates the `.ipynb` from the `.py` after edits (stdlib only): `python make_ipynb.py`.
- `microdiffusion_model.py` — canonical torch-only module: arch (DiT 8-unique tied / TE 6-layer / VAE), word tokenizer, Q3 quant/dequant, `Q3Streamer` (streaming per-layer dequant for <220MB inference), bundle IO, rectified-flow samplers. Imported by inference hosts; Cell 14 cross-checks it against the trained checkpoint.

## Architecture (~148M nominal / ~90M stored)

| Component | Config | Unique | Counted |
|---|---|---|---|
| DiT | hidden 576, 8 unique + 8 tied (16 effective), 12 heads, patch 2 | ~59M | ~117M |
| Text encoder | 6 layers, hidden 448, 8 heads, word vocab | ~23M | ~23M |
| VAE | ch (64,128,256), 3 downs, 4 latent channels | ~8M | ~8M |

## Datasets (Kaggle only — attach both via Add Data)

1. Photos: https://www.kaggle.com/datasets/adityajn105/flickr30k (`captions.txt` + `Images/`)
2. Art: https://www.kaggle.com/datasets/trungit/wikiart30k (per-style folders; captions auto-derived as `"<subject>, <style> painting"`, stratified 12k sample)

## Pipeline (15 cells, ~3h)

VAE (2 epochs) → cache latents → tied-DiT + text encoder (18k steps rectified flow,
~30% faster/step than the 12-block design) → txt2img + img2img sampling →
Q3_K_S-style quant → export `.hk` (or `.safetensors` fallback) + `tokenizer.json`
→ verify → **Cell 14**: packaging (bundle + Q3Streamer self-test + smoke image) +
optional HuggingFace Hub upload (`HF_TOKEN` + `HF_REPO` env vars).

## Runtime weight budget (<220MB)

| Mode | Weights | Total est. |
|---|---|---|
| Q3 file on disk | ~45MB | — |
| Streaming inference (packed resident + one fp32 layer + activations) | ~60MB | **~100–130MB ✅** |
| Full fp16 dequant (fallback) | ~180MB | ~210–230MB (borderline) |

## Notes

- Fresh architecture (8≠12 blocks): cannot reuse 0.25B weights — train from scratch.
- Same data + tokenizer pipeline keeps prompt compatibility.
- Smaller model = lower quality ceiling; 18k steps partially compensates. Quality scales with steps — resume past 18k if quota allows.
