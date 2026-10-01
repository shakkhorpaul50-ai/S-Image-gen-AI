# %% [markdown]
# # MicroDiffusion 0.15B — Train tiny txt2img + img2img on Kaggle (~3h on T4)
# **Datasets (Kaggle only):**
# - Photos: https://www.kaggle.com/datasets/adityajn105/flickr30k (`captions.txt` + `Images/`, ~31k photos / ~159k captions)
# - Art: https://www.kaggle.com/datasets/trungit/wikiart30k (~30k paintings in per-style folders; captions auto-derived as "<subject>, <style> painting")
# Attach both via Add Data, or the notebook downloads them with the Kaggle API.
# Pipeline: train tiny VAE (2 epochs) → cache latents → train tied-DiT + text encoder (18k steps, rectified flow) → sample txt2img + img2img → Q3_K_S-style quant → export `.hk` → package service bundle.
# Target: ~150M nominal params / ~90M stored params → ~45MB file → ~100-130MB streaming runtime (<220MB).

# %% 1. Setup: GPU check + installs
import os, sys, math, json, random, subprocess
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
torch.backends.cudnn.benchmark = True
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hknt", "safetensors"], check=False)

# %% 2. Datasets: Flickr30k (photos) + WikiArt-30k (art styles) — Kaggle only
import glob
import re
import pandas as pd
from PIL import Image

def find_dataset():
    for base in ["/kaggle/input"]:
        hits = glob.glob(os.path.join(base, "**", "captions.txt"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None

inp = find_dataset()
if inp is None:  # fallback: download with Kaggle API (works inside Kaggle notebooks)
    inp = "/kaggle/working/data"
    os.makedirs(inp, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "adityajn105/flickr30k",
                    "-p", inp, "--unzip"], check=True)
print("flickr dir:", inp)
cap_csv = glob.glob(os.path.join(inp, "**", "captions.txt"), recursive=True)[0]
img_dir = None
for cand in ["Images", "images", "flickr30k_images", "Flickr30k"]:
    p = os.path.join(inp, cand)
    if os.path.isdir(p):
        img_dir = p
        break
if img_dir is None:  # search any dir with many jpgs
    for root, _, files in os.walk(inp):
        if sum(f.lower().endswith(".jpg") for f in files) > 1000:
            img_dir = root
            break
print("captions:", cap_csv, "\nimages:", img_dir)
df = pd.read_csv(cap_csv)  # columns: image, caption
print(df.head(3), "\nrows:", len(df))

def resolve(name):
    for p in (os.path.join(img_dir, name), os.path.join(img_dir, os.path.basename(name))):
        if os.path.exists(p):
            return p
    return None

df["path"] = df["image"].apply(resolve)
df = df[df["path"].notna()].reset_index(drop=True)
print("usable photo pairs:", len(df), "| unique photos:", df["image"].nunique())

# ---- second dataset: art (style diversity). Any /kaggle/input images NOT under the flickr dir count as art;
# style = parent folder name, subject = cleaned filename -> caption "<subject>, <style> painting".
ART_SLUG = "trungit/wikiart30k"
ART_MAX = 12000  # keep in sync with CFG.art_max below
all_jpgs = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
    all_jpgs += glob.glob(os.path.join("/kaggle/input", "**", ext), recursive=True)
art_jpgs = [p for p in all_jpgs if not p.startswith(os.path.abspath(inp) + os.sep)]
if not art_jpgs:  # download fallback
    subprocess.run(["kaggle", "datasets", "download", "-d", ART_SLUG,
                    "-p", "/kaggle/working/data", "--unzip"], check=True)
    all_jpgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        all_jpgs += glob.glob(os.path.join("/kaggle/working/data", "**", ext), recursive=True)
    art_jpgs = [p for p in all_jpgs if not p.startswith(os.path.abspath(inp) + os.sep)]
print("art images found:", len(art_jpgs))

GENERIC_DIRS = {"images", "image", "data", "train", "test", "val", "wikiart", "art", "resized", "full", "files"}

def style_of(path):
    d = os.path.dirname(path)
    while d and os.path.basename(d).lower() in GENERIC_DIRS:
        d = os.path.dirname(d)
    return os.path.basename(d).replace("_", " ").replace("-", " ").strip().lower() or "painting"

def subject_of(path):
    b = os.path.splitext(os.path.basename(path))[0]
    b = re.sub(r"[_\-]+", " ", b)
    b = re.sub(r"\b(19|20)\d{2}\b", "", b)   # years
    b = re.sub(r"\(\d+\)", "", b)             # (1) copies
    return re.sub(r"\s+", " ", b).strip().lower()

by_style = {}
for p in art_jpgs:
    by_style.setdefault(style_of(p), []).append(p)
print("styles found:", len(by_style))
per = max(1, ART_MAX // max(1, len(by_style)))
picked = []
for s, lst in sorted(by_style.items()):
    random.Random(7).shuffle(lst)
    picked += [(p, s) for p in lst[:per]]
random.Random(7).shuffle(picked)
picked = picked[:ART_MAX]

art_rows = []
for p, s in picked:
    subj = subject_of(p)
    art_rows.append({"path": p, "caption": f"{subj}, {s} painting" if subj else f"a {s} painting", "src": "art"})
art_df = pd.DataFrame(art_rows)
print("art pairs:", len(art_df), "| e.g.:", art_df["caption"].iloc[0] if len(art_df) else None)

all_df = pd.concat([df[["path", "caption"]].assign(src="photo"), art_df], ignore_index=True)
print("combined pairs:", len(all_df), "| unique images:", all_df["path"].nunique())

# %% 3. Config + word tokenizer (built from captions, zero downloads)
from collections import Counter
from dataclasses import dataclass, asdict

@dataclass
class Cfg:
    img_size: int = 256
    latent_ch: int = 4
    vae_ch: tuple = (64, 128, 256)
    te_dim: int = 448; te_layers: int = 6; te_heads: int = 8; te_ff: int = 1792
    te_vocab_cap: int = 30000
    max_len: int = 77
    dit_dim: int = 576; dit_heads: int = 12; dit_unique: int = 8; dit_passes: int = 2
    dit_patch: int = 2; dit_mlp: int = 2304; dit_grid: int = 16
    batch: int = 16; vae_batch: int = 24
    lr_vae: float = 2e-4; lr_dit: float = 1e-4
    vae_epochs: int = 2; dit_steps: int = 18000; ckpt_every: int = 2000
    art_max: int = 12000
    cfg_drop: float = 0.1; seed: int = 7

CFG = Cfg()
torch.manual_seed(CFG.seed); random.seed(CFG.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.cuda.is_available()
print("device:", device, "| cfg:", asdict(CFG))

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
cnt = Counter()
for cap in all_df["caption"].astype(str):
    cnt.update(cap.lower().split())
VOCAB = [PAD, BOS, EOS, UNK] + [w for w, c in cnt.most_common(CFG.te_vocab_cap - 4) if c >= 2]
stoi = {w: i for i, w in enumerate(VOCAB)}
print("vocab size:", len(VOCAB))

def encode_text(cap, max_len=CFG.max_len):
    ids = [stoi[BOS]] + [stoi.get(w, stoi[UNK]) for w in str(cap).lower().split()[:max_len - 2]] + [stoi[EOS]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids

# %% 4. Datasets & loaders (VAE trains on unique images only -> faster)
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

tfm = T.Compose([T.Resize((CFG.img_size, CFG.img_size)), T.ToTensor(),
                 T.Normalize([0.5] * 3, [0.5] * 3)])  # [-1, 1]

class CapDS(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return tfm(img), torch.tensor(encode_text(r["caption"]), dtype=torch.long)

train_ld = DataLoader(CapDS(df), batch_size=CFG.batch, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)
xb, tb = next(iter(train_ld))
print("batch:", xb.shape, tb.shape)

vae_df = all_df.drop_duplicates("path").reset_index(drop=True)
vae_ld = DataLoader(CapDS(vae_df), batch_size=CFG.vae_batch, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
print("vae images (unique photo+art):", len(vae_df))

# %% 5. Tiny VAE (~8M)
import torch.nn as nn
import torch.nn.functional as F

class ResB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1),
                               nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))
    def forward(self, x):
        return x + self.b(x)

class VAE(nn.Module):
    def __init__(self, ch=(64, 128, 256), z=4):
        super().__init__()
        c0, c1, c2 = ch
        self.enc = nn.Sequential(
            nn.Conv2d(3, c0, 3, 1, 1), ResB(c0),
            nn.Conv2d(c0, c1, 4, 2, 1), ResB(c1), ResB(c1),
            nn.Conv2d(c1, c2, 4, 2, 1), ResB(c2), ResB(c2),
            nn.Conv2d(c2, c2, 4, 2, 1), ResB(c2),
        )
        self.mu = nn.Conv2d(c2, z, 1)
        self.lv = nn.Conv2d(c2, z, 1)
        self.dec_in = nn.Conv2d(z, c2, 1)
        self.dec = nn.Sequential(
            ResB(c2), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, 3, 1, 1), ResB(c1), ResB(c1), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, c0, 3, 1, 1), ResB(c0), nn.Upsample(scale_factor=2, mode="nearest"),
            nn.GroupNorm(8, c0), nn.SiLU(), nn.Conv2d(c0, 3, 3, 1, 1), nn.Tanh(),
        )
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)
    def sample(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    def decode(self, z):
        return self.dec(self.dec_in(z))
    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decode(self.sample(mu, lv)), mu, lv

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# %% 6. Train VAE (~15-25 min)
vae = VAE(CFG.vae_ch, CFG.latent_ch).to(device)
print(f"VAE params: {n_params(vae) / 1e6:.1f}M")
opt = torch.optim.AdamW(vae.parameters(), lr=CFG.lr_vae)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
vae.train()
step = 0
for ep in range(CFG.vae_epochs):
    for x, _ in vae_ld:
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=AMP):
            xr, mu, lv = vae(x)
            l1 = F.l1_loss(xr, x)
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            loss = l1 + 1e-6 * kl
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % 200 == 0:
            print(f"ep{ep} step{step} l1={l1.item():.4f} kl={kl.item():.4f}")
torch.save(vae.state_dict(), "/kaggle/working/vae.pt")
print("saved /kaggle/working/vae.pt")

# %% 7. Cache latents (32x32x4) + scale factor
vae.eval()
paths = all_df["path"].unique().tolist()
lat = torch.empty(len(paths), CFG.latent_ch, CFG.dit_grid * 2, CFG.dit_grid * 2, dtype=torch.float16)
with torch.no_grad():
    bs = 32
    for i in range(0, len(paths), bs):
        chunk = []
        for name in paths[i:i + bs]:
            chunk.append(tfm(Image.open(name).convert("RGB")))
        x = torch.stack(chunk).to(device)
        with torch.amp.autocast("cuda", enabled=AMP):
            mu, _ = vae.encode(x)
        lat[i:i + len(chunk)] = mu.float().cpu().half()
        if i % 3200 == 0:
            print(i, "/", len(paths))
scale = 1.0 / lat.float().std().item()
lat = (lat.float() * scale).half()
print("latent scale:", round(scale, 4), "| latents:", tuple(lat.shape))
torch.save({"lat": lat, "paths": paths, "scale": scale}, "/kaggle/working/latents.pt")
path2idx = {p: i for i, p in enumerate(paths)}
torch.save(torch.tensor([path2idx[p] for p in all_df["path"]]), "/kaggle/working/row_img.pt")

# %% 8. Text encoder (~22M) + tied DiT (~59M stored / ~117M nominal)
class TELayer(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.a = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.m = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def forward(self, x, key_mask=None):
        n = self.n1(x)
        x = x + self.a(n, n, n, key_padding_mask=key_mask, need_weights=False)[0]
        return x + self.m(self.n2(x))

class TextEnc(nn.Module):
    def __init__(self, vocab, d=448, L=6, h=8, ff=1792, ml=77):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(ml, d) * 0.02)
        self.layers = nn.ModuleList([TELayer(d, h, ff) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
    def forward(self, ids):
        mask = (ids == 0)
        x = self.emb(ids) + self.pos[:ids.size(1)]
        for l in self.layers:
            x = l(x, mask)
        return self.ln(x), mask

def mod(x, s):
    return x * (1 + s)

class DiTBlock(nn.Module):
    def __init__(self, d, h, mlp, td):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.sa = nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.xa = nn.MultiheadAttention(d, h, kdim=td, vdim=td, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.m = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
    def forward(self, x, c, txt, tmask):
        s1, g1, s2, g2, s3, g3 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        h = mod(self.n1(x), s1)
        x = x + g1 * self.sa(h, h, h, need_weights=False)[0]
        h = mod(self.n2(x), s2)
        x = x + g2 * self.xa(h, txt, txt, key_padding_mask=tmask, need_weights=False)[0]
        h = mod(self.n3(x), s3)
        return x + g3 * self.m(h)

class DiT(nn.Module):
    def __init__(self, z=4, d=576, h=12, mlp=2304, td=448, unique=8, passes=2, p=2, grid=16):
        super().__init__()
        self.unique, self.passes = unique, passes
        self.grid, self.p, self.z = grid, p, z
        self.inp = nn.Conv2d(z, d, p, p)
        self.pos = nn.Parameter(torch.randn(grid * grid, d) * 0.02)
        self.tmlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(d, h, mlp, td) for _ in range(unique)])
        self.out_n = nn.LayerNorm(d, elementwise_affine=False)
        self.out_a = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, p * p * z)
        nn.init.zeros_(self.out_a.weight); nn.init.zeros_(self.out_a.bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def t_emb(self, t, dim=256):
        f = torch.exp(-math.log(10000) * torch.arange(0, dim, 2, device=t.device) / dim)
        e = t[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)
    def forward(self, x, t, txt, tmask, drop=0.0):
        B = x.size(0)
        if drop > 0 and self.training:
            m = (torch.rand(B, 1, 1, device=x.device) < drop)
            txt = torch.where(m, torch.zeros_like(txt), txt)
        h = self.inp(x).flatten(2).transpose(1, 2) + self.pos.unsqueeze(0)
        c = self.tmlp(self.t_emb(t))
        for _ in range(self.passes):          # 2 passes over the SAME 8 blocks = tied weights
            for b in self.blocks:
                h = b(h, c, txt, tmask)
        sh, sc = self.out_a(c).unsqueeze(1).chunk(2, dim=-1)
        h = self.out(mod(self.out_n(h), sc) + sh)
        B, T, _ = h.shape
        h = h.view(B, self.grid, self.grid, self.p, self.p, self.z)
        return h.permute(0, 5, 1, 3, 2, 4).reshape(B, self.z, self.grid * self.p, self.grid * self.p)

te = TextEnc(len(VOCAB), CFG.te_dim, CFG.te_layers, CFG.te_heads, CFG.te_ff, CFG.max_len).to(device)
dit = DiT(CFG.latent_ch, CFG.dit_dim, CFG.dit_heads, CFG.dit_mlp, CFG.te_dim,
          CFG.dit_unique, CFG.dit_passes, CFG.dit_patch, CFG.dit_grid).to(device)
blk = sum(p.numel() for b in dit.blocks for p in b.parameters())
n_dit, n_te, n_vae = n_params(dit), n_params(te), n_params(vae)
print(f"DiT stored: {n_dit/1e6:.1f}M | nominal (tied counted 2x): {(n_dit+blk)/1e6:.1f}M")
print(f"TE: {n_te/1e6:.1f}M | VAE: {n_vae/1e6:.1f}M")
print(f"TOTAL nominal: {(n_dit+blk+n_te+n_vae)/1e6:.1f}M | TOTAL stored: {(n_dit+n_te+n_vae)/1e6:.1f}M")

# %% 9. Train DiT + text encoder (rectified flow, with resume, ~2-2.5h)
class LatDS(Dataset):
    def __init__(self, lat, row_img, caps):
        self.lat, self.ri, self.caps = lat, row_img, caps
    def __len__(self):
        return len(self.caps)
    def __getitem__(self, i):
        return self.lat[self.ri[i]].float(), torch.tensor(encode_text(self.caps[i]), dtype=torch.long)

cache = torch.load("/kaggle/working/latents.pt", weights_only=False)
SCALE = cache["scale"]
ri = torch.load("/kaggle/working/row_img.pt", weights_only=False)
lds = DataLoader(LatDS(cache["lat"], ri, all_df["caption"].tolist()), batch_size=CFG.batch,
                 shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

params = list(dit.parameters()) + list(te.parameters())
opt = torch.optim.AdamW(params, lr=CFG.lr_dit)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG.dit_steps)
scaler = torch.amp.GradScaler("cuda", enabled=AMP)
os.makedirs("/kaggle/working/ckpt", exist_ok=True)
start = 1
ckpts = sorted(glob.glob("/kaggle/working/ckpt/step*.pt"))
if ckpts:  # resume latest
    ck = torch.load(ckpts[-1], map_location=device, weights_only=False)
    dit.load_state_dict(ck["dit"]); te.load_state_dict(ck["te"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start = ck["step"] + 1
    print("resumed from", ckpts[-1])
dit.train(); te.train()
it = iter(lds)
for step in range(start, CFG.dit_steps + 1):
    try:
        x0, ids = next(it)
    except StopIteration:
        it = iter(lds)
        x0, ids = next(it)
    x0, ids = x0.to(device), ids.to(device)
    t = torch.rand(x0.size(0), device=device)
    eps = torch.randn_like(x0)
    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * eps
    vt = eps - x0
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=AMP):
        txt, tm = te(ids)
        vp = dit(xt, t, txt, tm, drop=CFG.cfg_drop)
        loss = F.mse_loss(vp, vt)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    if step % 200 == 0:
        print(f"step {step}/{CFG.dit_steps} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}")
    if step % CFG.ckpt_every == 0:
        torch.save({"step": step, "dit": dit.state_dict(), "te": te.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict()},
                   f"/kaggle/working/ckpt/step{step}.pt")
        for old in sorted(glob.glob("/kaggle/working/ckpt/step*.pt"))[:-2]:  # keep last 2 (~2GB each)
            os.remove(old)
torch.save({"dit": dit.state_dict(), "te": te.state_dict(), "vocab": VOCAB,
            "scale": SCALE, "cfg": asdict(CFG)}, "/kaggle/working/dit_te.pt")
print("saved /kaggle/working/dit_te.pt")

# %% 10. Sample: text -> image (photos + styles)
import matplotlib.pyplot as plt

ckpt = torch.load("/kaggle/working/dit_te.pt", map_location=device, weights_only=False)
dit.load_state_dict(ckpt["dit"]); te.load_state_dict(ckpt["te"])
SCALE = ckpt["scale"]
vae.load_state_dict(torch.load("/kaggle/working/vae.pt", map_location=device, weights_only=False))
dit.eval(); te.eval(); vae.eval()

@torch.no_grad()
def sample_txt(prompts, steps=24, cfg=5.0, seed=0):
    B = len(prompts)
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([encode_text(p) for p in prompts], device=device)
    with torch.amp.autocast("cuda", enabled=AMP):
        txt, tm = te(ids)
    null = torch.zeros_like(txt)                       # uncond: zeros, NOT masked (mask-all breaks softmax)
    nullm = torch.zeros_like(tm)
    x = torch.randn(B, CFG.latent_ch, CFG.dit_grid * 2, CFG.dit_grid * 2, generator=g, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), 1 - i * dt, device=device)
        with torch.amp.autocast("cuda", enabled=AMP):
            vc = dit(x, t, txt, tm)
            vu = dit(x, t, null, nullm)
        x = x - (vu + cfg * (vc - vu)) * dt            # Euler t: 1 -> 0
    img = vae.decode(x / SCALE).clamp(-1, 1)
    return ((img + 1) / 2).cpu()

prompts = ["a dog running in the park", "a red car on the street",
           "a woman holding an umbrella", "a cat sitting on a chair",
           "a lake at sunset, impressionism painting", "a portrait of a lady, baroque painting",
           "mountains, ukiyo-e woodblock print", "a castle, fantasy digital art"]
out = sample_txt(prompts, steps=24, cfg=5.0, seed=7)
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    r, c = divmod(i, 4)
    ax[r][c].imshow(out[i].permute(1, 2, 0).numpy()); ax[r][c].axis("off"); ax[r][c].set_title(prompts[i][:26])
plt.tight_layout(); plt.savefig("/kaggle/working/txt2img.png", dpi=100); plt.show()

# %% 11. Sample: image -> image (strength 0=keep, 1=regenerate)
@torch.no_grad()
def sample_img2img(pil_img, prompt, strength=0.6, steps=24, cfg=5.0, seed=0):
    dit.eval(); te.eval(); vae.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=AMP):
        mu, _ = vae.encode(x0)
    z0 = mu * SCALE
    if strength <= 0.01:
        img = vae.decode(z0 / SCALE).clamp(-1, 1)
        return ((img + 1) / 2).cpu()
    t0 = float(min(max(strength, 0.02), 1.0))
    eps = torch.randn(z0.shape, generator=g, device=device)
    x = (1 - t0) * z0 + t0 * eps                       # noise the latent up to t0
    ids = torch.tensor([encode_text(prompt)], device=device)
    with torch.amp.autocast("cuda", enabled=AMP):
        txt, tm = te(ids)
    null = torch.zeros_like(txt); nullm = torch.zeros_like(tm)
    n = max(1, int(steps * t0)); dt = t0 / n
    for i in range(n):                                 # denoise t0 -> 0
        t = torch.full((1,), t0 - i * dt, device=device)
        with torch.amp.autocast("cuda", enabled=AMP):
            vc = dit(x, t, txt, tm)
            vu = dit(x, t, null, nullm)
        x = x - (vu + cfg * (vc - vu)) * dt
    img = vae.decode(x / SCALE).clamp(-1, 1)
    return ((img + 1) / 2).cpu()

demo = Image.open(all_df["path"].iloc[0])
for s, p in [(0.3, "same photo"), (0.6, "the same scene at sunset"), (1.0, "a painting of a landscape")]:
    r = sample_img2img(demo, p, strength=s, steps=24, cfg=5.0, seed=7)
    Image.fromarray((r[0].permute(1, 2, 0).numpy() * 255).astype("uint8")).save(f"/kaggle/working/i2i_{s}.png")
print("saved /kaggle/working/i2i_*.png")

# %% 12. Q3_K_S-style quant (3.5 bpw) + export to .hk
import numpy as np

def quant_q3(w: torch.Tensor, group=32):
    """Symmetric grouped 3-bit quant. Returns (packed uint8, fp16 scales, shape, n)."""
    f = w.detach().float().flatten()
    n = f.numel()
    pad = (-n) % group
    if pad:
        f = torch.cat([f, torch.zeros(pad, device=f.device)])
    g = f.view(-1, group)
    s = g.abs().amax(dim=1).clamp_min(1e-8) / 3.0      # levels -3..+3
    q = torch.clamp((g / s[:, None]).round(), -3, 3).to(torch.int8) + 3  # 0..6
    q = q.flatten()
    pad8 = (-len(q)) % 8
    if pad8:
        q = torch.cat([q, torch.zeros(pad8, dtype=torch.int8, device=q.device)])
    q = q.view(-1, 8).to(torch.int64)
    pack = (q[:, 0] | q[:, 1] << 3 | q[:, 2] << 6 | q[:, 3] << 9 |
            q[:, 4] << 12 | q[:, 5] << 15 | q[:, 6] << 18 | q[:, 7] << 21).to(torch.int32)
    b = torch.stack([(pack >> 16) & 255, (pack >> 8) & 255, pack & 255], dim=1).to(torch.uint8).flatten()
    return b.cpu().numpy(), s.cpu().half().numpy(), tuple(w.shape), n

def dequant_q3(b, s, shape, n, group=32):
    b = np.asarray(b, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    pack = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
    q = np.stack([(pack >> (3 * i)) & 7 for i in range(8)], axis=1).reshape(-1).astype(np.float32) - 3.0
    s = np.asarray(s, dtype=np.float32).repeat(group)[:len(q)]
    return torch.from_numpy((q * s)[:n].reshape(shape))

MODELS = {"dit": dit, "te": te, "vae": vae}
payload, meta = {}, {"tensors": {}, "quant": "q3ks-group32-sym3.5bpw",
                     "tie": {"dit.blocks": {"unique": CFG.dit_unique, "passes": CFG.dit_passes}},
                     "budget": "220MB"}
for name, m in MODELS.items():
    for k, p in m.state_dict().items():
        key = f"{name}.{k}"
        if p.dim() >= 2 and p.numel() >= 4096:
            b, s, shape, n = quant_q3(p)
            payload[key + ".q3"] = torch.from_numpy(b)
            payload[key + ".s"] = torch.from_numpy(s)
            meta["tensors"][key] = {"q": "q3", "shape": list(shape), "n": n}
        else:
            payload[key] = p.detach().cpu().half()
            meta["tensors"][key] = {"q": "f16", "shape": list(p.shape)}
meta_json = json.dumps(meta).encode()
payload["__meta__"] = torch.from_numpy(np.frombuffer(meta_json, dtype=np.uint8).copy())
json.dump({"vocab": VOCAB, "scale": SCALE, "cfg": asdict(CFG)},
          open("/kaggle/working/tokenizer.json", "w"))

HK_PATH = "/kaggle/working/microdiffusion150m.hk"
try:
    from hk.torch import save_file as hk_save   # hknt: auto tied-weight dedup, 4KB aligned
    hk_save(payload, HK_PATH)
    print("saved .hk via hknt")
except Exception as e:
    print("hknt unavailable:", e, "-> safetensors fallback (same tensors)")
    from safetensors.torch import save_file as st_save
    HK_PATH = "/kaggle/working/microdiffusion150m.safetensors"
    st_save(payload, HK_PATH)
print(HK_PATH, f"{os.path.getsize(HK_PATH) / 1e6:.1f} MB")

# %% 13. Verify: sizes, round-trip quality, runtime estimate
w_bytes = 0
for k, v in payload.items():
    if k == "__meta__":
        continue
    w_bytes += v.numel() * v.element_size()
print(f"weights in file: {w_bytes / 1e6:.1f} MB")

test_key = next(k for k, v in meta["tensors"].items() if v["q"] == "q3")
info = meta["tensors"][test_key]
back = dequant_q3(payload[test_key + ".q3"].numpy(), payload[test_key + ".s"].numpy(), info["shape"], info["n"])
print(f"round-trip {test_key}: shape {tuple(back.shape)} OK")

file_mb = os.path.getsize(HK_PATH) / 1e6
stream_mb = file_mb + 15 + 25 + 20   # packed resident + one-layer working + activations + tiny engine
fullfp16_mb = (sum(v.numel() for k, v in payload.items() if k != "__meta__") * 2) / 1e6 + 60
print(f"file: {file_mb:.1f} MB | streaming runtime est: ~{stream_mb:.0f} MB | full-fp16 est: ~{fullfp16_mb:.0f} MB (budget <220MB)")

# %% 14. Package service bundle + streaming self-test (+ optional HuggingFace upload)
# Needs training/microdiffusion_model.py from this repo (pushed to main first).
import importlib.util
import shutil
import urllib.request

RAW_MODEL_PY = ("https://raw.githubusercontent.com/shakkhorpaul50-ai/S-Image-gen-AI"
                "/main/training/microdiffusion_model.py")
os.makedirs("/kaggle/working/bundle", exist_ok=True)
urllib.request.urlretrieve(RAW_MODEL_PY, "/kaggle/working/bundle/microdiffusion_model.py")
spec = importlib.util.spec_from_file_location("mdm", "/kaggle/working/bundle/microdiffusion_model.py")
mdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdm)

WQ = HK_PATH if HK_PATH.endswith(".safetensors") else "/kaggle/working/microdiffusion150m.safetensors"
TQ = "/kaggle/working/tokenizer.json"
assert os.path.exists(WQ) and os.path.exists(TQ), "run Cell 12 first (export)"
bundle = mdm.load_service_bundle(WQ, TQ, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"bundle OK: {bundle['unique_m']:.1f}M unique params | scale={bundle['scale']:.4f} | vocab={len(bundle['vocab'])}")

# streaming self-test: attach Q3Streamer to a fresh fp32 copy and compare one DiT forward
dev = "cuda" if torch.cuda.is_available() else "cpu"
dit2, te2, vae2 = mdm.build_models(bundle["cfg"], len(bundle["vocab"]), device="cpu")
dit2.load_state_dict(bundle["dit"].state_dict())
te2.load_state_dict(bundle["te"].state_dict())
packed, rest, qmeta = mdm.pack_state_dict({f"dit.{k}": v for k, v in bundle["dit"].state_dict().items()})
mdm.apply_rest_state({"dit": dit2}, {f"dit.{k}": v for k, v in rest.items()})
stream = mdm.Q3Streamer(packed, qmeta, device="cpu").attach(dit2)
with torch.no_grad():
    z = torch.randn(1, 4, 32, 32)
    t = torch.full((1,), 0.5)
    ids = torch.tensor([[1] + [5] * 20 + [2] + [0] * 55])
    txt, tm = bundle["te"](ids)
    a = bundle["dit"](z, t, txt, tm)
    b = dit2(z, t, txt, tm)
print(f"streaming vs full fp32 max-abs-diff: {(a.float() - b.float()).abs().max().item():.5f} (expect small, ~= quant noise)")
stream.remove()

imgs = mdm.sample_txt(bundle["dit"], bundle["te"], bundle["vae"], bundle["stoi"],
                      ["a cat, ukiyo-e woodblock print"], bundle["scale"],
                      steps=4, cfg=5.0, seed=7, device=dev)
imgs[0].save("/kaggle/working/bundle/smoke.png")
print("smoke.png saved")

shutil.copy(WQ, "/kaggle/working/bundle/microdiffusion150m_q3.safetensors")
shutil.copy(TQ, "/kaggle/working/bundle/tokenizer.json")
files = sorted(os.listdir("/kaggle/working/bundle"))
total = sum(os.path.getsize(os.path.join("/kaggle/working/bundle", f)) for f in files)
print(f"bundle dir: {total / 1e6:.1f} MB ->", files)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "")  # e.g. "shakkhorpaul50-ai/microdiffusion-015b-q3"
if HF_TOKEN and HF_REPO:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO, exist_ok=True)
    api.upload_folder(folder_path="/kaggle/working/bundle", repo_id=HF_REPO)
    print(f"uploaded -> https://huggingface.co/{HF_REPO}")
    print("Space env URLs:")
    print(f"  WEIGHTS_URL=https://huggingface.co/{HF_REPO}/resolve/main/microdiffusion150m_q3.safetensors")
    print(f"  TOKENIZER_URL=https://huggingface.co/{HF_REPO}/resolve/main/tokenizer.json")
else:
    print("to upload: set HF_TOKEN + HF_REPO env vars and re-run this cell,")
    print("or download /kaggle/working/bundle and upload manually to a Hub model repo.")
