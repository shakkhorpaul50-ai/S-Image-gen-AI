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

# %% 15. Export ONNX quartet (fp32) + static INT8 (~90MB) + HF upload
# Proven recipe (validated locally): legacy exporter, dynamo=False, opset 17.
# Exports from fp32 dit_te.pt (NOT the Q3 file) for best numerics.
# Self-contained: re-downloads microdiffusion_model.py, rebuilds from files.
import subprocess as _sp2
_sp2.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=False)
import importlib.util as _ilu
import shutil as _sh
import urllib.request as _url
import numpy as _np
import pandas as _pd
import onnxruntime as _ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

_mraw = "https://raw.githubusercontent.com/shakkhorpaul50-ai/S-Image-gen-AI/main/training/microdiffusion_model.py"
os.makedirs("/kaggle/working/onnx", exist_ok=True)
_url.urlretrieve(_mraw, "/kaggle/working/onnx/_mdm.py")
_sp = importlib.util.spec_from_file_location("_mdm", "/kaggle/working/onnx/_mdm.py")
_mdm = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_mdm)

ONNX_DIR = "/kaggle/working/onnx"
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_calib_prov = (["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"])

_tj = json.load(open("/kaggle/working/tokenizer.json"))
_cfg = _tj["cfg"]
_dit, _te, _vae = _mdm.build_models(_cfg, len(_tj["vocab"]), device="cpu")
_ckpt = torch.load("/kaggle/working/dit_te.pt", map_location="cpu", weights_only=False)
_dit.load_state_dict(_ckpt["dit"]); _te.load_state_dict(_ckpt["te"])
_vae.load_state_dict(torch.load("/kaggle/working/vae.pt", map_location="cpu", weights_only=False))
_dit.eval(); _te.eval(); _vae.eval()

class _DecOnly(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.dec_in = vae.dec_in
        self.dec = vae.dec
    def forward(self, z):
        return self.dec(self.dec_in(z))

class _EncOnly(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.enc = vae.enc
        self.mu = vae.mu
    def forward(self, x):
        return self.mu(self.enc(x))

def _export(mod, args, names_in, names_out, path):
    with torch.no_grad():
        ref = mod(*args)
    torch.onnx.export(mod, args, path, input_names=names_in, output_names=names_out,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    sess = _ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {k: (v.numpy() if torch.is_tensor(v) else v)
                          for k, v in zip(names_in, args)})[0]
    r = ref.numpy() if torch.is_tensor(ref) else ref
    cos = float((r.flatten() * o.flatten()).sum() / (_np.linalg.norm(r.flatten()) * _np.linalg.norm(o.flatten()) + 1e-12)) if (o := out) is not None else 0.0
    print(f"{os.path.basename(path)}: cosine vs torch = {cos:.6f} (want >0.999)")
    return path

_ids = torch.tensor([encode_text("a cat, ukiyo-e woodblock print")])
_zt = torch.randn(1, 4, 32, 32)
_xt = torch.randn(1, 3, 256, 256)
_export(_te, (_ids,), ["ids"], ["txt"], f"{ONNX_DIR}/te150m_fp32.onnx")
_export(_DecOnly(_vae), (_zt,), ["z"], ["img"], f"{ONNX_DIR}/vae150m_dec_fp32.onnx")
_export(_EncOnly(_vae), (_xt,), ["x"], ["mu"], f"{ONNX_DIR}/vae150m_enc_fp32.onnx")
with torch.no_grad():
    _txt, _tm = _te(_ids)
_export(_dit, (_zt, torch.tensor([0.5]), _txt, (_ids == 0)),
        ["x", "t", "txt", "tmask"], ["v"], f"{ONNX_DIR}/dit150m_fp32.onnx")

# calibration data: REAL cached latents + captions (reload session-safe)
_lat = torch.load("/kaggle/working/latents.pt", weights_only=False)["lat"].float()
_capcsvs = glob.glob("/kaggle/input/**/captions.txt", recursive=True)
_caps = _pd.read_csv(_capcsvs[0])["caption"].astype(str).tolist() if _capcsvs else ["a cat"]
_rng = random.Random(11)

class _DiTCal(CalibrationDataReader):
    def __init__(self, n=48):
        self.n, self.i = n, 0
    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        r = _rng.randrange(len(_caps))
        ids = torch.tensor([encode_text(_caps[r])])
        with torch.no_grad():
            txt, tm = _te(ids)
        return {"x": _lat[_rng.randrange(len(_lat))].unsqueeze(0).numpy(),
                "t": _np.array([random.random()], _np.float32),
                "txt": txt.numpy(), "tmask": tm.numpy()}
    def rewind(self):
        self.i = 0

class _TECal(CalibrationDataReader):
    def __init__(self, n=24):
        self.n, self.i = n, 0
    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return {"ids": _np.array([encode_text(_caps[_rng.randrange(len(_caps))])], _np.int64)}
    def rewind(self):
        self.i = 0

class _ZCal(CalibrationDataReader):
    def __init__(self, n=24):
        self.n, self.i = n, 0
    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return {"z": _lat[_rng.randrange(len(_lat))].unsqueeze(0).numpy()}
    def rewind(self):
        self.i = 0

class _XCal(CalibrationDataReader):
    def __init__(self, n=16):
        self.paths = None
        self.n, self.i = n, 0
    def get_next(self):
        if self.paths is None:
            cand = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                cand += glob.glob(os.path.join("/kaggle/input", "**", ext), recursive=True)
            self.paths = cand
        if self.i >= self.n or not self.paths:
            return None
        self.i += 1
        from PIL import Image as _PIL
        im = _PIL.open(self.paths[_rng.randrange(len(self.paths))]).convert("RGB")
        return {"x": tfm(im).unsqueeze(0).numpy()}
    def rewind(self):
        self.i = 0

for _src, _cal in [("dit150m_fp32.onnx", _DiTCal()), ("te150m_fp32.onnx", _TECal()),
                   ("vae150m_dec_fp32.onnx", _ZCal()), ("vae150m_enc_fp32.onnx", _XCal())]:
    _dst = _src.replace("_fp32", "_s8")
    quantize_static(f"{ONNX_DIR}/{_src}", f"{ONNX_DIR}/{_dst}", _cal,
                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                    per_channel=True, calibration_providers=_calib_prov)
    _sess = _ort.InferenceSession(f"{ONNX_DIR}/{_dst}", providers=["CPUExecutionProvider"])
    print(f"{_dst}: {os.path.getsize(f'{ONNX_DIR}/{_dst}') / 1e6:.1f} MB, "
          f"inputs={[i.name for i in _sess.get_inputs()]}")

_tot = sum(os.path.getsize(os.path.join(ONNX_DIR, f)) for f in os.listdir(ONNX_DIR) if f.endswith("_s8.onnx"))
print(f"INT8 quartet total: {_tot / 1e6:.1f} MB (budget: <150MB file, ~350-400MB runtime on Render free)")
_sh.copy("/kaggle/working/tokenizer.json", f"{ONNX_DIR}/tokenizer.json")

_hf_token = os.environ.get("HF_TOKEN", "")
_hf_repo = os.environ.get("HF_REPO", "")
if _hf_token and _hf_repo:
    _sp2.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
    from huggingface_hub import HfApi
    _api = HfApi(token=_hf_token)
    _api.create_repo(_hf_repo, exist_ok=True)
    _api.upload_folder(folder_path=ONNX_DIR, repo_id=_hf_repo)
    print(f"onnx uploaded -> https://huggingface.co/{_hf_repo}")
    for _f in ["dit150m_s8.onnx", "te150m_s8.onnx", "vae150m_dec_s8.onnx", "vae150m_enc_s8.onnx", "tokenizer.json"]:
        print(f"  https://huggingface.co/{_hf_repo}/resolve/main/{_f}")
    print(f"Render/Docker env: ONNX_BASE_URL=https://huggingface.co/{_hf_repo}/resolve/main")
else:
    print("set HF_TOKEN + HF_REPO env vars and re-run for upload, or download /kaggle/working/onnx manually.")
