"""MicroDiffusion 0.15B — canonical model module (torch-only, no training code).

Single source of truth for the 150M-nominal / ~90M-stored retrain, shared by:
  - the Kaggle training notebook (same arch; packaging cell cross-checks this
    file against a trained checkpoint), and
  - inference hosts (import, load the Q3 bundle, sample).

Architecture:
  VAE:  ch (64,128,256), 3 downs (256->32), 4 latent channels, ~8M
  Text: word-level vocab, d=448, L=6, h=8, ff=1792, max_len=77, ~23M
  DiT:  d=576, h=12, mlp=2304, patch=2, grid=16 (32x32 latent),
        8 UNIQUE blocks x 2 passes (tied weights), cross-attn text (td=448),
        adaLN-Zero, ~59M stored / ~117M nominal
  Total: ~90M stored unique / ~148M nominal (~0.15B).
Quant:  Q3_K_S-style symmetric grouped 3-bit, group=32 (~3.5 bpw).
        NOTE: exact GGUF Q3_K_M is llama.cpp-only and cannot run this arch;
        this packing is the same size class and what the .safetensors holds.
Runtime strategy (the <220MB guarantee):
  full fp16 dequant ~= 180MB weights + activations -> borderline;
  STREAMING (below) keeps packed Q3 (~45MB) resident and materialises one
  fp32 layer at a time -> ~100-130MB total. Use Q3Streamer on tiny hosts.
Sampling: rectified flow, Euler t:1->0, classifier-free guidance.
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------- tokenizer

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def build_vocab(captions, cap=30000, min_freq=2):
    from collections import Counter
    cnt = Counter()
    for c in captions:
        cnt.update(str(c).lower().split())
    vocab = [PAD, BOS, EOS, UNK] + [w for w, c in cnt.most_common(cap - 4) if c >= min_freq]
    return vocab


def encode_text(cap, stoi, max_len=77):
    ids = [stoi[BOS]] + [stoi.get(w, stoi[UNK]) for w in str(cap).lower().split()[:max_len - 2]] + [stoi[EOS]]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------- VAE (~8M)


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


# ---------------------------------------------------------------- text encoder (~23M)


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
        self.ml = ml
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


# ---------------------------------------------------------------- tied DiT (~59M stored)


def _mod(x, s):
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
        h = _mod(self.n1(x), s1)
        x = x + g1 * self.sa(h, h, h, need_weights=False)[0]
        h = _mod(self.n2(x), s2)
        x = x + g2 * self.xa(h, txt, txt, key_padding_mask=tmask, need_weights=False)[0]
        h = _mod(self.n3(x), s3)
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
        nn.init.zeros_(self.out_a.weight)
        nn.init.zeros_(self.out_a.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

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
        h = self.out(_mod(self.out_n(h), sc) + sh)
        B, T, _ = h.shape
        h = h.view(B, self.grid, self.grid, self.p, self.p, self.z)
        return h.permute(0, 5, 1, 3, 2, 4).reshape(B, self.z, self.grid * self.p, self.grid * self.p)


def build_models(cfg, vocab_size, device="cpu"):
    """cfg: dict with keys dit_dim, dit_heads, dit_mlp, te_dim, dit_unique,
    dit_passes, dit_patch, dit_grid, te_layers, te_heads, te_ff, max_len,
    vae_ch (list), latent_ch."""
    dit = DiT(cfg["latent_ch"], cfg["dit_dim"], cfg["dit_heads"], cfg["dit_mlp"], cfg["te_dim"],
              cfg["dit_unique"], cfg["dit_passes"], cfg["dit_patch"], cfg["dit_grid"])
    te = TextEnc(vocab_size, cfg["te_dim"], cfg["te_layers"], cfg["te_heads"], cfg["te_ff"], cfg["max_len"])
    vae = VAE(tuple(cfg["vae_ch"]), cfg["latent_ch"])
    return dit.to(device), te.to(device), vae.to(device)


def count_params(*models):
    return sum(p.numel() for m in models for p in m.parameters())


# ---------------------------------------------------------------- Q3 quant (device-safe)


def quant_q3(w: torch.Tensor, group=32):
    """Symmetric grouped 3-bit quant. Returns (packed uint8 np, fp16 scales np, shape, n)."""
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
    """Inverse of quant_q3. b/s may be numpy arrays or CPU tensors. Returns fp32 CPU tensor."""
    if torch.is_tensor(b):
        b = b.cpu().numpy()
    if torch.is_tensor(s):
        s = s.cpu().numpy()
    b = np.asarray(b, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
    pack = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
    q = np.stack([(pack >> (3 * i)) & 7 for i in range(8)], axis=1).reshape(-1).astype(np.float32) - 3.0
    s = np.asarray(s, dtype=np.float32).repeat(group)[:len(q)]
    return torch.from_numpy((q * s)[:n].reshape(shape))


# ---------------------------------------------------------------- bundle IO (Q3 .safetensors + tokenizer.json)


def _read_bundle(weights_path, tokenizer_path):
    from safetensors.torch import load_file
    tj = json.load(open(tokenizer_path))
    cfg, scale, vocab = tj["cfg"], tj["scale"], tj["vocab"]
    raw = load_file(weights_path, device="cpu")
    meta = json.loads(bytes(raw.pop("__meta__").numpy()).decode())
    return raw, meta, vocab, scale, cfg


def load_service_bundle(weights_path, tokenizer_path, device="cpu"):
    """Full fp32/fp16 materialisation. Use on hosts with RAM to spare."""
    raw, meta, vocab, scale, cfg = _read_bundle(weights_path, tokenizer_path)
    stoi = {w: i for i, w in enumerate(vocab)}
    dit, te, vae = build_models(cfg, len(vocab), device="cpu")
    buckets = {"dit": {}, "te": {}, "vae": {}}
    for key, info in meta["tensors"].items():
        pre, name = key.split(".", 1)
        if info["q"] == "q3":
            t = dequant_q3(raw[key + ".q3"], raw[key + ".s"], info["shape"], info["n"])
        else:
            t = raw[key].float()
        buckets[pre][name] = t
    dit.load_state_dict(buckets["dit"])
    te.load_state_dict(buckets["te"])
    vae.load_state_dict(buckets["vae"])
    dit.to(device).eval()
    te.to(device).eval()
    vae.to(device).eval()
    return {"dit": dit, "te": te, "vae": vae, "scale": scale,
            "vocab": vocab, "stoi": stoi, "cfg": cfg,
            "unique_m": count_params(dit, te, vae) / 1e6}


def pack_state_dict(state):
    """Split a fp32 state dict into packed-Q3 tensors + small fp16 rest.
    Returns (packed, rest, meta) with the same key scheme as the Cell-12 export
    (payload keys '<name>.q3' / '<name>.s', meta['tensors'][name] descriptors)."""
    packed, rest, meta = {}, {}, {"tensors": {}}
    for key, p in state.items():
        t = p.detach().cpu()
        if t.dim() >= 2 and t.numel() >= 4096:
            b, s, shape, n = quant_q3(t)
            packed[key + ".q3"] = torch.from_numpy(b)
            packed[key + ".s"] = torch.from_numpy(s)
            meta["tensors"][key] = {"q": "q3", "shape": list(shape), "n": n}
        else:
            rest[key] = t.half()
            meta["tensors"][key] = {"q": "f16", "shape": list(t.shape)}
    return packed, rest, meta


class Q3Streamer:
    """Streaming Q3 execution for tiny hosts (<220MB runtime weights).

    Keeps packed Q3 resident (~45MB for the 150M model) and materialises one
    fp32 layer at a time via forward hooks, releasing it right after.
    Inference only (hooks break gradient flow by design).

    Usage:
        streamer = Q3Streamer(packed, meta, device="cpu").attach(dit)
        ... run model ...
        streamer.remove()
    """

    def __init__(self, packed, meta, device="cpu"):
        self.packed = packed
        self.meta = meta
        self.device = device
        self.handles = []

    def attach(self, model):
        for key, info in self.meta["tensors"].items():
            if info["q"] != "q3":
                continue
            # meta keys carry the model prefix ("dit."/"te."/"vae.") matching the
            # Cell-12 export format; strip it to get the in-module path.
            p = key
            for pre in ("dit.", "te.", "vae."):
                if p.startswith(pre):
                    p = p[len(pre):]
                    break
            path, _, attr = p.rpartition(".")
            mod = model.get_submodule(path)  # attr is usually "weight",
            b, s = self.packed[key + ".q3"], self.packed[key + ".s"]  # sometimes "in_proj_weight" (MHA)
            shape, n = tuple(info["shape"]), info["n"]
            dev = self.device

            def pre_hook(m, inp, _b=b, _s=s, _sh=shape, _n=n, _a=attr):
                setattr(m, _a, torch.nn.Parameter(dequant_q3(_b, _s, _sh, _n).to(dev)))

            def fwd_hook(m, inp, out, _a=attr):
                delattr(m, _a)  # release fp32 copy; rebuilt on next forward

            self.handles.append(mod.register_forward_pre_hook(pre_hook))
            self.handles.append(mod.register_forward_hook(fwd_hook))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def load_packed_bundle(weights_path, tokenizer_path):
    """Load without materialising: returns packed Q3 + small fp32 rest + meta.
    Pair with Q3Streamer per model for <220MB inference (see module docstring)."""
    raw, meta, vocab, scale, cfg = _read_bundle(weights_path, tokenizer_path)
    packed, rest = {}, {}
    for key, info in meta["tensors"].items():
        if info["q"] == "q3":
            packed[key + ".q3"] = raw[key + ".q3"]
            packed[key + ".s"] = raw[key + ".s"]
        else:
            rest[key] = raw[key].float()
    stoi = {w: i for i, w in enumerate(vocab)}
    return {"packed": packed, "rest": rest, "meta": meta, "vocab": vocab,
            "stoi": stoi, "scale": scale, "cfg": cfg}


def apply_rest_state(models, rest):
    """Load the small fp32 'rest' tensors (biases, norms, embeddings-small...)
    into already-built models. Quantized weights are handled by Q3Streamer."""
    for mname, model in models.items():
        sd = {k.split(".", 1)[1]: v for k, v in rest.items() if k.startswith(mname + ".")}
        model.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------- sampling (rectified flow, Euler, CFG)


@torch.no_grad()
def sample_txt(dit, te, vae, stoi, prompts, scale, steps=24, cfg=5.0, seed=0,
               device="cpu", max_len=77, latent_hw=32, latent_ch=4):
    dit.eval()
    te.eval()
    vae.eval()
    B = len(prompts)
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([encode_text(p, stoi, max_len) for p in prompts], device=device)
    txt, tm = te(ids)
    null = torch.zeros_like(txt)                       # uncond: zeros, NOT masked
    nullm = torch.zeros_like(tm)
    x = torch.randn(B, latent_ch, latent_hw, latent_hw, generator=g, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), 1 - i * dt, device=device)
        vc = dit(x, t, txt, tm)
        vu = dit(x, t, null, nullm)
        x = x - (vu + cfg * (vc - vu)) * dt            # Euler t: 1 -> 0
    img = vae.decode(x / scale).clamp(-1, 1)
    img = ((img + 1) / 2).clamp(0, 1)
    out = []
    for i in range(B):
        arr = (img[i].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        out.append(Image.fromarray(arr))
    return out


@torch.no_grad()
def sample_img2img(dit, te, vae, stoi, pil_img, prompt, scale, strength=0.6,
                   steps=24, cfg=5.0, seed=0, device="cpu", max_len=77, img_size=256):
    dit.eval()
    te.eval()
    vae.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    x0 = pil_to_tensor(pil_img, img_size).unsqueeze(0).to(device)
    mu, _ = vae.encode(x0)
    z0 = mu * scale
    if strength <= 0.01:
        img = vae.decode(z0 / scale).clamp(-1, 1)
        return [_to_pil(img[0])]
    t0 = float(min(max(strength, 0.02), 1.0))
    eps = torch.randn(z0.shape, generator=g, device=device)
    x = (1 - t0) * z0 + t0 * eps
    ids = torch.tensor([encode_text(prompt, stoi, max_len)], device=device)
    txt, tm = te(ids)
    null = torch.zeros_like(txt)
    nullm = torch.zeros_like(tm)
    n = max(1, int(steps * t0))
    dt = t0 / n
    for i in range(n):
        t = torch.full((1,), t0 - i * dt, device=device)
        vc = dit(x, t, txt, tm)
        vu = dit(x, t, null, nullm)
        x = x - (vu + cfg * (vc - vu)) * dt
    img = vae.decode(x / scale).clamp(-1, 1)
    return [_to_pil(img[0])]


def pil_to_tensor(pil_img, img_size=256):
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                     T.Normalize([0.5] * 3, [0.5] * 3)])
    return tfm(pil_img.convert("RGB"))


def _to_pil(t):
    t = ((t.clamp(-1, 1) + 1) / 2).clamp(0, 1)
    arr = (t.permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)
