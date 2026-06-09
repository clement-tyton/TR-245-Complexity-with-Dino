"""Compare la degradation des features DINOv3 selon le decoupage d'une zone 1024x1024.

Zone test = coin superieur gauche 1024x1024 du blob <uuid> (defaut d58d544c).
On extrait le meme champ de patch-tokens (64x64, ViT-L/16) de 3 facons :
  - 16 tuiles 256x256  (peu de contexte par forward)
  - 4  tuiles 512x512
  - 1  image 1024x1024 (contexte global = REFERENCE)
1024/16=64, 512/16=32, 256/16=16 -> les 3 grilles se recollent en 64x64 exact.

Visualisation : PCA fittee UNE fois sur la reference 1024, appliquee aux 3 champs
(meme base -> couleurs comparables, les coutures de tuilage ressortent).
Metrique : cosinus moyen de chaque token vs la reference 1024.

Lancer a la main :
    .venv/bin/python dino_tiling_compare.py            # d58d544c, coin (0,0)
    .venv/bin/python dino_tiling_compare.py <uuid> <x0> <y0>
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np
import torch
from dinov3.models.vision_transformer import DinoVisionTransformer

SITE = "/home/clement/local_copy_train_data/29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg"
WEIGHTS = "/home/clement/Desktop/projets/1_Core_tyton_AI/tytonai-python-activities/dinov3_embedding/test_data/dinov3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
OUT = "/home/clement/Desktop/projets/2_Actual_jira_tickets/complexity_with_dino"
MEAN = (0.430, 0.411, 0.296)   # stats satellite (comme exploration.py)
STD = (0.213, 0.156, 0.143)
P = 16          # patch size ViT-L/16
EMBED = 1024
CROP = 1024
pat = re.compile(r"image_(.+)_(\d+)_(\d+)\.npz")


# ---------- reconstruction blob ----------
def load_rgb(path):
    d = np.load(path, allow_pickle=True)
    return np.stack([d["RED"], d["GREEN"], d["BLUE"]], -1).astype(np.float32)  # HWC 0-255


def reconstitute(uid):
    cells = {}
    for f in glob.glob(os.path.join(SITE, "train", f"image_{uid}*")) + \
             glob.glob(os.path.join(SITE, "val", f"image_{uid}*")):
        m = pat.search(os.path.basename(f))
        cells[(int(m.group(2)), int(m.group(3)))] = f  # (X, Y)
    imgs = {k: load_rgb(p) for k, p in cells.items()}
    W = max(x + imgs[(x, y)].shape[1] for (x, y) in cells)
    H = max(y + imgs[(x, y)].shape[0] for (x, y) in cells)
    canvas = np.zeros((H, W, 3), np.float32)
    for (x, y), im in imgs.items():
        h, w = im.shape[:2]
        canvas[y:y + h, x:x + w] = im
    return canvas


# ---------- modele ----------
def build_model(device):
    model = DinoVisionTransformer(
        img_size=224, patch_size=P, in_chans=3,
        pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
        embed_dim=EMBED, depth=24, num_heads=16, ffn_ratio=4, qkv_bias=True,
        drop_path_rate=0.0, layerscale_init=1.0e-05, norm_layer="layernormbf16",
        ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4,
        mask_k_bias=True, untie_global_and_local_cls_norm=True, device=device)
    sd = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def normalize_batch(crop_rgb_uint, device):
    """list/array HWC 0-255 -> normalized CHW tensor batch."""
    t = torch.from_numpy(np.stack(crop_rgb_uint)).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return ((t - mean) / std).to(device)


@torch.inference_mode()
def patch_field(model, tiles_hwc, device):
    """tiles_hwc: list of square crops same size -> [N, gh, gw, EMBED] patch grids."""
    batch = normalize_batch(tiles_hwc, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        feats = model.get_intermediate_layers(batch, n=1, reshape=True, norm=True)[0]
    return feats.float().permute(0, 2, 3, 1).cpu().numpy()  # [N, gh, gw, D]


def tile_grid(area, tile):
    """decoupe area (CROPxCROPx3) en tuiles tile x tile, ordre row-major, + positions."""
    n = CROP // tile
    out, pos = [], []
    for r in range(n):
        for c in range(n):
            out.append(area[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile])
            pos.append((r, c))
    return out, pos, n


def assemble(fields, pos, n, gtile):
    """recolle les grilles de patchs des tuiles en un champ 64x64xD."""
    G = n * gtile
    full = np.zeros((G, G, EMBED), np.float32)
    for f, (r, c) in zip(fields, pos):
        full[r * gtile:(r + 1) * gtile, c * gtile:(c + 1) * gtile] = f
    return full


def pca_rgb(field2d, basis=None):
    """field2d [G,G,D] -> (rgb [G,G,3], basis). basis=(mean, components) fit sur reference."""
    G = field2d.shape[0]
    X = field2d.reshape(-1, EMBED)
    if basis is None:
        mean = X.mean(0)
        Xc = X - mean
        # 3 premiers axes via SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comp = Vt[:3]
        basis = (mean, comp)
    mean, comp = basis
    proj = (X - mean) @ comp.T            # [N,3]
    # normalisation robuste (2-98 percentile) pour le rendu
    lo = np.percentile(proj, 2, axis=0); hi = np.percentile(proj, 98, axis=0)
    rgb = np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb.reshape(G, G, 3), basis


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else "d58d544c"
    x0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    y0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    device = torch.device("cuda")

    blob = reconstitute(uid)
    H, W = blob.shape[:2]
    assert x0 + CROP <= W and y0 + CROP <= H, \
        f"blob {uid} fait {W}x{H}, crop 1024 a ({x0},{y0}) hors limites"
    area = blob[y0:y0 + CROP, x0:x0 + CROP]
    print(f"blob {uid} {W}x{H} -> crop 1024 a ({x0},{y0})")

    model = build_model(device)

    configs = {}  # name -> field 64x64xD
    for tile in (256, 512, 1024):
        tiles, pos, n = tile_grid(area, tile)
        gtile = tile // P
        fields = patch_field(model, tiles, device)      # [n*n, gtile, gtile, D]
        configs[tile] = assemble(fields, pos, n, gtile)
        print(f"  {n*n:2d} tuiles {tile}x{tile} -> patch grid {gtile}x{gtile} -> champ {n*gtile}x{n*gtile}")

    # PCA fittee sur la reference 1024, appliquee a tous
    ref = configs[1024]
    _, basis = pca_rgb(ref)
    rgbs = {t: pca_rgb(configs[t], basis)[0] for t in (256, 512, 1024)}

    # degradation : cosinus moyen vs reference 1024 (par token)
    def cos_vs_ref(t):
        a = configs[t].reshape(-1, EMBED); b = ref.reshape(-1, EMBED)
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
        return float((num / den).mean())
    cos = {t: cos_vs_ref(t) for t in (256, 512, 1024)}

    # ---------- figure ----------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(area.astype(np.uint8)); axes[0].set_title(f"crop 1024x1024\nblob {uid} @({x0},{y0})")
    titles = {256: "16 tuiles 256x256", 512: "4 tuiles 512x512", 1024: "1 image 1024 (REF)"}
    for ax, t in zip(axes[1:], (256, 512, 1024)):
        ax.imshow(rgbs[t])
        ax.set_title(f"{titles[t]}\nPCA RGB  | cos vs REF = {cos[t]:.3f}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        "DINOv3 ViT-L/16 (sat493m) — degradation des features selon le decoupage "
        "(PCA commune fittee sur la REF 1024)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, f"dino_tiling_compare_{uid}.png")
    fig.savefig(p, dpi=110)
    print("saved", p)
    print("cosinus moyen vs REF 1024 :", {t: round(c, 4) for t, c in cos.items()})


if __name__ == "__main__":
    main()
