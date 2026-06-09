"""Tractable DINOv3 embedding of a whole blob on a 12 GB GPU.

Why this exists: the production activity does ONE monolithic forward per bbox AND upscales
(x4 at 0.15 m native res), so a large area blows up the model input and can OOM. Here we
work at NATIVE resolution (patch 16 -> embed grid = input/16) and:

  - if the (padded) blob fits the GPU in one pass  -> single forward (no seams, best quality)
  - else                                            -> sliding-window tiles (>= min_tile px),
                                                       OVERLAP + center-crop, stitched seamlessly

Measured on this RTX 3060 (12 GB), ViT-L/16, bf16+SDPA: a 2304x2304 input peaks ~2.1 GB and
memory grows near-linearly, so every blob of 29M_2451 (<= 2055x1596) is a SINGLE 1-pass call.
Tiling matters for the 7B model, high_res(x2 upsample), or very large mosaics.

    from embed_blob import embed_blob
    grid = embed_blob("d58d544c")                 # -> [Gh, Gw, 1024] float32, Gh=H//16
    grid = embed_blob("d58d544c", max_input=1536) # force tiling to cap VRAM

CLI:  .venv/bin/python embed_blob.py <uuid> [max_input] [--preview]
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

from dino_tiling_compare import build_model, reconstitute, MEAN, STD, EMBED, P, OUT

# Largest square (multiple of 16) we run in one pass. 2304 -> ~2.1 GB here; 2048 keeps margin.
DEFAULT_MAX_INPUT = 2048


def _pad_to_multiple(area: np.ndarray, mult: int = P) -> np.ndarray:
    """edge-pad HWC so H and W are multiples of `mult` (DINO needs input % patch == 0)."""
    H, W = area.shape[:2]
    ph, pw = (-H) % mult, (-W) % mult
    if ph or pw:
        area = np.pad(area, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return area


@torch.inference_mode()
def _forward(model, area_hwc: np.ndarray, device) -> np.ndarray:
    """area HWC 0-255 (already padded to %16) -> patch grid [gh, gw, EMBED]."""
    t = torch.from_numpy(area_hwc).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    x = ((t - mean) / std).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
    return f.float().squeeze(0).permute(1, 2, 0).cpu().numpy()  # [gh, gw, EMBED]


def embed_blob(uid: str, max_input: int = DEFAULT_MAX_INPUT, min_tile: int = 1024,
               overlap: int = 256, model=None, device=None) -> np.ndarray:
    """Return the native-resolution DINOv3 patch grid for blob `uid` as [Gh, Gw, EMBED].

    Single forward when the padded blob fits `max_input`; otherwise overlap-tiled + stitched.
    Each output cell = one 16x16 native-pixel patch (Gh = ceil(H/16), Gw = ceil(W/16)).
    """
    device = device or torch.device("cuda")
    model = model or build_model(device)

    blob = reconstitute(uid)                       # HWC 0-255, native px
    H0, W0 = blob.shape[:2]
    area = _pad_to_multiple(blob)
    H, W = area.shape[:2]
    Gh0, Gw0 = (H0 + P - 1) // P, (W0 + P - 1) // P  # valid patch grid (trim padding later)

    if max(H, W) <= max_input:
        grid = _forward(model, area, device)
        return grid[:Gh0, :Gw0]

    # ---- sliding-window tiled inference with overlap + center crop ----
    T = max(min_tile, ((max_input) // P) * P)      # tile size (multiple of 16), capped to fit
    T = min(T, (max_input // P) * P)
    ov = (overlap // P) * P                         # overlap (multiple of 16)
    stride = T - ov
    mp = ov // (2 * P)                              # patch margin to drop on interior seams
    Gh, Gw = H // P, W // P
    out = np.zeros((Gh, Gw, EMBED), np.float32)
    filled = np.zeros((Gh, Gw), bool)

    ys = list(range(0, max(1, H - T + 1), stride)) or [0]
    xs = list(range(0, max(1, W - T + 1), stride)) or [0]
    if ys[-1] != H - T:
        ys.append(max(0, H - T))
    if xs[-1] != W - T:
        xs.append(max(0, W - T))

    for y0 in ys:
        for x0 in xs:
            win = area[y0:y0 + T, x0:x0 + T]
            g = _forward(model, win, device)        # [T/P, T/P, EMBED]
            gy, gx = y0 // P, x0 // P
            # keep center: drop `mp` patches on interior edges (not on the global border)
            top = 0 if y0 == 0 else mp
            left = 0 if x0 == 0 else mp
            bot = g.shape[0] if (y0 + T) >= H else g.shape[0] - mp
            right = g.shape[1] if (x0 + T) >= W else g.shape[1] - mp
            for r in range(top, bot):
                for c in range(left, right):
                    if not filled[gy + r, gx + c]:
                        out[gy + r, gx + c] = g[r, c]
                        filled[gy + r, gx + c] = True
    return out[:Gh0, :Gw0]


def _pca_rgb(grid):
    X = grid.reshape(-1, EMBED); Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:3].T
    lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
    return np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1).reshape(grid.shape[0], grid.shape[1], 3)


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else "d58d544c"
    max_input = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else DEFAULT_MAX_INPUT
    preview = "--preview" in sys.argv

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    grid = embed_blob(uid, max_input=max_input, device=dev)
    peak = torch.cuda.max_memory_allocated() / 1e9
    mode = "single-pass" if max(grid.shape[0] * P, grid.shape[1] * P) <= max_input else "tiled"
    print(f"blob {uid}: embed grid {grid.shape}  ({mode}, max_input={max_input})  peak {peak:.1f} GB")

    out_npy = os.path.join(OUT, f"embed_{uid}.npy")
    np.save(out_npy, grid)
    print("saved", out_npy, f"({grid.nbytes/1e6:.0f} MB)")

    if preview:
        import matplotlib.pyplot as plt
        rgb = _pca_rgb(grid)
        fig, ax = plt.subplots(figsize=(grid.shape[1] / 12, grid.shape[0] / 12))
        ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{uid} — PCA RGB of native-res embedding ({grid.shape[1]}x{grid.shape[0]} patches)")
        p = os.path.join(OUT, f"embed_{uid}_pca.png"); fig.tight_layout(); fig.savefig(p, dpi=110)
        print("saved", p)


if __name__ == "__main__":
    main()
