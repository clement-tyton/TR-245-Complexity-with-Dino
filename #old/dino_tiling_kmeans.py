"""Segmentation kmeans des patch-tokens DINOv3 selon le decoupage 256/512/1024.

Meme zone 1024x1024 qu'avec dino_tiling_compare. On extrait le champ 64x64 de
patch-tokens de 3 facons (16x256, 4x512, 1x1024). On fait UN kmeans (spherique =
par direction, coherent avec la cosine similarity) sur la REF 1024, puis on assigne
les 3 configs aux MEMES centres -> meme palette -> la degradation se voit comme une
segmentation qui se fragmente / change de classe quand on decoupe plus fin.

    .venv/bin/python dino_tiling_kmeans.py f7400fc2 0 0 [K]
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dino_tiling_compare import (
    reconstitute, build_model, patch_field, tile_grid, assemble, CROP, EMBED, P, OUT)
import torch


def spherical_kmeans(X, K, iters=50, seed=0):
    """kmeans sur vecteurs normalises (distance = 1 - cosinus). Retourne centres."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    rng = np.random.RandomState(seed)
    # init kmeans++ leger
    c = [Xn[rng.randint(len(Xn))]]
    for _ in range(K - 1):
        d = 1 - (Xn @ np.array(c).T).max(1)
        d = np.clip(d, 0, None); p = d / d.sum()
        c.append(Xn[rng.choice(len(Xn), p=p)])
    C = np.array(c)
    for _ in range(iters):
        lab = (Xn @ C.T).argmax(1)
        newC = np.zeros_like(C)
        for k in range(K):
            m = lab == k
            if m.any():
                v = Xn[m].mean(0); newC[k] = v / (np.linalg.norm(v) + 1e-8)
            else:
                newC[k] = Xn[rng.randint(len(Xn))]
        if np.allclose(newC, C):
            C = newC; break
        C = newC
    return C


def assign(field, C):
    X = field.reshape(-1, EMBED)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    G = field.shape[0]
    return (Xn @ C.T).argmax(1).reshape(G, G)


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else "f7400fc2"
    x0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    y0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    device = torch.device("cuda")

    blob = reconstitute(uid)
    area = blob[y0:y0 + CROP, x0:x0 + CROP]
    model = build_model(device)

    configs = {}
    for tile in (256, 512, 1024):
        tiles, pos, n = tile_grid(area, tile)
        fields = patch_field(model, tiles, device)
        configs[tile] = assemble(fields, pos, n, tile // P)

    # kmeans sur la REF 1024, memes centres pour tous
    C = spherical_kmeans(configs[1024].reshape(-1, EMBED), K)
    labels = {t: assign(configs[t], C) for t in (256, 512, 1024)}

    # % de patchs dont la classe DIFFERE de la ref (mesure de degradation)
    ref = labels[1024]
    diff = {t: float((labels[t] != ref).mean()) for t in (256, 512)}

    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, 10))[:K])
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(area.astype(np.uint8)); axes[0].set_title(f"crop 1024\nblob {uid} @({x0},{y0})")
    titles = {256: "16x256", 512: "4x512", 1024: "1x1024 (REF)"}
    for ax, t in zip(axes[1:], (256, 512, 1024)):
        ax.imshow(labels[t], cmap=cmap, vmin=0, vmax=K - 1, interpolation="nearest")
        extra = "" if t == 1024 else f"\n{diff[t]*100:.0f}% patchs mal classes vs REF"
        ax.set_title(f"kmeans K={K} — {titles[t]}{extra}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Segmentation kmeans (spherique) des features DINOv3 — memes centres, "
                 f"vue de la degradation par decoupage", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, f"dino_tiling_kmeans_{uid}.png")
    fig.savefig(p, dpi=110)
    print("saved", p)
    print("patchs mal classes vs REF 1024 :", {t: f"{d*100:.1f}%" for t, d in diff.items()})


if __name__ == "__main__":
    main()
