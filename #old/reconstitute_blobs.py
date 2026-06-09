"""Reconstitue les blobs (regroupements de tiles 384x384) d'un site et visualise.

Logique:
- Chaque UUID dans le nom de fichier = une image source = un "blob".
- Le nom image_<uuid>_<X>_<Y>.npz : X (champ1) = colonne (horizontal), Y (champ2) = ligne (vertical).
  Verifie via les tuiles de bord tronquees (W tronquee au max X, H tronquee au max Y).
- GEO_TRANSFORM partage le meme origin pour tous les UUIDs -> inutilisable pour
  positionner les blobs entre eux. On se base donc uniquement sur uuid + offsets.
- On reconstitue chaque blob, on overlay la grille 384 et la partition 768 (2x2).
"""
import glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SITE = "/home/clement/local_copy_train_data/29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg"
S = 384
OUT = "/home/clement/Desktop/projets/2_Actual_jira_tickets/complexity_with_dino"
pat = re.compile(r"image_(.+)_(\d+)_(\d+)\.npz")


def collect():
    tiles = defaultdict(dict)  # uuid -> {(x,y): path}
    for split in ("train", "val"):
        for f in glob.glob(os.path.join(SITE, split, "*.npz")):
            m = pat.search(os.path.basename(f))
            tiles[m.group(1)][(int(m.group(2)), int(m.group(3)))] = f  # (X, Y)
    return tiles


def load_rgb(path):
    d = np.load(path, allow_pickle=True)
    rgb = np.stack([d["RED"], d["GREEN"], d["BLUE"]], axis=-1) / 255.0
    return np.clip(rgb, 0, 1)


def reconstitute(cells):
    # cells indexe par (X=colonne, Y=ligne)
    xs = sorted({c[0] for c in cells}); ys = sorted({c[1] for c in cells})
    rgbs = {k: load_rgb(p) for k, p in cells.items()}
    W = max(x + rgbs[(x, y)].shape[1] for (x, y) in cells)
    H = max(y + rgbs[(x, y)].shape[0] for (x, y) in cells)
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for (x, y), img in rgbs.items():
        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img
    nr = len(ys); nc = len(xs)  # lignes, colonnes
    return canvas, nr, nc


def main():
    tiles = collect()
    # trie par taille decroissante pour un layout agreable
    order = sorted(tiles.items(), key=lambda kv: -len(kv[1]))
    n = len(order)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    n768 = 0
    for ax, (uid, cells) in zip(axes, order):
        img, nr, nc = reconstitute(cells)
        ax.imshow(img)
        # grille 384 (fin, gris)
        for i in range(1, nc):
            ax.axvline(i * S, color="white", lw=0.4, alpha=0.5)
        for i in range(1, nr):
            ax.axhline(i * S, color="white", lw=0.4, alpha=0.5)
        # partition 768 (2x2) -> rectangles verts pour les blocs propres
        nb = 0
        for by in range(0, nr - nr % 2, 2):
            for bx in range(0, nc - nc % 2, 2):
                ax.add_patch(Rectangle((bx * S, by * S), 2 * S, 2 * S,
                             fill=False, edgecolor="lime", lw=1.6))
                nb += 1
        n768 += nb
        leftover = (nr % 2) or (nc % 2)
        ax.set_title(f"{uid[:8]}  {nr}x{nc}={nr*nc}t\n{nb}x 768"
                     + ("  +reste" if leftover else ""), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Site 29M_2451_GG_manned — {n} blobs reconstitues (tiles 384px)\n"
        f"vert = bloc 768x768 propre (2x2) — total {n768} super-tiles 768",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "blobs_reconstitution.png")
    fig.savefig(p, dpi=110)
    print("saved", p)
    print(f"{n} blobs, {sum(len(c) for _,c in order)} tiles 384, {n768} blocs 768 propres")


if __name__ == "__main__":
    main()
