"""Agencement d'un blob avec la BONNE convention d'axes.

Nom: image_<uuid>_<X>_<Y>.npz
  X (champ 1) = offset horizontal (colonne)  -> verifie via tuiles bord droit (W tronquee)
  Y (champ 2) = offset vertical   (ligne)    -> verifie via tuiles bord bas  (H tronquee)
Placement canvas: canvas[Y:Y+h, X:X+w]
"""
import glob, os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SITE = "/home/clement/local_copy_train_data/29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg"
OUT = "/home/clement/Desktop/projets/2_Actual_jira_tickets/complexity_with_dino"
S = 384
pat = re.compile(r"image_(.+)_(\d+)_(\d+)\.npz")


def load_rgb(path):
    d = np.load(path, allow_pickle=True)
    return np.clip(np.stack([d["RED"], d["GREEN"], d["BLUE"]], -1) / 255.0, 0, 1)


def arrange(uid):
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
    return canvas, cells


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else "3a463913"
    canvas, cells = arrange(uid)
    fig, ax = plt.subplots(figsize=(canvas.shape[1] / 120, canvas.shape[0] / 120))
    ax.imshow(canvas)
    for (x, y) in cells:
        ax.add_patch(Rectangle((x, y), S, S, fill=False, edgecolor="yellow", lw=1))
        ax.text(x + 8, y + 30, f"x={x}\ny={y}", color="yellow", fontsize=8,
                va="top", ha="left",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    ax.set_title(f"blob {uid}  —  X=champ1 (colonne) , Y=champ2 (ligne)", fontsize=11)
    ax.set_xlabel("X (px, horizontal)"); ax.set_ylabel("Y (px, vertical)")
    fig.tight_layout()
    p = os.path.join(OUT, f"arrange_{uid}.png")
    fig.savefig(p, dpi=110)
    print("saved", p, "canvas", canvas.shape)


if __name__ == "__main__":
    main()
