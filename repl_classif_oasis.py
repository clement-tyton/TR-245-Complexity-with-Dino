# %% [markdown]
# repl_classif_oasis.py — QUICK & DIRTY classif map (NOT committed, just for fun)
# ==============================================================================
# Train a "poor" logistic regression on a FEW annotated tiles' DINO patch-embeddings, predict the
# whole dry-oasis site, paint a classification map. Relies ONLY on the local parquet embeddings +
# the local annotation tiles. No GPU, no torch — pure numpy/sklearn on the cached features.
#
# Patch<->mask alignment is done in GEO (metres), so the upscale/patch-size ratios (x2->64x64,
# x4->128x128, /16 ...) are IRRELEVANT: each patch's centre easting/northing is derived from its
# cell bbox and GW read from the array, each mask pixel from its GEO_TRANSFORM. Same CRS (EPSG:28350).
# Run the cells top to bottom.

# %% CELL 1 — paths + imports -------------------------------------------------------
import os
import sys
import glob
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, "src")             # run from the repo root
from classes import SEVEN_CLASS_MAPPING, remap_mask, class_name   # TytonAI taxonomy

SITE_ID  = "BHP_Creeks_2022_Manned_Bens_Oasis_Post_Dry_10cm_v2_tytonai_rg"
EMB      = "outputs/embeddings"
CELLS    = f"{EMB}/cells/site_id={SITE_ID}/cells.parquet"
TILES    = "/home/clement/local_copy_train_data/BHP Creeks 2022/Manned Bens Oasis Post Dry/10cm/v2_tytonai_rg"
N_TRAIN_TILES = 80                    # max tiles the greedy balancer may use (hard cap)
MAX_PER_CLASS = 8000                  # cap patches/class for TRAINING (balance + speed); None = no cap
TARGET_PX     = 60_000                # greedy aims for >= this mask-px support per PRESENT class
MIN_SITE_PX   = 2000                  # below this site-wide -> class too rare here, dropped from LABELS
CLASSIFIER    = "logreg"              # "logreg" (linear probe) or "hgb" (HistGradientBoosting)
# 7-class lifeform scheme (raw masks carry many fine classes -> remapped via SEVEN_CLASS_MAPPING)
LABELS   = (2, 3, 4, 5, 6, 40, 301)   # Ground, HumGras, Shrub, Tree, Herb, Sedge, TusGra
OUT      = "outputs/classif_quicktest"
os.makedirs(OUT, exist_ok=True)


# %% CELL 2 — load embedding cells (geometry + patch file), infer grid + patch size ----
man = gpd.read_parquet(CELLS).sort_values("cell_id").reset_index(drop=True)
man["npz"] = [os.path.join(EMB, p) for p in man["patch_npz"]]
GH, GW, C = np.load(man["npz"].iloc[0])["patch_grid"].shape
_b = man.geometry.iloc[0].bounds
PATCH_M = (_b[2] - _b[0]) / GW
print(f"{len(man)} cells | CRS EPSG:{man.crs.to_epsg()} | patch grid {GH}x{GW} | C={C} | {PATCH_M:.2f} m/patch")


def patch_centres(bounds):
    """(xs[GW], ys[GH]) easting/northing of each patch centre in a cell (row 0 = top = max north)."""
    x0, y0, x1, y1 = bounds
    xs = x0 + (np.arange(GW) + 0.5) / GW * (x1 - x0)
    ys = y1 - (np.arange(GH) + 0.5) / GH * (y1 - y0)
    return xs, ys


def load_tile(name):
    """-> (footprint box, (ox,oy,px,py), mask HxW remapped to the 7-class scheme). px>0, py<0."""
    gt = np.asarray(np.load(os.path.join(TILES, "train", name))["GEO_TRANSFORM"], float)
    mask = remap_mask(np.load(os.path.join(TILES, "trainannot", name))["CLASSIFY"], SEVEN_CLASS_MAPPING)
    h, w = mask.shape
    ox, oy, px, py = gt[2], gt[5], gt[0], gt[4]
    return box(ox, oy + h * py, ox + w * px, oy), (ox, oy, px, py), mask


# %% CELL 3 — GREEDY balanced tile selection: enough support for EVERY present class -----
site = man.geometry.union_all()
cands = []                                                  # each: (name, foot, geo, mask, counts)
for name in sorted(os.listdir(os.path.join(TILES, "trainannot"))):
    foot, geo, mask = load_tile(name)
    if not foot.intersects(site):
        continue
    cnt = {c: int((mask == c).sum()) for c in LABELS}
    if sum(cnt.values()) > 500:
        cands.append((name, foot, geo, mask, cnt))
print(f"{len(cands)} annotated tiles overlap the site")

# site-wide ceiling per class (max support if we used ALL tiles) -> drop classes too rare here
ceiling = {c: sum(t[4][c] for t in cands) for c in LABELS}
print("site-wide max mask-px/class:", {class_name(c): ceiling[c] for c in LABELS})
LABELS = tuple(c for c in LABELS if ceiling[c] >= MIN_SITE_PX)
dropped = [class_name(c) for c in ceiling if ceiling[c] < MIN_SITE_PX]
if dropped:
    print(f"  /!\\ too rare at this site (< {MIN_SITE_PX} px) -> dropped: {dropped}")

# greedy: each step add the tile that best fills the current per-class deficits (poorest weighted most)
picked, totals, remaining = [], {c: 0 for c in LABELS}, cands[:]
while remaining and len(picked) < N_TRAIN_TILES:
    deficit = {c: max(0, TARGET_PX - totals[c]) for c in LABELS}
    if not any(deficit.values()):
        break                                              # every present class hit its target
    weight = {c: TARGET_PX / (totals[c] + 1) for c in LABELS}      # poorer class -> bigger weight
    scores = [sum(min(t[4][c], deficit[c]) * weight[c] for c in LABELS) for t in remaining]
    picked.append(remaining.pop(int(np.argmax(scores))))
    for c in LABELS:
        totals[c] += picked[-1][4][c]
pick = picked
met = all(totals[c] >= TARGET_PX for c in LABELS)
print(f"selected {len(pick)} tiles | target {'MET' if met else 'NOT met (site-limited)'} "
      f"@ {TARGET_PX} px")
print("per-class mask-px support:", {class_name(c): totals[c] for c in LABELS})


# %% CELL 4 — build (features, labels, source-tile group) by GEO-sampling each tile's mask --
sidx = man.sindex
Xtr, ytr, gtr = [], [], []
for ti, (name, foot, (ox, oy, px, py), mask, _cnt) in enumerate(pick):
    h, w = mask.shape
    for ci in sidx.query(foot):
        cell = man.geometry.iloc[ci]
        if not cell.intersects(foot):
            continue
        xs, ys = patch_centres(cell.bounds)
        cols = np.floor((xs - ox) / px).astype(int)         # (GW,)
        rows = np.floor((ys - oy) / py).astype(int)         # (GH,)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")     # (GH,GW)
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        labs = np.zeros((GH, GW), np.uint16)
        labs[inside] = mask[rr[inside], cc[inside]]
        sel = np.isin(labs, LABELS)
        if sel.any():
            pg = np.load(man["npz"].iloc[ci])["patch_grid"]
            Xtr.append(pg[sel]); ytr.append(labs[sel]); gtr.append(np.full(int(sel.sum()), ti))
X = np.concatenate(Xtr).astype(np.float32)
y = np.concatenate(ytr)
g = np.concatenate(gtr)                                      # source tile index -> for grouped split
if MAX_PER_CLASS:                                           # cap per class: balance + speed
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(idx := np.where(y == c)[0], min(len(idx), MAX_PER_CLASS), replace=False)
                           for c in np.unique(y)])
    X, y, g = X[keep], y[keep], g[keep]
u, n = np.unique(y, return_counts=True)
print(f"labelled patches: {len(y)} from {len(np.unique(g))} tiles | per class: {dict(zip(u.tolist(), n.tolist()))}")


# %% CELL 5 — HONEST tile-grouped eval, then refit on everything for the map ---------
def make_clf():
    """L2-normalize (DINO cosine geometry) + the chosen head."""
    if CLASSIFIER == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return make_pipeline(Normalizer(),
                             HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                            max_depth=None, class_weight="balanced"))
    return make_pipeline(Normalizer(), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, class_weight="balanced"))


tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, y, g))
clf = make_clf(); clf.fit(X[tr], y[tr])                     # train on a disjoint SET OF TILES
yp = clf.predict(X[te])
print(f"[{CLASSIFIER}] tile-grouped held-out acc {clf.score(X[te], y[te]):.3f}  "
      f"({len(np.unique(g[tr]))} train / {len(np.unique(g[te]))} test tiles)")
print(classification_report(y[te], yp, zero_division=0))
print("confusion (held-out, true x pred):\n", confusion_matrix(y[te], yp))

clf = make_clf(); clf.fit(X, y)                             # refit on ALL labelled patches -> final map


# %% CELL 6 — predict EVERY patch of EVERY cell, paint a site-wide label canvas -------
bb = np.array([g.bounds for g in man.geometry])
xmin, ymin, xmax, ymax = bb[:, 0].min(), bb[:, 1].min(), bb[:, 2].max(), bb[:, 3].max()
W = int(round((xmax - xmin) / PATCH_M)); H = int(round((ymax - ymin) / PATCH_M))
CODE = {c: i + 1 for i, c in enumerate(LABELS)}              # class id -> compact 1..7 (301 > uint8!)
INV  = {v: k for k, v in CODE.items()}
canvas = np.zeros((H, W), np.uint8)                          # 0 = unpredicted/nodata
for ci in range(len(man)):
    b = man.geometry.iloc[ci].bounds
    pg = np.load(man["npz"].iloc[ci])["patch_grid"].reshape(-1, C)
    coded = np.vectorize(CODE.get)(clf.predict(pg)).astype(np.uint8).reshape(GH, GW)
    col = int(round((b[0] - xmin) / PATCH_M)); row = int(round((ymax - b[3]) / PATCH_M))
    canvas[row:row + GH, col:col + GW] = coded
print(f"canvas {H}x{W} @ {PATCH_M:.2f} m | predicted classes: "
      f"{[INV[v] for v in np.unique(canvas) if v]}")
# NB: nodata octagon areas get SOME class (DINO still embeds black) — cosmetic for this quick test.


# %% CELL 7a — real no-data -> transparent: borrow the alpha of the local PCA webmap -----
# Predictions cover EVERY patch (DINO embeds the black octagon too). Mask those out using the
# already-built local PCA webmap (same patch grid): its alpha band (RGBA) or any non-zero RGB
# (older 3-band) marks where real imagery exists. Fully local — no /mnt webmap needed.
PCA_TIF = f"{EMB}/{SITE_ID}/dino_pca_webmap.tif"
if os.path.exists(PCA_TIF):
    with rasterio.open(PCA_TIF) as r:
        win = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=r.transform)
        if r.count >= 4:                                  # RGBA -> alpha band = data mask
            data = r.read(r.count, window=win, boundless=True, fill_value=0, out_shape=(H, W)) > 0
        else:                                             # RGB  -> data where any band != 0
            data = (r.read(window=win, boundless=True, fill_value=0, out_shape=(3, H, W)) != 0).any(0)
    canvas[~data] = 0
    print(f"masked {int((~data).sum())} no-data px via {os.path.basename(PCA_TIF)}")
else:
    print("no PCA webmap found -> only inter-cell gaps will be transparent")


# %% CELL 7b — show it + save a QGIS-ready GeoTIFF (transparent no-data) --------------
# colours per the QGIS legend (Ground white, Shrub bright red, Tree vivid dark blue,
# Hummock apple green, Tussock ochre/orange-brown); Herb/Sedge unspecified -> kept distinct.
CLS_RGB = {2:  (255, 255, 255),   # Ground  — white
           3:  (124, 197,  40),   # HumGras — apple green   ("Hummock")
           4:  (227,  26,  28),   # Shrub   — bright red
           5:  ( 20,  50, 200),   # Tree    — vivid dark blue
           6:  (255, 237, 111),   # Herb    — pale yellow (not specified)
           40: (102, 194, 165),   # Sedge   — teal (not specified)
           301:(204, 145,  44)}   # TusGra  — ochre/orange-brown  ("Tussock")
PALETTE = {0: (0, 0, 0), **{CODE[c]: CLS_RGB[c] for c in LABELS}}   # storage code -> rgb
transform = Affine(PATCH_M, 0, xmin, 0, -PATCH_M, ymax)

rgb = np.full((H, W, 3), 30, np.uint8)                             # dark bg so white Ground pops (PNG)
for k, c in PALETTE.items():
    if k != 0:
        rgb[canvas == k] = c
present = [k for k in PALETTE if k != 0 and (canvas == k).any()]
handles = [Patch(facecolor=np.array(PALETTE[k]) / 255, edgecolor="k",
                 label=f"{INV[k]}: {class_name(INV[k])}") for k in present]
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(rgb); ax.axis("off")
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
ax.set_title(f"dry oasis — logreg on {N_TRAIN_TILES} tiles | 7-class lifeform scheme")
fig.savefig(f"{OUT}/classif_oasis.png", dpi=130, bbox_inches="tight"); plt.show()

# single-band paletted GeoTIFF: code 0 -> alpha 0 (transparent) + nodata=0 -> QGIS hides it
colormap = {0: (0, 0, 0, 0), **{CODE[c]: (*CLS_RGB[c], 255) for c in LABELS}}
with rasterio.open(f"{OUT}/classif_oasis.tif", "w", driver="GTiff", height=H, width=W, count=1,
                   dtype="uint8", crs=man.crs, transform=transform, nodata=0,
                   compress="DEFLATE", tiled=True) as dst:
    dst.write(canvas, 1)
    dst.write_colormap(1, colormap)
print(f"-> {OUT}/classif_oasis.png  +  {OUT}/classif_oasis.tif")
