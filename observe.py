# %% [markdown]
# observe.py — read-only QA viewer (safe to run WHILE the pipeline is embedding)
# ============================================================================
# Reads finished outputs under blob_work/ and plots them (CPU only — no GPU, no
# writes to anything the run touches). Select a site / blob by partial name.
#
#   status()                      -> sites in blob_work + embed progress
#   blobs("29Metals", done=True)  -> blobs of a site (done = has embedding.tif)
#   view("29Metals","f7400","embedding")  # RGB vs PCA-RGB   (also "cluster", "boxes")
#   grid("29Metals")              -> whole-site box grid

# %% CELL 1 — helpers ---------------------------------------------------------------
import glob
import os

import dino_blob as db
from dino_blob import config

WORK = config.WORK_DIR


def _dirs(pattern):
    return [d for d in glob.glob(pattern) if os.path.isdir(d)]


def _pick(parent, needle, kind):
    m = [d for d in _dirs(os.path.join(parent, "*")) if needle.lower() in os.path.basename(d).lower()]
    if not m:
        print(f"no {kind} matching '{needle}' under {parent}"); return None
    if len(m) > 1:
        print(f"{len(m)} {kind}s match '{needle}', using:", os.path.basename(m[0]))
    return m[0]


def show_png(path):
    """Display a saved PNG inline (VS Code interactive) / in a window (plain REPL)."""
    import matplotlib.pyplot as plt
    img = plt.imread(path)
    fig = plt.figure(figsize=(min(18, img.shape[1] / 100 + 2), min(11, img.shape[0] / 100 + 1)))
    plt.imshow(img); plt.axis("off")
    plt.title(f"{os.path.basename(os.path.dirname(path))} / {os.path.basename(path)}", fontsize=8)
    plt.show()


def status():
    """Every site in blob_work with <embedded>/<total> blob counts."""
    rows = []
    for sd in sorted(_dirs(os.path.join(WORK, "*"))):
        blobs = _dirs(os.path.join(sd, "*"))
        done = sum(os.path.exists(os.path.join(b, "embedding.tif")) for b in blobs)
        rows.append((os.path.basename(sd), len(blobs), done))
        print(f"  {done:3d}/{len(blobs):<3d}  {os.path.basename(sd)}")
    return rows


def blobs(site, done=False):
    """List blobs of a site (partial name ok); done=True -> only embedded ones."""
    sd = _pick(WORK, site, "site")
    if not sd:
        return []
    out = []
    for b in sorted(_dirs(os.path.join(sd, "*"))):
        ok = os.path.exists(os.path.join(b, "embedding.tif"))
        if done and not ok:
            continue
        out.append((os.path.basename(b), ok))
        print(f"  {'OK ' if ok else '...'}  {os.path.basename(b)}")
    return out


def view(site, blob, what="embedding", k=6):
    """Plot one blob: what in {'embedding' (RGB vs PCA-RGB), 'cluster' (KMeans), 'boxes'}."""
    sd = _pick(WORK, site, "site")
    if not sd:
        return None
    d = _pick(sd, blob, "blob")
    if not d:
        return None
    has_emb = os.path.exists(os.path.join(d, "embedding.tif"))
    if what in ("embedding", "cluster") and not has_emb:
        print(f"{os.path.basename(d)} not embedded yet (no embedding.tif) — try what='boxes'")
        return None
    png = (db.plot_blob_vs_embedding(d) if what == "embedding"
           else db.cluster_blob(d, k=k) if what == "cluster"
           else db.plot_blob_boxes(d))
    print("->", png)
    show_png(png)
    return png


def grid(site):
    """Whole-site box grid (all blobs built so far)."""
    sd = _pick(WORK, site, "site")
    if not sd:
        return None
    dirs = {os.path.basename(b): b for b in _dirs(os.path.join(sd, "*"))}
    png = db.plot_site_boxes_grid(dirs, os.path.join(sd, "_boxes_grid.png"))
    show_png(png)
    return png


# %% CELL 2 — what's available right now --------------------------------------------
status()

# %% CELL 3 — blobs of a site (only the embedded ones) ------------------------------
blobs("29Metals", done=True)

# %% CELL 4 — observe a blob ---------------------------------------------------------
view("29Metals", "f7400", "embedding")    # RGB | PCA-RGB of the DINO embedding
# view("29Metals", "f7400", "cluster")    # RGB | PCA-RGB | KMeans label map
# view("29Metals", "e127",  "boxes")      # bbox outlines + coverage heatmap

# %% CELL 5 — whole-site box grid ----------------------------------------------------
# grid("29Metals")
