# %% [markdown]
# DINOv3 blob embedding — REPL walkthrough
# =========================================
# Thin driver over the `dino_blob` package. Run cell by cell (the `# %%` markers).
# Interpreter: this project's venv
#   /home/clement/Desktop/projets/2_Actual_jira_tickets/complexity_with_dino/.venv/bin/python
#
# The pipeline (per site): reconstitute blobs -> blob.tif + overlapping boxes.fgb ->
# DINOv3 activity per box -> mean-blend -> embedding.tif (+ QA pngs + metadata).
# Code lives in dino_blob/: config, core (blobs/boxes/metadata), embedding, plots,
# pipeline (run_site), multi (run_all_sites, 2-GPU).

# %% CELL 1 — imports + pick a site -------------------------------------------------
import os

import dino_blob as db
from dino_blob import config

SITE_DIR = os.path.join(config.TRAIN_ROOT, "29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg")
print("site_id:", config.site_id_from_dir(SITE_DIR))


# %% CELL 2 — discover blobs (no heavy work) ----------------------------------------
blobs = db.discover_blobs(SITE_DIR)
print(f"{len(blobs)} blobs")
for uid, meta in sorted(blobs.items(), key=lambda kv: -__import__('numpy').prod(kv[1].extent_px))[:5]:
    print(f"  {uid[:8]}  {meta.extent_px[0]}x{meta.extent_px[1]}px  res={meta.native_res:.3f}  "
          f"{len(meta.cells)} tiles")


# %% CELL 3 — build blobs + boxes + grid (fast, no DINO) -----------------------------
# Writes blob.tif + boxes.fgb + metadata.json per blob, and _boxes_grid.png for the site.
summary = db.run_site(SITE_DIR, embed=False)
site_out = summary["out"]
print("box grid ->", os.path.join(site_out, "_boxes_grid.png"))


# %% CELL 4 — embed ONE blob + assess ------------------------------------------------
# Pick any blob dir; embed it, then look at RGB vs embedding and a KMeans clustering.
import numpy as np
uid = max(blobs, key=lambda u: np.prod(blobs[u].extent_px))      # biggest blob
db.run_site(SITE_DIR, uids=[uid[:8]], embed=True, make_grid=False)
d = summary["blob_dirs"][uid]
print("assess     ->", db.plot_blob_vs_embedding(d))
print("clustering ->", db.cluster_blob(d, k=6))


# %% CELL 5 — embed the WHOLE site (clean bar + per-box sub-bar) ---------------------
summary = db.run_site(SITE_DIR)              # nested progress: blobs (outer) + boxes (inner)


# %% CELL 6 — ALL SITES across both GPUs (one call) ---------------------------------
# Equivalent CLI:  python run_sites.py --all --gpus 0,1
sites = db.list_sites()
print(f"{len(sites)} sites on disk")
# db.run_all_sites(sites, gpus=(0, 1))       # <- uncomment to launch the full multi-site run
