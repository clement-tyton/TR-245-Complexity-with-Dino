"""Central config: paths, the activity's resolution policy, tunables, and env setup.

Importing this module sets the environment variables the DINOv3 activity needs (weights
folder, local S3) BEFORE the activity is ever imported. Paths can be overridden via env.
"""
from __future__ import annotations

import os
import re
import warnings

warnings.filterwarnings("ignore")   # silence rio-tiler NoOverviewWarning etc. across the pipeline

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PKG_DIR)

# ---- paths (env-overridable) ----
TRAIN_ROOT = os.environ.get("DINO_BLOB_TRAIN_ROOT", "/home/clement/local_copy_train_data")
WORK_DIR = os.environ.get("DINO_BLOB_WORK_DIR", os.path.join(_PROJECT_ROOT, "blob_work"))
WEIGHTS_FOLDER = os.environ.get(
    "DINO_WEIGHTS_FOLDER",
    "/home/clement/Desktop/projets/1_Core_tyton_AI/tytonai-python-activities/"
    "dinov3_embedding/test_data/dinov3_weights",
)
STATS_PARQUET = os.environ.get(
    "DINO_BLOB_STATS_PARQUET", os.path.join(_PROJECT_ROOT, "tiles_stat_db", "site_resolution.parquet"))

# ---- env the activity reads (set on import, before the activity is loaded) ----
os.environ["DINO_WEIGHTS_FOLDER"] = WEIGHTS_FOLDER
os.environ.setdefault("S3_FILE_BUCKET", "")          # S3Mock -> plain local files
os.environ.setdefault("SAVE_DEBUG_IMG", "false")

# ---- tunables (defaults = the cheap/tractable choice) ----
DINO_MODEL = "dinov3_vitl16"     # or "dinov3_vit7b16"
HIGH_RES = False                 # True -> 2x finer embedding, ~4x GPU mem
TILE_PATCHES = 1                 # bbox native size = TILE_PATCHES * patch_size
OVERLAP_CELLS = 16               # neighbour overlap, in EMBED cells

# ---- constants ----
TILE_RE = re.compile(r"^(.+)_(\d+)_(\d+)\.npz$")   # generic <key>_<col>_<row>.npz (greedy)
RGB_KEYS = ("RED", "GREEN", "BLUE")
HIGH_RES_T, MED_RES_T, LOW_RES_T = 0.07, 0.15, 0.3  # activity resolution thresholds
DINO_PATCH = 16


def site_id_from_dir(site_dir: str) -> str:
    """Stable, collision-free site id from the path, e.g. '29Metals__<site>__10cm__v2_tytonai_rg'."""
    return os.path.relpath(site_dir, TRAIN_ROOT).replace(os.sep, "__")


def activity_params(native_res: float, high_res: bool = False) -> dict:
    """patch_size / upsample / embed_gsd for a native resolution — mirrors the activity exactly."""
    if native_res < HIGH_RES_T:
        patch_size, upsample = 1024, 1
    elif native_res < MED_RES_T:
        patch_size, upsample = 512, 2
    else:
        patch_size, upsample = 256, 4
    if high_res:
        upsample *= 2
    return {"patch_size": patch_size, "upsample": upsample,
            "embed_gsd": native_res * DINO_PATCH / upsample}
