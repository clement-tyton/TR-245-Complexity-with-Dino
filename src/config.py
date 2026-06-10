"""Central config: paths, env setup (load-bearing), and per-site RGB resolution.

Importing this module sets the environment variables the DINOv3 activity needs BEFORE the
activity is ever imported. Every other src module does ``import config`` first; entry points
(pipeline.py, repl_onesite.py) put ``src/`` on sys.path, so this always runs before any code
path can reach the activity. Keep the env side-effect ONLY here.
"""
from __future__ import annotations

import functools
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")   # silence rio-tiler NoOverviewWarning etc. across the pipeline

# ---- env the dinov3 activity reads (MUST be set before importing it) --------------
os.environ["DINO_WEIGHTS_FOLDER"] = os.environ.get(
    "DINO_WEIGHTS_FOLDER",
    "/home/clement/Desktop/projets/1_Core_tyton_AI/tytonai-python-activities/"
    "dinov3_embedding/test_data/dinov3_weights")
os.environ.setdefault("S3_FILE_BUCKET", "")          # empty -> S3Mock uses plain local files
os.environ.setdefault("SAVE_DEBUG_IMG", "false")

# ---- paths (resolved from this file, not cwd; all env-overridable) ----
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
TRAIN_ROOT = os.environ.get("DINO_TRAIN_ROOT", "/home/clement/local_copy_train_data")
SITE_DATA_ROOT = os.environ.get("DINO_SITE_DATA_ROOT", "/mnt/spatial/DeepThought/SiteData")  # shared store
DATASET_VERSION = os.environ.get("DINO_DATASET_VERSION", "v2_tytonai_rg")  # = config dataset_version
CONFIG_DIR = os.environ.get("DINO_CONFIG_DIR", os.path.join(_REPO_ROOT, "config"))
PIC_DIR = os.environ.get("DINO_PIC_DIR", os.path.join(_REPO_ROOT, "outputs", "pictures"))
EMB_ROOT = os.environ.get("DINO_EMB_ROOT", "/mnt/ai/DeepThought/dino_embeddings")  # shared net store
STATS_PARQUET = os.path.join(CONFIG_DIR, "tiles_stat_db", "site_resolution.parquet")  # multi-site only
os.makedirs(PIC_DIR, exist_ok=True)

# ---- tunables ----
DINO_MODEL = os.environ.get("DINO_MODEL", "dinov3_vitl16")   # or "dinov3_vit7b16"
HIGH_RES = os.environ.get("DINO_HIGH_RES", "0") == "1"       # doubles the upsample factor
TILE_PATCHES = 1            # grid cell native size = TILE_PATCHES * patch_size
MIN_DATA_COV = 0.02         # drop grid cells with < this fraction of RGB data (all-black voids)
DINO_PATCH = 16

# ---- the dinov3 activity's resolution policy (m/px thresholds) ----
HIGH_RES_T, MED_RES_T = 0.07, 0.15


def site_id_from_dir(site_dir: str, train_root: str = TRAIN_ROOT) -> str:
    """Stable, collision-free site id from the path (matches the original REPL exactly).

    e.g. 'BHP Creeks 2022/.../v2_tytonai_rg' -> 'BHP_Creeks_2022_..._v2_tytonai_rg'.
    """
    rel = os.path.relpath(site_dir, train_root)
    return re.sub(r"[^0-9A-Za-z]+", "_", rel).strip("_")


def activity_params(native_res: float, high_res: bool = False) -> dict:
    """patch_size / upsample / embed_gsd for a native resolution — mirrors the activity."""
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


@functools.lru_cache(maxsize=1)
def _load_json(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return json.load(f)


def site_key_from_dir(site_dir: str, train_root: str = TRAIN_ROOT) -> str:
    """Map a SITE_DIR to the '<Project>/<Site>' key used in the config JSONs.

    The training path is '<Project>/<Site>/<res>/v2_tytonai_rg'; the JSON keys are
    '<Project>/<Site>'. So we take the relative path minus its last two components.
    """
    rel = os.path.relpath(site_dir, train_root)
    parts = rel.split(os.sep)
    return os.sep.join(parts[:-2]) if len(parts) >= 2 else rel


def default_res(site_key: str) -> str:
    """The site's native/first resolution from config/sites_to_resolutions.json."""
    rr = _load_json("sites_to_resolutions.json").get(site_key, {}).get("resolutions")
    if not rr:
        raise KeyError(f"site '{site_key}' not in sites_to_resolutions.json")
    return rr[0]


def site_id_from_key(site_key: str, res: str) -> str:
    """Output-path site_id from a '<Project>/<Site>' key + resolution. Matches
    site_id_from_dir() exactly so /mnt-driven and local-driven runs share output paths."""
    return re.sub(r"[^0-9A-Za-z]+", "_", f"{site_key}/{res}/{DATASET_VERSION}").strip("_")


def resolve_tiles(site_key: str, res: str | None = None) -> str:
    """Training-tile dir on the shared /mnt store (the multi-site source — no local copy).

    Layout: <SITE_DATA_ROOT>/<Project>/<Site>/Raster/ObjectData/<res>/<DATASET_VERSION>/{train,val}.
    This is the SOURCE side of the copy_sitedata_local.sh script (which drops Raster/ObjectData/).
    """
    res = res or default_res(site_key)
    d = os.path.join(SITE_DATA_ROOT, site_key, "Raster", "ObjectData", res, DATASET_VERSION)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no training tiles at {d}")
    return d


def resolve_rgb(site_key: str, res: str | None = None) -> dict:
    """Resolve a site's RGB raster from config/site_rgb_paths.json. Returns the entry dict
    ({rgb_path, source, bands, crs, size, ...}). If res is None, pick the site's resolution
    from config/sites_to_resolutions.json (first listed)."""
    paths = _load_json("site_rgb_paths.json")
    if site_key not in paths:
        raise KeyError(f"site '{site_key}' not in site_rgb_paths.json "
                       f"({len(paths)} sites; e.g. {list(paths)[:3]})")
    by_res = paths[site_key]
    if res is None:
        res = default_res(site_key) if site_key in _load_json("sites_to_resolutions.json") else list(by_res)[0]
    if res not in by_res:
        raise KeyError(f"resolution '{res}' not for site '{site_key}' (have {list(by_res)})")
    rec = by_res[res]
    if not rec.get("rgb_path"):
        raise KeyError(f"site '{site_key}' @ {res} has no resolved rgb_path (source={rec.get('source')})")
    return rec
