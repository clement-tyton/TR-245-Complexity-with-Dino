#!/usr/bin/env python3
"""Incrementally update the tile-statistics parquet DB under tiles_stat_db/.

The DB was originally built by
  segmentation-tytonai/scripts/build_tile_stats_db.py
which scans every mask NPZ under the local tile cache and writes three parquet
files. That scan is ~3 weeks old and ~1,100 tiles have since been added to disk.

This script does the same per-tile computation but only for tiles **not already
present** in tiles.parquet, appends them, and rebuilds the per-site rollup. The
per-tile logic (no-data set, seven-class rollup, bbox signature) is copied
verbatim from the original so the new rows are byte-for-byte comparable — run
`--validate` to prove it against rows already in the DB.

Outputs (all under tiles_stat_db/):
  - tiles.parquet              one row per mask NPZ (wide: 7-class rollup)
  - tiles_orig_classes.parquet long format, one row per (tile, original_class)
  - site_resolution.parquet    aggregates per (site x resolution)

Usage:
  python update_tile_stats_db.py                 # incremental append (default)
  python update_tile_stats_db.py --validate 200  # check parity, write nothing
  python update_tile_stats_db.py --prune         # also drop rows whose mask is gone
  python update_tile_stats_db.py --full          # rebuild from scratch
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
LOCAL_ROOT = Path("/home/clement/local_copy_train_data")
OUTPUT_DIR = Path(__file__).resolve().parent / "tiles_stat_db"
MAP_TYPE = "seven_class"
ROLLED_UP_CLASSES: list[int] = [2, 3, 4, 5, 6, 40, 301]

logger = logging.getLogger("update_tile_stats_db")

# ── vendored from segmentation-tytonai/utils/utils.py (provenance: keep in sync)
# Pixel values that mean "no annotation" rather than a real class.
no_data_class_list = [0, 15, 127, -128, 255, 65535]
NO_DATA_SET = set(int(v) for v in no_data_class_list)

# Original-class -> seven-class string remap. Keys/values are strings exactly as
# in the source. Any class not listed (and not no-data) makes gt_mask_mapping
# raise, which we treat as a corrupted tile (same as the original scanner).
seven_class_mapping_dict = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '40': '40', '301': '301',
    '7': '301', '9': '2', '201': '2', '200': '2', '9003': '3',
    '5007': '4', '401': '4', '9004': '4', '402': '4', '501': '4', '5203': '4',
    '4900': '4', '450': '4',
    '601': '5', '603': '5', '607': '5', '5103': '5', '5114': '5', '6008': '5',
    '6012': '5', '6015': '5', '6016': '5',
    '70': '6', '302': '6', '1791': '6',
    '303': '301', '10001': '4', '10003': '5', '10004': '5', '10005': '5',
    '10006': '5', '10007': '4', '10008': '5', '10009': '301', '10010': '5',
    '10011': '4', '10012': '5', '10013': '5', '10050': '6', '10092': '4',
}


def gt_mask_mapping(gt_mask, class_values, missing_class_map_type):
    """Verbatim copy of utils.utils.gt_mask_mapping (seven_class path only)."""
    if missing_class_map_type != "seven_class":
        raise ValueError(f"only seven_class supported here, got {missing_class_map_type}")
    mapping_dict = seven_class_mapping_dict
    all_classes_list = [int(v) for v in mapping_dict.keys()]

    unique_classes = np.unique(gt_mask)
    invalid_classes = np.setdiff1d(unique_classes, all_classes_list + no_data_class_list)
    if invalid_classes.size > 0:
        raise Exception(
            f"Classes {invalid_classes} not found in tytonAI possible class map "
            f"{all_classes_list} or no data classes {no_data_class_list}"
        )

    class_mapping = {}
    for v in unique_classes:
        if v not in class_values:
            if str(v) in mapping_dict.keys():
                class_mapping[v] = int(mapping_dict[str(v)])
                if class_mapping[v] not in class_values:
                    class_mapping[v] = 0
            else:
                class_mapping[v] = 0

    for k, v in class_mapping.items():
        gt_mask[gt_mask == k] = v

    return gt_mask


# ── walk ─────────────────────────────────────────────────────────────────────
def discover_tiles(root: Path) -> list[dict]:
    """Walk the local cache and return one descriptor per mask NPZ found."""
    tiles: list[dict] = []
    if not root.exists():
        raise FileNotFoundError(f"local root not found: {root}")

    for annot_dir in root.rglob("*annot"):
        if not annot_dir.is_dir() or annot_dir.name not in ("trainannot", "valannot"):
            continue
        try:
            dataset_version = annot_dir.parent.name
            resolution = annot_dir.parent.parent.name
            site = str(annot_dir.parent.parent.parent.relative_to(root))
        except ValueError:
            logger.warning("Could not derive site from %s — skipping", annot_dir)
            continue

        split = annot_dir.name.replace("annot", "")  # "train" or "val"
        image_dir = annot_dir.parent / split

        for mask_path in sorted(annot_dir.glob("*.npz")):
            image_path = _resolve_image_path(image_dir, mask_path.name)
            tiles.append({
                "site": site,
                "resolution": resolution,
                "dataset_version": dataset_version,
                "split": split,
                "mask_path": str(mask_path),
                "image_path": str(image_path),
                "image_exists": image_path.exists(),
            })
    return tiles


def _resolve_image_path(image_dir: Path, mask_name: str) -> Path:
    """Find the RGB tile paired with a mask.

    Sites differ in naming: some share the filename between mask and image,
    others use a ``mask_<id>`` / ``image_<id>`` prefix pair. The original
    scanner only tried the identical name, so prefix-pair sites were wrongly
    recorded as image_exists=False. We try the identical name first, then the
    ``mask_`` -> ``image_`` substitution. Returns the identical-name candidate
    (non-existent) if neither is found, so .exists() reports False as before.
    """
    same = image_dir / mask_name
    if same.exists():
        return same
    if mask_name.startswith("mask_"):
        alt = image_dir / ("image_" + mask_name[len("mask_"):])
        if alt.exists():
            return alt
    return same


# ── per-tile work (verbatim from the original scanner) ───────────────────────
def process_mask(descriptor: dict) -> "tuple[dict, list[dict]] | None":
    """Read one mask NPZ and compute pixel counts.

    Returns (tile_row, orig_class_rows) or None on failure.
    """
    mask_path = descriptor["mask_path"]
    try:
        with np.load(mask_path, allow_pickle=True) as d:
            if "CLASSIFY" not in d.files:
                logger.warning("No CLASSIFY key in %s", mask_path)
                return None
            arr = d["CLASSIFY"]
        if arr.ndim != 2:
            logger.warning("Unexpected shape %s in %s", arr.shape, mask_path)
            return None
    except Exception as exc:
        logger.warning("Failed to load %s: %s", mask_path, exc)
        return None

    H, W = int(arr.shape[0]), int(arr.shape[1])
    total = H * W

    vals, counts = np.unique(arr, return_counts=True)
    vals_int = [int(v) for v in vals]
    counts_int = [int(c) for c in counts]

    pixels_no_data = sum(c for v, c in zip(vals_int, counts_int) if v in NO_DATA_SET)

    # Spatial signature of the annotated region — detects partial-edge tiles.
    annotated_mask_2d = ~np.isin(arr, list(NO_DATA_SET))
    if annotated_mask_2d.any():
        rows_any = annotated_mask_2d.any(axis=1)
        cols_any = annotated_mask_2d.any(axis=0)
        y0, y1 = np.where(rows_any)[0][[0, -1]]
        x0, x1 = np.where(cols_any)[0][[0, -1]]
        bbox_h = int(y1 - y0 + 1)
        bbox_w = int(x1 - x0 + 1)
        bbox_area = bbox_h * bbox_w
        annot_bbox_area_frac = float(bbox_area) / total if total else 0.0
        annot_in_bbox_density = (
            float(total - pixels_no_data) / bbox_area if bbox_area else 0.0
        )
    else:
        annot_bbox_area_frac = 0.0
        annot_in_bbox_density = 0.0

    class_counts: dict[str, int] = {f"pixels_class_{k}": 0 for k in ROLLED_UP_CLASSES}
    try:
        remapped = gt_mask_mapping(arr.copy(), ROLLED_UP_CLASSES, MAP_TYPE)
    except Exception as exc:
        logger.warning("Rollup failed on %s: %s", mask_path, exc)
        return None

    vals_r, counts_r = np.unique(remapped, return_counts=True)
    for v, c in zip(vals_r, counts_r):
        v_int = int(v)
        if v_int in ROLLED_UP_CLASSES:
            class_counts[f"pixels_class_{v_int}"] = int(c)

    pixels_unmapped = total - pixels_no_data - sum(class_counts.values())
    no_data_frac = float(pixels_no_data) / total if total else 0.0

    tile_id = Path(mask_path).stem
    tile_row: dict = {
        "tile_id": tile_id,
        "site": descriptor["site"],
        "resolution": descriptor["resolution"],
        "dataset_version": descriptor["dataset_version"],
        "split": descriptor["split"],
        "mask_path": mask_path,
        "image_path": descriptor["image_path"] if descriptor["image_exists"] else None,
        "image_exists": descriptor["image_exists"],
        "height": H,
        "width": W,
        "total_pixels": total,
        "pixels_no_data": int(pixels_no_data),
        **class_counts,
        "pixels_unmapped": int(pixels_unmapped),
        "no_data_frac": float(no_data_frac),
        "n_unique_orig_classes": int(len(vals_int)),
        "annot_bbox_area_frac": float(annot_bbox_area_frac),
        "annot_in_bbox_density": float(annot_in_bbox_density),
    }

    orig_rows = [
        {
            "tile_id": tile_id,
            "site": descriptor["site"],
            "resolution": descriptor["resolution"],
            "split": descriptor["split"],
            "original_class_id": v,
            "pixel_count": c,
        }
        for v, c in zip(vals_int, counts_int)
    ]
    return tile_row, orig_rows


# ── aggregation (verbatim from the original scanner) ─────────────────────────
def build_site_resolution(tiles_df: pd.DataFrame, orig_df: pd.DataFrame) -> pd.DataFrame:
    class_cols = [f"pixels_class_{k}" for k in ROLLED_UP_CLASSES]

    grouped = tiles_df.groupby(["site", "resolution"], as_index=False)
    base = grouped.agg(
        dataset_version=("dataset_version", "max"),
        n_tiles_total=("tile_id", "count"),
        total_pixels=("total_pixels", "sum"),
        total_no_data_pixels=("pixels_no_data", "sum"),
        mean_no_data_frac=("no_data_frac", "mean"),
        median_no_data_frac=("no_data_frac", "median"),
        p90_no_data_frac=("no_data_frac", lambda s: float(np.quantile(s, 0.9))),
        frac_tiles_mostly_no_data=("no_data_frac", lambda s: float((s > 0.9).mean())),
        **{c: (c, "sum") for c in class_cols},
    )

    by_split = (
        tiles_df.groupby(["site", "resolution", "split"])["tile_id"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    by_split = by_split.rename(columns={"train": "n_tiles_train", "val": "n_tiles_val"})
    for col in ("n_tiles_train", "n_tiles_val"):
        if col not in by_split.columns:
            by_split[col] = 0

    orig_union = (
        orig_df.groupby(["site", "resolution"])["original_class_id"]
        .nunique()
        .reset_index()
        .rename(columns={"original_class_id": "n_unique_orig_classes_union"})
    )

    out = base.merge(by_split, on=["site", "resolution"], how="left")
    out = out.merge(orig_union, on=["site", "resolution"], how="left")
    out["n_unique_orig_classes_union"] = out["n_unique_orig_classes_union"].fillna(0).astype("int16")
    out["scanned_at"] = pd.Timestamp.now(tz="UTC")

    front = [
        "site", "resolution", "dataset_version",
        "n_tiles_train", "n_tiles_val", "n_tiles_total",
        "total_pixels", "total_no_data_pixels",
        "mean_no_data_frac", "median_no_data_frac", "p90_no_data_frac",
        "frac_tiles_mostly_no_data",
    ]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


# ── helpers ──────────────────────────────────────────────────────────────────
TILES_COLS_FRONT = [
    "tile_id", "site", "resolution", "dataset_version", "split",
    "mask_path", "image_path", "image_exists",
    "height", "width", "total_pixels",
    "pixels_no_data", *[f"pixels_class_{k}" for k in ROLLED_UP_CLASSES], "pixels_unmapped",
    "no_data_frac", "annot_bbox_area_frac", "annot_in_bbox_density",
    "n_unique_orig_classes", "scanned_at",
]


def _coerce_tiles_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in TILES_COLS_FRONT if c in df.columns]].copy()
    for col in ("height", "width"):
        df[col] = df[col].astype("int32")
    df["n_unique_orig_classes"] = df["n_unique_orig_classes"].astype("int16")
    df["no_data_frac"] = df["no_data_frac"].astype("float32")
    df["annot_bbox_area_frac"] = df["annot_bbox_area_frac"].astype("float32")
    df["annot_in_bbox_density"] = df["annot_in_bbox_density"].astype("float32")
    return df


def _process_many(descriptors: list[dict], workers: int, desc: str):
    """Run process_mask over descriptors with a thread pool. Returns (tile_rows, orig_rows)."""
    tile_rows: list[dict] = []
    orig_rows: list[dict] = []
    if not descriptors:
        return tile_rows, orig_rows
    try:
        from tqdm import tqdm
        wrap = lambda it: tqdm(it, total=len(descriptors), desc=desc)
    except ImportError:
        wrap = lambda it: it
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_mask, d): d for d in descriptors}
        for fut in wrap(as_completed(futs)):
            result = fut.result()
            if result is None:
                continue
            tile_row, orig = result
            tile_rows.append(tile_row)
            orig_rows.extend(orig)
    return tile_rows, orig_rows


# ── modes ────────────────────────────────────────────────────────────────────
def run_validate(sample: int, workers: int) -> int:
    """Recompute `sample` rows already in the DB and assert exact match."""
    tiles_out = OUTPUT_DIR / "tiles.parquet"
    if not tiles_out.exists():
        logger.error("No existing tiles.parquet to validate against.")
        return 1
    existing = pd.read_parquet(tiles_out)
    # Only rows whose mask still exists on disk can be recomputed.
    existing = existing[existing["mask_path"].apply(lambda p: Path(p).exists())]
    if existing.empty:
        logger.error("No existing rows have masks on disk — cannot validate.")
        return 1
    sample = min(sample, len(existing))
    # Deterministic sample (head) so re-runs are reproducible without RNG.
    sub = existing.head(sample)
    descriptors = sub[[
        "site", "resolution", "dataset_version", "split",
        "mask_path", "image_path", "image_exists",
    ]].to_dict("records")
    logger.info("Recomputing %d existing rows for parity check ...", len(descriptors))

    num_cols = [
        "height", "width", "total_pixels", "pixels_no_data",
        *[f"pixels_class_{k}" for k in ROLLED_UP_CLASSES], "pixels_unmapped",
        "no_data_frac", "annot_bbox_area_frac", "annot_in_bbox_density",
        "n_unique_orig_classes",
    ]
    mismatches = 0
    for desc in descriptors:
        res = process_mask(desc)
        if res is None:
            logger.warning("process_mask returned None for %s", desc["mask_path"])
            mismatches += 1
            continue
        new_row, _ = res
        old_row = existing[existing["mask_path"] == desc["mask_path"]].iloc[0]
        for c in num_cols:
            a, b = new_row[c], old_row[c]
            if isinstance(a, float) or isinstance(b, float):
                ok = abs(float(a) - float(b)) <= 1e-5 * max(1.0, abs(float(b)))
            else:
                ok = a == b
            if not ok:
                logger.error("MISMATCH %s col=%s recomputed=%s stored=%s",
                             desc["mask_path"], c, a, b)
                mismatches += 1
    if mismatches == 0:
        logger.info("✅ Parity confirmed: %d rows match exactly across %d columns.",
                    len(descriptors), len(num_cols))
        return 0
    logger.error("❌ %d mismatches found — do NOT trust incremental output.", mismatches)
    return 1


def _load_classify(mask_path: str):
    try:
        with np.load(mask_path, allow_pickle=True) as d:
            return d["CLASSIFY"] if "CLASSIFY" in d.files else None
    except Exception:
        return None


def run_dedup(workers: int, verify: bool) -> int:
    """Drop redundant duplicate-mask rows that share one RGB image.

    Some leaves store each tile's mask twice (canonical ``mask_<id>`` plus a
    redundant ``image_<id>`` copy in the annot dir); both rows then point at the
    same RGB tile. Within each colliding group we keep the ``mask_``-named row
    and drop the rest. With ``verify`` (default) we first confirm the dropped
    masks are byte-identical to the kept one; any group that disagrees is left
    untouched and reported, so real labelling conflicts are never hidden.
    """
    tiles_out = OUTPUT_DIR / "tiles.parquet"
    orig_out = OUTPUT_DIR / "tiles_orig_classes.parquet"
    site_out = OUTPUT_DIR / "site_resolution.parquet"
    if not tiles_out.exists():
        logger.error("No tiles.parquet to dedup.")
        return 1

    tiles_df = pd.read_parquet(tiles_out)
    orig_df = pd.read_parquet(orig_out) if orig_out.exists() else pd.DataFrame()

    dup = tiles_df[tiles_df.duplicated("image_path", keep=False) & tiles_df["image_exists"]]
    if dup.empty:
        logger.info("No duplicate image_path groups — nothing to dedup.")
        return 0
    n_groups = dup["image_path"].nunique()
    logger.info("Found %d rows across %d shared-image groups.", len(dup), n_groups)

    drop_keys: list[tuple] = []      # (mask_path,) rows to drop from tiles_df
    conflicts = 0
    for image_path, g in dup.groupby("image_path"):
        names = g["mask_path"].apply(lambda p: Path(p).name)
        canon = g[names.str.startswith("mask_")]
        keep_row = (canon.iloc[0] if len(canon) else g.iloc[0])
        losers = g[g["mask_path"] != keep_row["mask_path"]]

        if verify:
            keep_arr = _load_classify(keep_row["mask_path"])
            bad = False
            for _, lr in losers.iterrows():
                la = _load_classify(lr["mask_path"])
                if keep_arr is None or la is None or keep_arr.shape != la.shape \
                        or not np.array_equal(keep_arr, la):
                    bad = True
                    break
            if bad:
                conflicts += 1
                logger.warning("CONFLICT — masks differ for image %s; keeping all %d rows.",
                               Path(image_path).name, len(g))
                continue
        drop_keys.extend((mp,) for mp in losers["mask_path"])

    if not drop_keys:
        logger.info("Nothing dropped (%d conflicts kept intact).", conflicts)
        return 0

    drop_paths = {k[0] for k in drop_keys}
    dropped = tiles_df[tiles_df["mask_path"].isin(drop_paths)]
    # Anti-join orig_df on the dropped rows' identity (tile_id, site, resolution, split).
    if not orig_df.empty:
        key_cols = ["tile_id", "site", "resolution", "split"]
        drop_idx = dropped[key_cols].drop_duplicates()
        merged = orig_df.merge(drop_idx.assign(_drop=1), on=key_cols, how="left")
        orig_df = orig_df[merged["_drop"].isna().values].reset_index(drop=True)

    before = len(tiles_df)
    tiles_df = tiles_df[~tiles_df["mask_path"].isin(drop_paths)].reset_index(drop=True)
    logger.info("Dropping %d redundant rows (%d conflicts preserved). %d -> %d tiles.",
                len(drop_paths), conflicts, before, len(tiles_df))

    logger.info("Rebuilding per-(site, resolution) rollup ...")
    site_df = build_site_resolution(tiles_df, orig_df)

    logger.info("Writing parquet files to %s ...", OUTPUT_DIR)
    tiles_df.to_parquet(tiles_out, index=False, compression="snappy")
    orig_df.to_parquet(orig_out, index=False, compression="snappy")
    site_df.to_parquet(site_out, index=False, compression="snappy")
    logger.info("Done. %d tiles, %d (site, resolution) groups.", len(tiles_df), len(site_df))
    return 0


def run_update(workers: int, prune: bool, full: bool) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles_out = OUTPUT_DIR / "tiles.parquet"
    orig_out = OUTPUT_DIR / "tiles_orig_classes.parquet"
    site_out = OUTPUT_DIR / "site_resolution.parquet"

    logger.info("Discovering mask tiles under %s ...", LOCAL_ROOT)
    t0 = time.time()
    descriptors = discover_tiles(LOCAL_ROOT)
    logger.info("Discovered %d mask tiles on disk in %.1fs", len(descriptors), time.time() - t0)

    have_existing = tiles_out.exists() and not full
    existing_tiles = pd.read_parquet(tiles_out) if have_existing else None
    existing_orig = (
        pd.read_parquet(orig_out) if (have_existing and orig_out.exists()) else None
    )

    if have_existing:
        known = set(existing_tiles["mask_path"].astype(str))
        new_desc = [d for d in descriptors if d["mask_path"] not in known]
        logger.info("Existing DB: %d rows. New tiles to scan: %d.",
                    len(existing_tiles), len(new_desc))

        # Reconcile image linkage for rows already in the DB: the original
        # scanner mis-resolved prefix-pair image names, so refresh
        # image_exists/image_path from the corrected discovery. Cheap: a dict
        # lookup keyed by mask_path, no NPZ reads.
        desc_by_mask = {d["mask_path"]: d for d in descriptors}
        def _img_path(mp):
            d = desc_by_mask.get(mp)
            return (d["image_path"] if d and d["image_exists"] else None)
        def _img_exists(mp):
            d = desc_by_mask.get(mp)
            return bool(d["image_exists"]) if d else False
        new_exists = existing_tiles["mask_path"].map(_img_exists)
        changed = int((new_exists != existing_tiles["image_exists"]).sum())
        if changed:
            existing_tiles = existing_tiles.copy()
            existing_tiles["image_path"] = existing_tiles["mask_path"].map(_img_path)
            existing_tiles["image_exists"] = new_exists
            logger.info("Reconciled image linkage on %d existing rows "
                        "(corrected mask_/image_ prefix mismatch).", changed)
    else:
        new_desc = descriptors
        logger.info("Full build: scanning all %d tiles.", len(new_desc))

    new_tile_rows, new_orig_rows = _process_many(new_desc, workers, "new masks")
    logger.info("Computed %d new tile rows (%d failed/skipped).",
                len(new_tile_rows), len(new_desc) - len(new_tile_rows))

    # Assemble tiles_df
    if have_existing:
        if new_tile_rows:
            new_df = pd.DataFrame(new_tile_rows)
            new_df["scanned_at"] = pd.Timestamp.now(tz="UTC")
            new_df = _coerce_tiles_dtypes(new_df)
            tiles_df = pd.concat([existing_tiles, new_df], ignore_index=True)
        else:
            tiles_df = existing_tiles.copy()
    else:
        tiles_df = pd.DataFrame(new_tile_rows)
        tiles_df["scanned_at"] = pd.Timestamp.now(tz="UTC")
        tiles_df = _coerce_tiles_dtypes(tiles_df)

    # Assemble orig_df
    if have_existing and existing_orig is not None:
        orig_df = (
            pd.concat([existing_orig, pd.DataFrame(new_orig_rows)], ignore_index=True)
            if new_orig_rows else existing_orig.copy()
        )
    else:
        orig_df = pd.DataFrame(new_orig_rows)
    if not orig_df.empty:
        orig_df["original_class_id"] = orig_df["original_class_id"].astype("int32")

    # Optional prune of rows whose mask vanished from disk
    if prune:
        on_disk = {d["mask_path"] for d in descriptors}
        before = len(tiles_df)
        keep = tiles_df["mask_path"].isin(on_disk)
        dropped_ids = set(tiles_df.loc[~keep, "tile_id"])
        tiles_df = tiles_df[keep].reset_index(drop=True)
        if not orig_df.empty:
            orig_df = orig_df[~orig_df["tile_id"].isin(dropped_ids)].reset_index(drop=True)
        logger.info("Prune: dropped %d rows whose mask no longer exists.", before - len(tiles_df))

    if tiles_df.empty:
        logger.error("No tile rows — aborting without writing.")
        return 1

    logger.info("Rebuilding per-(site, resolution) rollup ...")
    site_df = build_site_resolution(tiles_df, orig_df)

    logger.info("Writing parquet files to %s ...", OUTPUT_DIR)
    tiles_df.to_parquet(tiles_out, index=False, compression="snappy")
    orig_df.to_parquet(orig_out, index=False, compression="snappy")
    site_df.to_parquet(site_out, index=False, compression="snappy")

    logger.info("Done. %d total tiles (%d new), %d (site, resolution) groups.",
                len(tiles_df), len(new_tile_rows), len(site_df))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8, help="Threads for NPZ reads.")
    p.add_argument("--validate", type=int, metavar="N", default=None,
                   help="Recompute N existing rows, assert they match, write nothing.")
    p.add_argument("--prune", action="store_true",
                   help="Drop rows whose mask NPZ no longer exists on disk.")
    p.add_argument("--dedup", action="store_true",
                   help="Drop redundant duplicate-mask rows that share one RGB image.")
    p.add_argument("--no-verify-dedup", action="store_true",
                   help="With --dedup, skip the byte-identical mask check before dropping.")
    p.add_argument("--full", action="store_true",
                   help="Ignore existing DB and rebuild from scratch.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    )

    if args.validate is not None:
        return run_validate(args.validate, args.workers)
    if args.dedup:
        return run_dedup(args.workers, verify=not args.no_verify_dedup)
    return run_update(args.workers, args.prune, args.full)


if __name__ == "__main__":
    raise SystemExit(main())
