"""One-site orchestrator: training tiles -> grid -> DINO embeddings -> manifest + PCA webmap.

Usage:
    python src/pipeline.py "/home/clement/local_copy_train_data/BHP Creeks 2022/.../v2_tytonai_rg"
or:
    from pipeline import run_site; run_site(site_dir)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make src/ modules importable

import config
import transforms
import dino
import store as sink   # src/store.py (persistence) — named 'store' to avoid the stdlib 'io' clash
import pca


def run_site(site_dir=None, *, site_key=None, res=None, rgb_path=None, out_root=config.EMB_ROOT,
             dino_model=config.DINO_MODEL, high_res=config.HIGH_RES,
             tile_patches=config.TILE_PATCHES, min_data_cov=config.MIN_DATA_COV,
             upsample=None, make_plots=False, make_webmap=True, show_bar=True, resume=True) -> dict:
    """Build + embed every grid cell of one site. Returns a summary dict.

    Two ways to point at a site:
      - site_key='<Project>/<Site>' (+ optional res): tiles + webmap resolved from the shared
        /mnt store (config.resolve_tiles / resolve_rgb) — this is the MULTI-SITE entry point.
      - site_dir=<path>: an explicit tiles dir (e.g. a local copy); key/res derived from it.
    resume=True: skip (return {"skipped": True}) if the site's cells.parquet already exists.
    Outputs under out_root/<site_id>/ + the Hive-partitioned out_root/cells/site_id=<id>/.
    """
    if site_dir is None:
        if site_key is None:
            raise ValueError("pass site_key='<Project>/<Site>' (multi-site) or site_dir=<path>")
        res = res or config.default_res(site_key)
        site_id = config.site_id_from_key(site_key, res)
    else:
        site_key = config.site_key_from_dir(site_dir)
        site_id = config.site_id_from_dir(site_dir)

    part_dir = os.path.join(out_root, "cells", f"site_id={site_id}")
    if resume and os.path.exists(os.path.join(part_dir, "cells.parquet")):
        print(f"[{site_id}] already embedded -> skip")
        return {"site_id": site_id, "skipped": True}

    if site_dir is None:
        site_dir = config.resolve_tiles(site_key, res)        # /mnt ObjectData tiles
    if rgb_path is None:
        rgb_path = config.resolve_rgb(site_key, res)["rgb_path"]

    tiles = transforms.read_tile_bboxes(site_dir)
    tiles_clip, extent = transforms.crop_tiles_to_webmap(tiles, rgb_path)
    area = transforms.study_area(tiles_clip)
    grid, ginfo = transforms.build_tile_grid(area, tiles_clip, rgb_path,
                                             tile_patches=tile_patches, high_res=high_res,
                                             min_data_cov=min_data_cov)
    print(f"[{site_id}] {len(tiles)} tiles -> {ginfo['n_cells']} cells {ginfo}")

    patch_dir, part_dir = sink.site_emb_dirs(site_id, out_root)

    # per-site 2x2 QA control image (tiles -> crop -> study area -> grid), always written
    import plots
    qa_png = plots.plot_qa_grid(tiles, tiles_clip, extent, area, grid, info=ginfo,
                                out_png=os.path.join(out_root, site_id, "qa_steps.png"),
                                title=f"{site_id} — {ginfo['n_cells']} cells")
    print(f"  QA -> {qa_png}")

    act, model, device, grid_w = dino.setup_activity(rgb_path, grid, dino_model=dino_model,
                                                     high_res=high_res)
    npz_paths, cls_vecs = sink.embed_grid(act, model, device, grid_w, rgb_path, patch_dir,
                                          upsample=upsample, show_bar=show_bar, desc=site_id)
    sink.write_manifest(grid_w, npz_paths, cls_vecs, site_id, part_dir, emb_root=out_root)

    # self-describing meta.json (resolved model + effective upscale read back from the activity)
    _rm = getattr(getattr(act, "input_model", None), "dino_model", None)
    resolved_model = _rm.value if _rm is not None else dino_model
    eff_upsample = upsample if upsample is not None else getattr(act, "patch_upsample_factor", None)
    sink.write_site_meta(site_id, part_dir, ginfo, npz_paths, dino_model=resolved_model,
                         high_res=high_res, upsample=eff_upsample, emb_root=out_root)

    webmap_tif = None
    if make_webmap and npz_paths:
        webmap_tif = os.path.join(out_root, site_id, "dino_pca_webmap.tif")
        pca.build_pca_webmap(npz_paths, list(grid_w.geometry), grid_w.crs, webmap_tif,
                             webmap_path=rgb_path)

    if make_plots:
        import plots
        plots.plot_tiles(tiles)
        plots.plot_webmap_crop(tiles, tiles_clip, extent)
        plots.plot_study_area(tiles_clip, area)
        plots.plot_grid(tiles_clip, area, grid, info=ginfo)
        if npz_paths:
            plots.plot_site_pca(npz_paths, list(grid_w.geometry),
                                os.path.join(out_root, site_id, "site_patch_pca.png"),
                                webmap_path=rgb_path)

    print(f"[{site_id}] done: {len(npz_paths)} cells -> {patch_dir}  |  {part_dir}/cells.parquet")
    return {"site_id": site_id, "n_tiles": len(tiles), "n_cells": len(grid_w), "ginfo": ginfo,
            "patch_dir": patch_dir, "part_dir": part_dir, "webmap_tif": webmap_tif,
            "qa_png": qa_png, "npz_paths": npz_paths}


def all_site_keys():
    """Every site in the catalog (config/sites_to_resolutions.json), file order."""
    return list(config._load_json("sites_to_resolutions.json"))


def _write_run_manifest(out_root, keys, res, run_kw):
    """Dump the exact settings of this batch to <out_root>/runs/run_<ts>_gpu<dev>_<pid>.json so a
    months-old embedding set stays reproducible (model, high_res, weights, git commit, sites).
    One file per process => shards/GPUs never clobber each other. Best-effort; never fails the run."""
    import datetime
    import json
    import socket
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
                                capture_output=True, text=True).stdout.strip() or None
    except Exception:
        commit = None
    wdir = os.environ.get("DINO_WEIGHTS_FOLDER", "")
    weights = ({f: os.path.getsize(os.path.join(wdir, f)) for f in os.listdir(wdir)
                if f.endswith(".pth")} if wdir and os.path.isdir(wdir) else {})
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit, "host": socket.gethostname(), "pid": os.getpid(),
        "cuda_visible_devices": dev, "emb_root": out_root, "res": res,
        "dino_model": run_kw.get("dino_model", config.DINO_MODEL),
        "high_res": run_kw.get("high_res", config.HIGH_RES),
        "dino_dtype": config.DINO_DTYPE,
        "upsample": run_kw.get("upsample", None),
        "vram_gb_for_7b": config.VRAM_GB_FOR_7B,
        "tile_patches": config.TILE_PATCHES, "min_data_cov": config.MIN_DATA_COV,
        "weights_folder": wdir, "weights_files": weights,
        "n_sites": len(keys), "sites": keys,
    }
    runs_dir = os.path.join(out_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"run_{ts}_gpu{dev or 'x'}_{os.getpid()}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"run manifest -> {path}")
    return path


def run_sites(site_keys=None, *, res=None, out_root=config.EMB_ROOT, shard=None, limit=None,
              **run_kw) -> dict:
    """Embed many sites. site_keys=None -> ALL in the catalog. Skips (and reports) sites with
    no resolved webmap / no tiles. shard=(i, n) takes site_keys[i::n] (run n terminals, one GPU
    each via CUDA_VISIBLE_DEVICES). resume defaults on, so re-running continues where it stopped.
    """
    keys = list(site_keys) if site_keys is not None else all_site_keys()
    if shard is not None:
        i, n = shard
        keys = keys[i::n]
    if limit:
        keys = keys[:limit]
    _write_run_manifest(out_root, keys, res, run_kw)      # reproducibility: settings + commit + sites
    done, fails = [], []
    for k, key in enumerate(keys, 1):
        print(f"\n===== [{k}/{len(keys)}] {key} =====", flush=True)
        try:
            r = run_site(site_key=key, res=res, out_root=out_root, **run_kw)
            done.append(r)
        except (KeyError, FileNotFoundError) as e:           # no webmap / no tiles -> expected skip
            print(f"  SKIP: {e}"); fails.append((key, "skip", str(e)))
        except Exception as e:                                # anything else -> record + keep going
            print(f"  FAIL: {e}"); fails.append((key, "fail", str(e)))
    embedded = sum(not r.get("skipped") for r in done)
    print(f"\n===== {len(done)}/{len(keys)} ok ({embedded} embedded, {len(done)-embedded} already-done) "
          f"| {len(fails)} skipped/failed =====")
    for key, kind, msg in fails:
        print(f"  {kind.upper():4s} {key}: {msg[:80]}")
    return {"done": done, "fails": fails, "keys": keys}


def _gpu_worker(gpu, idx, n, site_keys, run_kw):
    """Child process pinned to ONE GPU, running sites[idx::n]. Pins CUDA_VISIBLE_DEVICES
    BEFORE any torch import (our modules import torch lazily, so this holds)."""
    import os as _os
    import sys as _sys
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import pipeline as _pl
    print(f"[gpu {gpu}] shard {idx}/{n}", flush=True)
    _pl.run_sites(site_keys, shard=(idx, n), **run_kw)


def run_all_gpus(gpus=(0, 1), site_keys=None, **run_kw):
    """One command -> one process per GPU, each embedding sites[i::len(gpus)] (modulo split).
    Uses spawn so each child gets a fresh CUDA context pinned to its GPU. Output interleaves;
    for clean per-GPU progress bars run two terminals with --shard instead (see below)."""
    import multiprocessing as mp
    keys = list(site_keys) if site_keys is not None else all_site_keys()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_gpu_worker, args=(g, i, len(gpus), keys, run_kw))
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Embed site grid(s) with DINOv3 + PCA webmap.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site-key", help="'<Project>/<Site>' — resolves tiles + webmap from /mnt")
    g.add_argument("--site-dir", help="explicit tiles dir …/<res>/v2_tytonai_rg (e.g. a local copy)")
    g.add_argument("--all-sites", action="store_true", help="every site in the catalog")
    ap.add_argument("--res", default=None, help="resolution (e.g. 10cm); default = the site's native")
    ap.add_argument("--rgb-path", default=None, help="override the webmap raster (else resolved from config)")
    ap.add_argument("--out-root", default=config.EMB_ROOT)
    ap.add_argument("--upsample", type=int, default=None, help="forward upscale (2=1024, 4=2048)")
    ap.add_argument("--shard", default=None, help="i/n — run sites[i::n] (one terminal/GPU)")
    ap.add_argument("--gpus", default=None, help="comma ids, e.g. 0,1 — spawn one process per GPU")
    ap.add_argument("--limit", type=int, default=None, help="only the first N sites (with --all-sites)")
    ap.add_argument("--no-webmap", dest="make_webmap", action="store_false")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--plots", dest="make_plots", action="store_true")
    a = ap.parse_args()
    common = dict(res=a.res, out_root=a.out_root, upsample=a.upsample,
                  make_webmap=a.make_webmap, make_plots=a.make_plots, resume=a.resume)
    if a.all_sites:
        if a.gpus:                                            # one command -> one process per GPU
            run_all_gpus(gpus=tuple(int(x) for x in a.gpus.split(",")), **common)
        else:
            shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
            run_sites(shard=shard, limit=a.limit, **common)
    else:
        run_site(site_dir=a.site_dir, site_key=a.site_key, rgb_path=a.rgb_path, **common)
