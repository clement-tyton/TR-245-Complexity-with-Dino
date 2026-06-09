"""Run many sites, split across GPUs.

Each GPU runs in its own process, pinned via CUDA_VISIBLE_DEVICES set BEFORE torch/the activity
is imported (the activity is imported lazily inside run_site, so a fresh spawned process is clean).
Sites are sorted largest-first and dealt round-robin to balance the two GPUs.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import pandas as pd

from . import config


def list_sites(train_root=config.TRAIN_ROOT, parquet=config.STATS_PARQUET):
    """All site dirs (…/<res>/<version>) that exist on disk, largest-first by tile count."""
    df = pd.read_parquet(parquet).sort_values("n_tiles_total", ascending=False)
    dirs = []
    for _, r in df.iterrows():
        p = os.path.join(train_root, r["site"], r["resolution"], r["dataset_version"])
        if os.path.isdir(os.path.join(p, "train")):
            dirs.append(p)
    return dirs


def _gpu_worker(gpu, site_dirs, run_kw):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)   # pin BEFORE torch is imported
    from .pipeline import run_site
    for i, sd in enumerate(site_dirs, 1):
        try:
            s = run_site(sd, show_bar=False, **run_kw)
            print(f"[gpu{gpu}] {i}/{len(site_dirs)} {s['site']}: "
                  f"{s['n_embedded']}/{s.get('n_blobs', 0)} embedded, {len(s['fails'])} failed", flush=True)
        except Exception as e:  # never let one site kill the worker
            print(f"[gpu{gpu}] {i}/{len(site_dirs)} {sd} CRASHED: {e}", flush=True)


def run_all_sites(site_dirs=None, gpus=(0, 1), **run_kw):
    """Embed every site, split across `gpus`. run_kw is forwarded to run_site (embed, limit, ...)."""
    if site_dirs is None:
        site_dirs = list_sites()
    gpus = tuple(gpus)
    chunks = {g: site_dirs[i::len(gpus)] for i, g in enumerate(gpus)}   # round-robin on size-sorted list
    print(f"running {len(site_dirs)} sites across gpus {gpus} "
          f"({', '.join(f'gpu{g}:{len(chunks[g])}' for g in gpus)})", flush=True)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_gpu_worker, args=(g, chunks[g], run_kw)) for g in gpus]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"all {len(site_dirs)} sites done in {time.time() - t0:.0f}s across gpus {gpus}", flush=True)
