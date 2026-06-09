#!/usr/bin/env python
"""One-line entry point to run the blob→DINO pipeline over sites.

Examples:
    # every site on disk, split across both GPUs
    python run_sites.py --all --gpus 0,1

    # a single site (one GPU)
    python run_sites.py --site "/home/clement/local_copy_train_data/29Metals/29M_2451_GG_manned/10cm/v2_tytonai_rg"

    # build blobs + boxes + grid only (no DINO), quick
    python run_sites.py --all --no-embed

    # first 3 (largest) blobs per site, for a smoke test
    python run_sites.py --all --limit 3
"""
from __future__ import annotations

import argparse
import os

from dino_blob import run_site, run_all_sites, list_sites


def main():
    ap = argparse.ArgumentParser(description="Reconstitute blobs and embed them with DINOv3.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="process every site on disk")
    g.add_argument("--site", metavar="DIR", help="process one site dir (…/<res>/v2_tytonai_rg)")
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU ids for --all auto-split (default 0,1)")
    ap.add_argument("--shard", metavar="i/n", help="run only sites[i::n] in THIS process with a live "
                    "progress bar (pin the GPU yourself via CUDA_VISIBLE_DEVICES). e.g. --shard 0/2")
    ap.add_argument("--no-embed", dest="embed", action="store_false", help="build blobs/boxes only")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false", help="FP32 forward (byte-identical, slower)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N (largest) blobs per site")
    ap.add_argument("--png", dest="png", action="store_true",
                    help="render per-blob PCA previews during the run (default off — use observe.py)")
    ap.add_argument("--keep-intermediates", dest="clean", action="store_false",
                    help="keep per-box COGs (uses much more disk)")
    args = ap.parse_args()

    run_kw = dict(embed=args.embed, make_embed_png=args.png, clean_intermediates=args.clean,
                  limit=args.limit, bf16=args.bf16)
    if args.site:
        run_site(args.site, **run_kw)
    elif args.shard:                                  # one shard, live bar, this process's GPU
        i, n = (int(x) for x in args.shard.split("/"))
        sites = list_sites()[i::n]
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"shard {i}/{n}: {len(sites)} sites on GPU {gpu}", flush=True)
        for k, s in enumerate(sites, 1):
            r = run_site(s, show_bar=True, **run_kw)
            print(f"[shard {i}/{n}] {k}/{len(sites)} done: {r['site']}  "
                  f"{r['n_embedded']}/{r.get('n_blobs', 0)} embedded", flush=True)
    else:                                             # --all: auto-split across GPUs (quiet per-site)
        gpus = tuple(int(x) for x in args.gpus.split(","))
        sites = list_sites()
        print(f"{len(sites)} sites found.")
        run_all_sites(sites, gpus=gpus, **run_kw)


if __name__ == "__main__":
    main()
