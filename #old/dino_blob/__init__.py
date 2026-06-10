"""dino_blob — reconstitute training tiles into blobs and embed them with the DINOv3 activity.

Public API:
    from dino_blob import run_site, run_all_sites, list_sites
    from dino_blob import discover_blobs, plot_blob_vs_embedding, cluster_blob
"""
from __future__ import annotations

from . import config
from .core import (BlobMeta, discover_blobs, blob_dir_ids, blob_array, blob_to_geotiff,
                   ensure_activity_size, overlapping_bboxes, write_blob_metadata)
from .embedding import muted, run_activity_on_blob, mean_mosaic, clean_box_outputs
from .plots import (pca_rgb_preview, plot_blob_boxes, plot_site_boxes_grid,
                    plot_blob_vs_embedding, cluster_blob)
from .pipeline import run_site
from .multi import list_sites, run_all_sites

__version__ = "0.1.0"

__all__ = [
    "config", "BlobMeta", "discover_blobs", "blob_dir_ids", "blob_array", "blob_to_geotiff",
    "ensure_activity_size", "overlapping_bboxes", "write_blob_metadata",
    "muted", "run_activity_on_blob", "mean_mosaic", "clean_box_outputs",
    "pca_rgb_preview", "plot_blob_boxes", "plot_site_boxes_grid", "plot_blob_vs_embedding",
    "cluster_blob", "run_site", "list_sites", "run_all_sites",
]
