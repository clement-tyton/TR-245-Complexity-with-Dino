# Idea — KNN label propagation of annotations in DINO feature space

Status: idea / not implemented. Captured for later.

## Goal
We have per-pixel annotator labels for *some* areas. Propagate them to the **unannotated**
patches of every site by k-nearest-neighbours in DINO embedding space (FAISS for speed):
for each unlabeled patch, find the k nearest *annotated* patches and assign the
(distance-weighted) majority class.

## Why it's sound
KNN on frozen DINO features is exactly how DINO is evaluated in the papers — the features
are locally separable by semantic content, so no training is needed. This is essentially
label propagation / few-shot segmentation by feature matching.

## How it maps to our data
- Annotations live next to the tiles: `…/v2_tytonai_rg/{trainannot,valannot}` — per-pixel
  class masks at native res (~0.097 m).
- DINO labels are per-**patch** at `embed_gsd` (~0.78 m); each patch covers ~8×8 native px.
- **Bridge (aggregation):** for each patch, majority-vote the annotation pixels under its
  footprint -> a patch-level label + a *purity* score (how unanimous). Patches with annotation
  = the **labeled set**; patches without = the **query set**.

## Pipeline
1. **Labeled index:** gather `(patch_embedding, majority_label, purity)` for all annotated
   patches across sites (we already store the patch grids as `.npz` / parquet).
2. **FAISS:** L2-normalize vectors, `IndexFlatIP` (cosine) for exact, or `IVF`/`HNSW` at scale.
   Use the **256-d PCA-reduced** vectors from `GPUPCA` / `transform_all_tiles` (fast, ~no loss).
3. **Classify:** per query patch, k neighbours -> **distance-weighted majority vote** ->
   class + confidence (vote margin / mean similarity).
4. **Output:** a georeferenced patch-level class map + a **confidence map** (QGIS-ready,
   alongside `dino_pca_webmap.tif`), optionally upsampled to pixel res.

## Design choices
- Embedding: 256-d PCA, normalized, cosine. (1024-d works; PCA is the scale win.)
- Distance-weighted KNN > plain majority.
- **Abstention/confidence** is the big value-add: if neighbours disagree or are far, abstain
  and surface it -> tells annotators where to look (confidence map for free).
- Nearest-class-mean (one centroid/class) as a cheap baseline + handles rare classes.

## Gotchas
- **Resolution/boundaries:** predictions at ~0.78 m -> blurry class boundaries vs per-pixel GT.
  Good for land-structure/coverage, weaker for thin features.
- **Class imbalance:** majority vote drifts to frequent classes; distance-weight + per-class
  normalization; drop low-purity patches.
- **Cross-site vs per-site:** global index = more labels but domain shift; per-site = safe
  baseline. Test both.
- **Ignore/background:** keep annotator "ignore" pixels out of the labeled set.

## Fit
Natural consumer of the parquet/cls store: `cls` for whole-cell context, patch grids for fine
labels. Only real engineering = patch↔pixel aggregation + the FAISS index.

## First experiment (one site)
Build the labeled patch set from `trainannot`, fit FAISS on the 256-d PCA vectors, KNN-label
the unannotated patches, render a class map + confidence map georeferenced next to the PCA
webmap — eyeball whether the neighbours make sense before scaling.
