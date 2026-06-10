# Résumé session — tuilage des blobs & dégradation DINOv3

_Date : 2026-06-08_

## Objectif
Les tuiles d'entraînement font 384×384 px, jugé trop petit pour DINOv3. But : savoir
si on peut **reconstituer des blocs plus gros** (768/1024) et **quantifier le gain**
de qualité des features selon la taille de découpage.

## Données
- `/home/clement/local_copy_train_data/<site>/.../v2_tytonai_rg/{train,trainannot,val,valannot}/*.npz`
- Chaque npz = tuile 384×384, clés `RED/GREEN/BLUE` (float32 0–255), `SRID`, `GEO_TRANSFORM`, `VERSION`.
- Site étudié : **29M_2451_GG_manned** (29 blobs, 318 tuiles).

## Acquis 1 — reconstruction des blobs (rien que sur les noms de fichier)
- Nom = `image_<uuid>_<X>_<Y>.npz` → **X (1er champ) = colonne (horizontal)**, **Y (2e) = ligne (vertical)**.
  Vérifié via les tuiles de bord tronquées. Placement : `canvas[Y:Y+h, X:X+w]`.
- **1 uuid = 1 blob = 1 image source** ; grille dense rectangulaire, pas de trou, pas 384.
- ⚠️ La numérotation est **locale à chaque blob** (tous commencent à `0_0`) → ne permet PAS
  de positionner les blobs entre eux.
- ⚠️ `GEO_TRANSFORM` est un **placeholder constant** (même origin 486168.53/6826819.17, px=0.15
  pour les 29 blobs) → inutilisable pour recoller les blobs. On n'en a pas besoin.
- 29/29 blobs reconstruits proprement. Tailles de 447×391 à **2055×1596** px ; 9 blobs ≥1024 dans les 2 dims.

## Acquis 2 — dégradation des features DINOv3 selon le découpage
Zone test = crop 1024×1024 ; mêmes 64×64 patch-tokens extraits de 3 façons :
16×256, 4×512, 1×1024 (référence contexte plein). Modèle = ViT-L/16 sat493m.

| Découpage | cosinus vs REF (f7400fc2) | % patchs changeant de classe kmeans |
|-----------|---------------------------|-------------------------------------|
| 16 × 256² | 0.72 | 17.6 % |
| 4 × 512²  | 0.90 | 9.7 % |
| 1 × 1024² | 1.00 (réf) | — |
# TROP bien pour mesurer la stabilite REFAIRE ca porur 768 384 !!!
**Conclusion : 256 très pénalisant (~-25 %), 512 raisonnable (~-9 %). Viser ≥512, idéalement 768/1024.**
Le gros saut de qualité est 256→512 ; 512→1024 est plus modeste.

## Setup technique
- venv : `.venv/bin/python` (torch 2.11+cu128, **RTX 3060 12 GB**, package `dinov3` FAIR).
- Poids : `/home/clement/Desktop/projets/1_Core_tyton_AI/tytonai-python-activities/dinov3_embedding/test_data/dinov3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`
- Norm satellite : MEAN=(0.430,0.411,0.296) STD=(0.213,0.156,0.143), patch 16, embed 1024.
- matplotlib installé dans le venv (`uv pip install matplotlib`).

## Scripts produits
- `reconstitute_blobs.py` — planche des 29 blobs (overlay grille 384 + blocs 768) → `blobs_reconstitution.png`
- `arrange_blob.py <uuid>` — zoom 1 blob avec labels (x,y) → `arrange_<uuid>.png`
- `dino_tiling_compare.py <uuid> <x0> <y0>` — PCA RGB 256/512/1024 + cosinus → `dino_tiling_compare_<uuid>.png`
- `dino_tiling_kmeans.py <uuid> <x0> <y0> [K]` — segmentation kmeans 256/512/1024 → `dino_tiling_kmeans_<uuid>.png`

## Pistes pour demain
- [ ] Générer réellement les super-tuiles **768×768** (fusion 2×2) + gérer le « reste » des grilles
      impaires (fenêtre glissante avec recouvrement / drop / pad).
- [ ] Étendre la mesure de dégradation (cosinus + % kmeans) sur **plusieurs blobs/sites** → stat agrégée.
- [ ] Décider la taille cible finale (768 vs 1024) selon budget VRAM de la prod.
- [ ] (option) Heatmap spatiale de (1−cosinus) pour localiser où la dégradation se concentre (coutures).
