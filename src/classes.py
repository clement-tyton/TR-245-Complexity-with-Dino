"""Class taxonomy for TytonAI segmentation masks.

Sources of truth: short + full class names, class-hierarchy mappings to
coarser schemes (7 / 6 / 3 / 2 class), and helper utilities for plotting
class masks with proper legends.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# --- Nodata values ---------------------------------------------------------

NO_DATA_VALUES = [0, 15, 127, -128, 255, 65535]


# --- Class name dictionaries ----------------------------------------------

CLASS_NAMES_SHORT: dict[int, str] = {
    0: "No class",
    1: "Non erosion", 141: "Rills", 142: "Gullies",
    2: "Ground", 3: "HumGras", 4: "Shrub", 5: "Tree", 6: "Herb", 7: "Grass",
    9: "Debris", 40: "Sedge", 70: "AnnHer",
    200: "Abiotic", 201: "Water",
    301: "TusGra", 302: "Herb2", 303: "CenSpp",
    401: "SkeShr", 402: "AerJav", 450: "MacSpp",
    501: "CalPro",
    601: "Melaleu", 603: "EucSpp", 607: "BanSpp",
    1791: "PasFoe", 4900: "XanPre",
    5007: "Mulga", 5103: "NelPal", 5114: "ParAcu", 5203: "VacFar",
    6008: "EucCam", 6012: "EucVic", 6015: "MelArg", 6016: "PhoDac",
    9003: "DeadTr", 9004: "DeadSh",
    10001: "SteveSpi",
    10003: "BBT1", 10004: "BBT2", 10005: "BBT3", 10006: "BBT4",
    10007: "NGLowPalm", 10008: "NGTallPalm", 10009: "NGGrass",
    10010: "NGBroad", 10011: "NGShrub", 10012: "NGTree", 10013: "NGWisTree",
    10050: "HerbRumVes", 10092: "ShrubTecticorna",
}

CLASS_NAMES_FULL: dict[int, str] = {
    0: "No class",
    1: "Non erosion", 141: "Rills", 142: "Gullies",
    2: "Ground", 3: "Hummock Grass", 4: "Shrub", 5: "Tree", 6: "Herb", 7: "Grass",
    9: "Generic Debris", 40: "Sedge", 70: "Annual herbs and grasses",
    200: "Abiotic", 201: "Water",
    301: "Tussock Grass", 302: "Herb 2", 303: "Cenchrus spp.",
    401: "Skeletal shrub", 402: "Aerva javanica", 450: "Macrozamia spp.",
    501: "Calotropis procera",
    601: "Melaleuca spp.", 603: "Eucalyptus spp.", 607: "Banksia spp.",
    1791: "Passiflora foetida", 4900: "Xanthorrhoea preissii",
    5007: "Mulga", 5103: "Neltuma pallida", 5114: "Parkinsonia aculeata",
    5203: "Vachellia farnesiana",
    6008: "Eucalyptus camaldulensis", 6012: "Eucalyptus victrix",
    6015: "Melaleuca argentea", 6016: "Phoenix dactylifera",
    9003: "Dead Triodia", 9004: "Dead Shrub",
    10001: "Steve the spikey plant",
    10003: "BBT1", 10004: "BBT2", 10005: "BBT3", 10006: "BBT4",
    10007: "New Guinea low Palm", 10008: "New Guinea Tall Palm",
    10009: "New Guinea Grass", 10010: "New Guinea Broadleaf",
    10011: "New Guinea shrub", 10012: "New Guinea Tree",
    10013: "New Guinea Wispy Tree",
    10014: "ALC Other Veg", 10016: "Marrie",
    10050: "Rumex Vesicarius", 10092: "Tecticorna spp",
}

ALL_CLASS_VALUES: list[int] = list(CLASS_NAMES_FULL.keys())


# --- Hierarchical mappings (fine class -> coarser class) ------------------
# Keys are string ints so they can be applied to JSON-serialised masks.

LIFEFORM_SEVEN_CLASS_PARENT_MAP: dict[int, int] = {
    7: 301, 9: 2, 200: 2, 201: 2,
    5007: 4, 401: 4, 402: 4, 501: 4, 5203: 4, 4900: 4, 450: 4,
    601: 5, 603: 5, 607: 5, 5103: 5, 5114: 5, 6016: 5,
    6008: 603, 6012: 603,
    6015: 601,
    70: 6, 302: 6, 1791: 6,
    9003: 9, 9004: 9,
    303: 301,
    10001: 4, 10003: 5, 10004: 5, 10005: 5, 10006: 5,
    10007: 4, 10008: 5, 10009: 301, 10010: 5,
    10011: 4, 10012: 5, 10013: 5,
    10050: 6, 10092: 4,
}

SEVEN_CLASS_MAPPING: dict[str, str] = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "40": "40", "301": "301",
    "7": "301", "9": "2", "201": "2", "200": "2", "9003": "3",
    "5007": "4", "401": "4", "9004": "4", "402": "4", "501": "4", "5203": "4",
    "4900": "4", "450": "4",
    "601": "5", "603": "5", "607": "5", "5103": "5", "5114": "5",
    "6008": "5", "6012": "5", "6015": "5", "6016": "5",
    "70": "6", "302": "6", "1791": "6",
    "303": "301",
    "10001": "4", "10003": "5", "10004": "5", "10005": "5", "10006": "5",
    "10007": "4", "10008": "5", "10009": "301", "10010": "5",
    "10011": "4", "10012": "5", "10013": "5",
    "10050": "6", "10092": "4",
}

SIX_CLASS_MAPPING: dict[str, str] = {
    "2": "2", "3": "7", "4": "4", "5": "5", "6": "6", "7": "7",
    "40": "40", "301": "7",
    "9": "2", "201": "2", "200": "2", "9003": "7",
    "5007": "4", "401": "4", "9004": "4", "402": "4", "501": "4", "5203": "4",
    "4900": "4", "450": "4",
    "601": "5", "603": "5", "607": "5", "5103": "5", "5114": "5",
    "6008": "5", "6012": "5", "6015": "5", "6016": "5",
    "70": "6", "302": "6", "1791": "6",
    "303": "7",
    "10001": "4", "10003": "5", "10004": "5", "10005": "5", "10006": "5",
    "10007": "4", "10008": "5", "10009": "7", "10010": "5",
    "10011": "4", "10012": "5", "10013": "5",
    "10050": "6", "10092": "4",
}

THREE_CLASS_MAPPING: dict[str, str] = {
    "2": "2", "3": "3", "4": "4",
    "7": "3", "9": "2", "201": "2", "200": "2", "9003": "3", "6": "3",
    "301": "3", "40": "3",
    "5007": "4", "401": "4", "9004": "4", "402": "4", "501": "4", "5203": "4",
    "4900": "4", "450": "4",
    "601": "4", "603": "4", "607": "4", "5103": "4", "5114": "4",
    "6008": "4", "6012": "4", "6015": "4", "6016": "4", "5": "4",
    "70": "3", "302": "3", "1791": "3",
    "303": "3",
    "10001": "4", "10003": "4", "10004": "4", "10005": "4", "10006": "4",
    "10007": "4", "10008": "4", "10009": "3", "10010": "4",
    "10011": "4", "10012": "4", "10013": "4",
    "10050": "3", "10092": "4",
}

TWO_CLASS_MAPPING: dict[str, str] = {
    "2": "2", "3": "3",
    "7": "3", "9": "2", "201": "2", "200": "2", "9003": "3", "6": "3",
    "301": "3", "40": "3", "4": "3",
    "5007": "3", "401": "3", "9004": "3", "402": "3", "501": "3", "5203": "3",
    "4900": "3", "450": "3",
    "601": "3", "603": "3", "607": "3", "5103": "3", "5114": "3",
    "6008": "3", "6012": "3", "6015": "3", "6016": "3", "5": "3",
    "70": "3", "302": "3", "1791": "3",
    "303": "3",
    "10001": "3", "10003": "3", "10004": "3", "10005": "3", "10006": "3",
    "10007": "3", "10008": "3", "10009": "3", "10010": "3",
    "10011": "3", "10012": "3", "10013": "3",
    "10050": "3", "10092": "3",
}

TWO_CLASS_EROSION_MAPPING: dict[str, str] = {
    "1": "1", "14": "14", "141": "14", "142": "14",
}

IDENTITY_MAPPING: dict[str, str] = {str(k): str(k) for k in ALL_CLASS_VALUES}


# --- Plotting helpers -----------------------------------------------------

def remap_mask(mask: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    """Remap fine-grain class IDs in a mask using one of the *_CLASS_MAPPING dicts.

    Values not present in `mapping` are left unchanged (treat as nodata).
    """
    out = mask.copy()
    for src, dst in mapping.items():
        out[mask == int(src)] = int(dst)
    return out


def class_name(value: int, full: bool = False) -> str:
    """Return the display name for a class value (or a stringified value)."""
    table = CLASS_NAMES_FULL if full else CLASS_NAMES_SHORT
    return table.get(int(value), str(value))


def plot_mask(
    ax,
    mask: np.ndarray,
    *,
    full_names: bool = False,
    cmap_name: str = "tab20",
    title: str | None = None,
):
    """Render a class mask with a colour-mapped image and a legend.

    Each unique value in `mask` gets its own colour from `cmap_name`.
    The legend uses short or full class names from this module.
    """
    values = np.unique(mask).tolist()
    base_cmap = plt.get_cmap(cmap_name, max(len(values), 1))
    colors = [base_cmap(i) for i in range(len(values))]

    value_to_index = {v: i for i, v in enumerate(values)}
    indexed = np.vectorize(value_to_index.get)(mask)

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(values) + 1) - 0.5, len(values))

    ax.imshow(indexed, cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)

    handles = [
        Patch(facecolor=colors[i], edgecolor="black",
              label=f"{v}: {class_name(v, full=full_names)}")
        for i, v in enumerate(values)
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, frameon=False,
    )
