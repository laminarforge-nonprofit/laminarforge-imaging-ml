"""Cell segmentation via Cellpose 3.x (cyto3).

Cellpose is heavy (torch + pretrained weights) so this module imports lazily. When
the library isn't available — e.g., CI installs without the ``ml`` extra — we fall
back to a classical Otsu + watershed segmenter so the pipeline still produces numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_CELLPOSE_MODEL = None


@dataclass
class Confluence:
    """Per-chip confluence result."""

    percent: float
    cell_count: int
    mask: np.ndarray  # int32, 0 = background, >0 = cell ID


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        # Luminance; keep it simple, avoids pulling in cv2 for this.
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(image.dtype)
    raise ValueError(f"Unsupported image shape for segmentation: {image.shape}")


def _load_cellpose(model_type: str = "cyto3"):  # noqa: ANN202 - external lib type
    """Lazy-load Cellpose so import cost is paid once, not at package import."""
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL
    try:
        from cellpose import models  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        _CELLPOSE_MODEL = models.Cellpose(gpu=False, model_type=model_type)
    except Exception:
        # Network failure on first weight download, etc.
        return None
    return _CELLPOSE_MODEL


def _classical_segment(image: np.ndarray) -> np.ndarray:
    """Otsu threshold + watershed fallback that never touches the network."""
    from scipy import ndimage as ndi
    from skimage import filters, morphology, segmentation

    gray = _to_grayscale(image).astype(np.float32)
    if gray.max() > 0:
        gray = gray / gray.max()
    thr = filters.threshold_otsu(gray) if gray.size else 0.5
    fg = gray > thr
    fg = morphology.remove_small_objects(fg, min_size=32)
    fg = morphology.closing(fg, morphology.disk(2))

    distance = ndi.distance_transform_edt(fg)
    coords = ndi.maximum_filter(distance, size=11) == distance
    coords = coords & fg
    markers, _ = ndi.label(coords)
    labels = segmentation.watershed(-distance, markers, mask=fg)
    return labels.astype(np.int32)


def segment_cells(image: np.ndarray, model: str = "cyto3") -> np.ndarray:
    """Return an int32 label mask for ``image``. 0 = background, >0 = individual cells.

    Uses Cellpose ``cyto3`` when available, classical watershed otherwise.
    """
    cp = _load_cellpose(model)
    if cp is not None:
        channels = [0, 0]  # grayscale
        masks, _flows, _styles, _diams = cp.eval(image, diameter=None, channels=channels)
        return np.asarray(masks, dtype=np.int32)
    return _classical_segment(image)


def compute_confluence(image: np.ndarray, model: str = "cyto3") -> Confluence:
    """Segment and compute per-chip confluence + cell count."""
    mask = segment_cells(image, model=model)
    total_pixels = int(mask.size)
    foreground = int(np.count_nonzero(mask))
    percent = (foreground / total_pixels) * 100.0 if total_pixels else 0.0
    # Unique positive labels.
    unique = np.unique(mask)
    cell_count = int(max(0, len(unique) - (1 if 0 in unique else 0)))
    return Confluence(percent=percent, cell_count=cell_count, mask=mask)
