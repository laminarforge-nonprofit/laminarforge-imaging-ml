"""Smoke tests for the imaging pipeline on synthetic data.

We synthesise 2-3 example channels (brightfield, calcein, pi, hoechst) so tests
run without network access (Cellpose weight download is optional). When the
``cellpose`` library is installed the real model is exercised; otherwise the
classical watershed fallback runs. Either way, outputs must land in sensible ranges.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lfimaging.barrier import score_barrier
from lfimaging.ingest import ImageBundle, load_bundle_from_dir
from lfimaging.livedead import analyze_livedead
from lfimaging.nuclear import count_nuclei
from lfimaging.pipeline import process_chip
from lfimaging.segmentation import compute_confluence

HERE = Path(__file__).parent
EXAMPLE_DIR = HERE.parent / "data" / "example"


def _make_cells(shape: tuple[int, int], n: int, rng: np.random.Generator) -> np.ndarray:
    """Paint ``n`` bright gaussian blobs onto a dark background."""
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.integers(20, h - 20), rng.integers(20, w - 20)
        sigma = rng.uniform(4, 7)
        intensity = rng.uniform(0.6, 1.0)
        img += intensity * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def synthetic_bundle() -> ImageBundle:
    rng = np.random.default_rng(42)
    shape = (256, 256)
    brightfield = _make_cells(shape, 60, rng)
    calcein = _make_cells(shape, 50, rng)  # mostly live
    pi = _make_cells(shape, 10, rng)  # few dead
    hoechst = _make_cells(shape, 60, rng)
    return ImageBundle(
        chip_id="C1-S1",
        timestamp=datetime(2026, 4, 20, 12, 0, 0),
        channels={
            "brightfield": brightfield,
            "calcein": calcein,
            "pi": pi,
            "hoechst": hoechst,
        },
    )


def test_segmentation_runs(synthetic_bundle: ImageBundle) -> None:
    result = compute_confluence(synthetic_bundle.channels["brightfield"])
    assert 0.0 <= result.percent <= 100.0
    assert result.cell_count >= 0
    assert result.mask.shape == synthetic_bundle.channels["brightfield"].shape


def test_livedead_heuristic(synthetic_bundle: ImageBundle) -> None:
    r = analyze_livedead(
        synthetic_bundle.channels["calcein"], synthetic_bundle.channels["pi"]
    )
    assert r.live_count >= 0
    assert r.dead_count >= 0
    assert 0.0 <= r.viability_pct <= 100.0


def test_nuclear_count(synthetic_bundle: ImageBundle) -> None:
    n = count_nuclei(synthetic_bundle.channels["hoechst"])
    assert n.total_cells >= 0
    assert n.density_per_mm2 >= 0.0


def test_barrier_score(synthetic_bundle: ImageBundle) -> None:
    b = score_barrier(synthetic_bundle.channels["brightfield"])
    assert 0.0 <= b.pct_coverage <= 100.0
    assert b.hole_count >= 0
    assert b.mean_hole_area >= 0.0


def test_process_chip(synthetic_bundle: ImageBundle) -> None:
    metrics = process_chip(synthetic_bundle, campaign="test-2026-04")
    assert metrics.chip_id == "C1-S1"
    assert metrics.campaign == "test-2026-04"
    assert 0.0 <= metrics.confluence_pct <= 100.0
    assert metrics.cell_count >= 0
    assert 0.0 <= metrics.viability_pct <= 100.0
    assert 0.0 <= metrics.barrier_pct <= 100.0
    d = metrics.to_dict()
    assert isinstance(d["timestamp"], str)


def test_process_chip_missing_channels() -> None:
    """Pipeline must degrade gracefully when Calcein/PI/Hoechst aren't captured."""
    rng = np.random.default_rng(7)
    bf = _make_cells((128, 128), 20, rng)
    bundle = ImageBundle(
        chip_id="C2-S1",
        timestamp=datetime(2026, 4, 20, 13, 0, 0),
        channels={"brightfield": bf},
    )
    metrics = process_chip(bundle)
    assert metrics.live_count == 0
    assert metrics.dead_count == 0
    assert metrics.nuclear_total == 0


def test_load_bundle_from_dir(tmp_path: Path) -> None:
    """Filenames follow the gantry convention."""
    rng = np.random.default_rng(0)
    img = _make_cells((64, 64), 5, rng)
    for ch in ("brightfield", "calcein", "pi", "hoechst"):
        fn = tmp_path / f"C9-S3_2026-04-20T10-00-00_{ch}.tif"
        tifffile.imwrite(fn.as_posix(), img)
    bundle = load_bundle_from_dir(tmp_path)
    assert bundle.chip_id == "C9-S3"
    assert set(bundle.channels) == {"brightfield", "calcein", "pi", "hoechst"}


def test_example_images_present() -> None:
    """We ship 2-3 example images in data/example/."""
    tifs = sorted(EXAMPLE_DIR.glob("*.tif"))
    assert len(tifs) >= 2, f"Expected example images under {EXAMPLE_DIR}, found {tifs}"
