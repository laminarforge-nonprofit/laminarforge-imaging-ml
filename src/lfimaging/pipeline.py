"""Pipeline orchestrator — turn an ImageBundle into ChipMetrics."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

from .barrier import score_barrier
from .ingest import ImageBundle
from .livedead import analyze_livedead
from .nuclear import count_nuclei
from .segmentation import compute_confluence

log = logging.getLogger(__name__)


@dataclass
class ChipMetrics:
    """Per-chip-per-timepoint quantitative output of the pipeline."""

    chip_id: str
    timestamp: datetime
    confluence_pct: float
    cell_count: int
    viability_pct: float
    live_count: int
    dead_count: int
    nuclear_density: float
    nuclear_total: int
    barrier_pct: float
    barrier_hole_count: int
    campaign: str | None = None

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


def _pick_brightfield(bundle: ImageBundle):
    """Prefer brightfield for confluence; fall back to phase or whatever we have."""
    for ch in ("brightfield", "phase"):
        img = bundle.get(ch)
        if img is not None:
            return img
    # Last resort: any channel.
    for _, img in bundle.channels.items():
        return img
    raise ValueError(f"ImageBundle for {bundle.chip_id} has no channels")


def process_chip(bundle: ImageBundle, campaign: str | None = None) -> ChipMetrics:
    """Run every stage against an ImageBundle and assemble a ChipMetrics record.

    Missing channels are handled gracefully — the corresponding metric is set to 0/NaN
    and logged at WARNING level. The goal is a single contract (ChipMetrics) that the
    dashboard can rely on regardless of which channels the gantry captured.
    """
    log.info("process_chip chip=%s ts=%s", bundle.chip_id, bundle.timestamp)

    # Confluence + cell count from brightfield/phase.
    bf = _pick_brightfield(bundle)
    confluence = compute_confluence(bf)

    # LIVE/DEAD from Calcein + PI.
    calcein = bundle.get("calcein")
    pi = bundle.get("pi")
    if calcein is not None and pi is not None:
        livedead = analyze_livedead(calcein, pi)
    else:
        log.warning("chip=%s missing calcein/pi channels, skipping LIVE/DEAD", bundle.chip_id)
        from .livedead import LiveDeadResult

        livedead = LiveDeadResult(live_count=0, dead_count=0, viability_pct=0.0)

    # Nuclear count from Hoechst.
    hoechst = bundle.get("hoechst")
    if hoechst is not None:
        nuclear = count_nuclei(hoechst)
    else:
        log.warning("chip=%s missing hoechst channel, skipping nuclear count", bundle.chip_id)
        from .nuclear import NuclearCount

        nuclear = NuclearCount(total_cells=0, density_per_mm2=0.0)

    # Barrier integrity from phase/brightfield.
    barrier = score_barrier(bf)

    return ChipMetrics(
        chip_id=bundle.chip_id,
        timestamp=bundle.timestamp,
        confluence_pct=float(confluence.percent),
        cell_count=int(confluence.cell_count),
        viability_pct=float(livedead.viability_pct),
        live_count=int(livedead.live_count),
        dead_count=int(livedead.dead_count),
        nuclear_density=float(nuclear.density_per_mm2),
        nuclear_total=int(nuclear.total_cells),
        barrier_pct=float(barrier.pct_coverage),
        barrier_hole_count=int(barrier.hole_count),
        campaign=campaign,
    )
