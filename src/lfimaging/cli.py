"""``lf-imaging`` CLI entry point.

Subcommands:
    process         — process one chip's image bundle and publish metrics
    watch           — watch a directory, process new bundles as they arrive
    train-livedead  — train the LIVE/DEAD U-Net from a YAML config
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

from .ingest import load_bundle_from_dir
from .pipeline import process_chip
from .publish import MetricsPublisher

log = logging.getLogger("lfimaging.cli")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@main.command("process")
@click.option("--input", "input_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--chip-id", default=None, help="Filter to a single chip in the directory.")
@click.option("--campaign", default=None, help="Campaign label attached to metrics.")
@click.option("--publish/--no-publish", default=True, help="Publish to MQTT.")
@click.option("--output", type=click.Path(), default=None, help="Also write JSON to this path.")
def process_cmd(
    input_dir: str, chip_id: str | None, campaign: str | None, publish: bool, output: str | None
) -> None:
    """Process a single chip's ImageBundle from a directory."""
    bundle = load_bundle_from_dir(Path(input_dir), chip_id=chip_id)
    metrics = process_chip(bundle, campaign=campaign)
    click.echo(json.dumps(metrics.to_dict(), indent=2))

    if output:
        Path(output).write_text(json.dumps(metrics.to_dict(), indent=2))
        click.echo(f"wrote {output}")

    if publish:
        with MetricsPublisher() as pub:
            pub.publish(metrics.to_dict())
            click.echo(f"published to {pub.topic}")


@main.command("watch")
@click.option("--dir", "watch_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--pattern", default="*.tif", help="Glob pattern for incoming files.")
@click.option("--campaign", default=None)
@click.option("--interval", default=5.0, type=float, help="Poll interval (s) for new bundles.")
def watch_cmd(watch_dir: str, pattern: str, campaign: str | None, interval: float) -> None:
    """Watch a directory; process each new complete bundle and publish metrics."""
    seen: set[tuple[str, str]] = set()  # (chip_id, timestamp-iso)
    pub = MetricsPublisher()
    pub.connect()
    click.echo(f"watching {watch_dir} pattern={pattern}")
    try:
        while True:
            try:
                bundle = load_bundle_from_dir(Path(watch_dir))
            except ValueError:
                time.sleep(interval)
                continue
            key = (bundle.chip_id, bundle.timestamp.isoformat())
            if key in seen:
                time.sleep(interval)
                continue
            seen.add(key)
            metrics = process_chip(bundle, campaign=campaign)
            pub.publish(metrics.to_dict())
            click.echo(f"processed {key} -> {pub.topic}")
            time.sleep(interval)
    finally:
        pub.close()


@main.command("train-livedead")
@click.option("--config", type=click.Path(exists=True), required=True, help="YAML training config.")
def train_livedead_cmd(config: str) -> None:
    """Train the LIVE/DEAD U-Net. Requires ``[ml]`` extras installed."""
    from .train_livedead import train

    train(Path(config))


if __name__ == "__main__":
    main()
