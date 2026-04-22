[![CI](https://github.com/laminarforge-nonprofit/laminarforge-imaging-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/laminarforge-nonprofit/laminarforge-imaging-ml/actions/workflows/ci.yml)
[![Release](https://github.com/laminarforge-nonprofit/laminarforge-imaging-ml/actions/workflows/release.yml/badge.svg)](https://github.com/laminarforge-nonprofit/laminarforge-imaging-ml/actions/workflows/release.yml)

# laminarforge-imaging-ml

Python ML pipeline that turns organ-chip imaging gantry output into quantitative
per-chip metrics and publishes them to MQTT for the dashboard.

- **Input**: multi-channel images (brightfield, Calcein, PI, Hoechst, phase) from
  the gantry (see `A-07C2AA94`).
- **Output**: one `ChipMetrics` JSON per chip per timepoint, published to MQTT
  topic `imaging/metrics` (consumed by the dashboard, `T-DB0EA365 / A-BCBDCDC3`).

## Architecture

```
     gantry (S3 or /data)
             |
             v
  +-------------------+
  |   ingest.py       |  filenames: <chip>_<ts>_<channel>.{tif,png}
  +---------+---------+
            |
            v
  +-------------------+       +-------------------+
  |  segmentation.py  |<----->|  Cellpose cyto3   |
  +---------+---------+       +-------------------+
            |
            +--> nuclear.py  (Hoechst LoG blobs)
            +--> livedead.py (Calcein/PI heuristic, U-Net when weights present)
            +--> barrier.py  (monolayer holes)
            |
            v
  +-------------------+
  |   pipeline.py     |  ChipMetrics
  +---------+---------+
            |
            v
  +-------------------+
  |   publish.py      |  MQTT topic: imaging/metrics
  +-------------------+
```

## Install

Python 3.12+. `uv` is the preferred installer.

```bash
# Base install (pipeline + tests, no torch)
uv pip install -e '.[dev]'

# With the ML stack (Cellpose, torch) for production + U-Net training
uv pip install -e '.[ml,dev]'
```

Cellpose downloads the `cyto3` weights on first use; pre-warm offline:

```bash
python -c "from cellpose import models; models.Cellpose(model_type='cyto3')"
```

## Usage

### Single chip

```bash
lf-imaging process \
  --input ./data/chip-C1-S2/2026-04-20T12-30-00/ \
  --chip-id C1-S2 \
  --campaign 2026-04
```

Prints ChipMetrics as JSON and publishes to MQTT.

### Watch a directory

```bash
lf-imaging watch --dir /data/incoming --pattern '*.tif'
```

### Docker compose

```bash
cp .env.example .env   # set LF_MQTT_USER / LF_MQTT_PASS
docker compose up --build
```

Mounts `$LF_IMAGE_DIR` (defaults to `./data`) into the container and publishes to
the MQTT broker from env vars.

## MQTT configuration

Broker comes from `LF_MQTT_BROKER` (default `100.119.87.128:1883` — Tailscale
Mac Mini). Auth via `LF_MQTT_USER` / `LF_MQTT_PASS`.

**ACL provisioning request** (for the MQTT admin per `A-BCBDCDC3`): add user
`lf-imaging` with publish permissions on `imaging/#`.

## Retraining the LIVE/DEAD U-Net

Put two-channel TIFs (Calcein + PI stacked) in `images_dir/` and integer masks
(`0=bg, 1=live, 2=dead`) in `masks_dir/` with matching filenames. Then:

```bash
cat > livedead.yaml <<EOF
images_dir: data/livedead/images
masks_dir:  data/livedead/masks
output:     models/livedead_v1.pt
epochs:     50
batch_size: 8
lr:         1e-3
device:     mps
EOF

lf-imaging train-livedead --config livedead.yaml
export LF_LIVEDEAD_WEIGHTS=$(pwd)/models/livedead_v1.pt
```

With `LF_LIVEDEAD_WEIGHTS` set the pipeline automatically routes LIVE/DEAD through
the U-Net instead of the heuristic.

## Evaluation

Target: **IoU >= 0.85** vs manually annotated masks on a 200-chip ground-truth set.
Until we have that dataset, the heuristic LIVE/DEAD ships as v0 and is documented
as a known gap in `A-<this artifact>`.

## Tests

```bash
pytest -v
```

Tests generate synthetic images on the fly and exercise the classical-fallback
segmenter so CI does not require network access.
