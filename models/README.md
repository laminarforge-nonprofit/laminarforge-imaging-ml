# Models

Weights are not checked in (see root `.gitignore`). Download/cache paths:

## Cellpose `cyto3`
Cellpose 3.x auto-downloads `cyto3` weights on first use from the official
release mirror. Cache location:

```
~/.cellpose/models/
```

To pre-warm for offline runs (e.g., air-gapped Mac Mini):

```bash
python -c "from cellpose import models; models.Cellpose(model_type='cyto3')"
```

## LIVE/DEAD U-Net (retrained)
When training completes via `lf-imaging train-livedead --config config.yaml`, the
state dict is written to:

```
models/livedead_v1.pt
```

Set `LF_LIVEDEAD_WEIGHTS=$(pwd)/models/livedead_v1.pt` to enable U-Net inference
in `analyze_livedead`. Until that file exists, the threshold heuristic is used.

## Ground-truth dataset
Target: 200 labeled organ-chip images (Calcein + PI two-channel TIFs with matching
integer mask PNGs). Label convention: `0=background, 1=live, 2=dead`. Bootstrap
sources:

- MIMETAS OrganoPlate public dataset (https://mimetas.com/)
- Emulate Organ-Chip published LIVE/DEAD stains
- Internal gantry captures from campaigns `2026-04` onward
