# STAG+

AI-powered image tagger with image quality and aesthetic scoring. Fork of [DIVISIO STAG](https://github.com/DIVISIO-AI/stag) with two new features: technical quality assessment and aesthetic rating.

All processing runs **100% locally** — no images are uploaded anywhere.

## Features

| Feature | Model | What it does | XMP output |
|---------|-------|-------------|------------|
| **Content Tagging** | RAM+ (Recognize Anything) | Identifies objects, scenes, concepts | `st\|landscape`, `st\|mushroom`, `st\|sky` |
| **Image Quality** | BRISQUE (no-reference IQA) | Detects blur, noise, distortion | `iq\|low`, `iq\|medium`, `iq\|high` |
| **Aesthetic Score** | NIMA (MobileNetV2) | Rates composition and visual appeal | `aes\|low`, `aes\|medium`, `aes\|high` + star rating |

- Tags are written to **XMP sidecar files** (originals are never modified)
- Aesthetic scores are also written as **xmp:Rating** (1-5 stars) for instant filtering in darktable/Lightroom
- Supports **RAW files** from 40+ camera manufacturers, plus JPEG, TIFF, PNG, HEIF
- **darktable-compatible** sidecar naming (`image.jpg.xmp`)
- GUI and CLI interfaces
- 3-bin (`low/medium/high`) or 5-bin (`poor/below_average/average/good/excellent`) scoring

## Quick Start

### GUI

```bash
python stag_gui.py
```

Select a directory, check the features you want, and click Run.

### CLI

```bash
# Tag + score everything
python stag.py /path/to/photos --all --prefer-exact-filenames

# Just IQ scoring (fast, no model download needed)
python stag.py /path/to/photos --no-tags --iq

# Just aesthetic scoring
python stag.py /path/to/photos --no-tags --aes

# 5-bin scoring, test mode (no writes)
python stag.py /path/to/photos --all --bins 5 --test

# Skip star ratings
python stag.py /path/to/photos --all --no-stars
```

## CLI Options

```
positional arguments:
  DIR                     Path to image directory

options:
  --prefix STR            Tag prefix (default: "st")
  --force                 Re-process already tagged images
  --test                  Dry run, don't write XMP files
  --prefer-exact-filenames  Use darktable-compatible naming (image.jpg.xmp)

feature toggles:
  --no-tags               Disable RAM+ content tagging
  --iq                    Enable BRISQUE image quality assessment
  --aes                   Enable NIMA aesthetic assessment
  --all                   Enable all features

scoring:
  --bins {3,5}            Number of quality bins (default: 3)
  --no-stars              Don't write aesthetic score as xmp:Rating
```

## How Scoring Works

### Image Quality (BRISQUE)

Uses blind/referenceless spatial statistics to assess technical quality. No neural network needed — runs on pure math (numpy/scipy). Very fast (~0.2s per image on CPU).

Detects:
- Blur / softness (low MSCN variance)
- Noise (high pair-product kurtosis)
- Distortion (skewness, extreme kurtosis)

### Aesthetic Score (NIMA)

Uses a MobileNetV2 backbone to predict aesthetic appeal on a 1-10 distribution, then normalises to 0-1. Evaluates composition, lighting, subject placement, and overall visual quality.

The aesthetic score is mapped to XMP star ratings:

| Score range | Stars | Tag (3-bin) |
|-------------|-------|-------------|
| 0.0 - 0.33 | 1-2   | `aes\|low` |
| 0.33 - 0.66 | 3     | `aes\|medium` |
| 0.66 - 1.0 | 4-5   | `aes\|high` |

## Requirements

- Python 3.10+
- PyTorch + torchvision
- See `requirements.txt` for full list

The RAM+ model (~3.2 GB) is downloaded automatically from HuggingFace on first run. BRISQUE needs no model download. NIMA uses torchvision's bundled MobileNetV2 ImageNet weights.

## darktable Workflow

1. Run STAG+ on your photo library with `--prefer-exact-filenames`
2. Open darktable and import the folder
3. darktable auto-detects the XMP sidecars
4. Use **lighttable > filtering** to:
   - Filter by star rating for quick aesthetic culling
   - Filter by tag (`iq|high` + `aes|high`) to find your best shots
   - Browse the `st|` tag tree for content-based discovery

## Credits

Fork of [DIVISIO STAG](https://github.com/DIVISIO-AI/stag) by DIVISIO AI (Apache-2.0).

Added features:
- BRISQUE image quality assessment (based on [BRISQUE paper](https://ieeexplore.ieee.org/document/6272356))
- NIMA aesthetic scoring (based on [NIMA paper](https://arxiv.org/abs/1709.05424), using torchvision MobileNetV2)

## License

Apache-2.0 — same as upstream STAG.
