# STAG+

AI-powered image tagger with image quality and aesthetic scoring. Fork of [DIVISIO STAG](https://github.com/DIVISIO-AI/stag) with two new features: technical quality assessment and aesthetic rating.

All processing runs **100% locally** — no images are uploaded anywhere.

## Features

| Feature | Model | What it does | XMP output |
|---------|-------|-------------|------------|
| **Content Tagging** | RAM+ (Recognize Anything) | Identifies objects, scenes, concepts | `st\|landscape`, `st\|mushroom`, `st\|sky` |
| **Image Quality** | BRISQUE (no-reference IQA) | Detects blur, noise, distortion | `iq\|low`, `iq\|medium`, `iq\|high` |
| **Aesthetic Score** | NIMA (InceptionV2, AVA-trained) | Rates composition and visual appeal | `aes\|low`, `aes\|medium`, `aes\|high` + star rating |

- Tags are written to **XMP sidecar files** (originals are never modified)
- Aesthetic scores are also written as **xmp:Rating** (1-5 stars) for instant filtering in darktable/Lightroom
- Supports **RAW files** from 40+ camera manufacturers, plus JPEG, TIFF, PNG, HEIF
- **darktable-compatible** sidecar naming (`image.jpg.xmp`)
- **RAW+JPG deduplication** — when both `image.JPG` and `image.ORF` exist, only the JPG is analyzed (faster), and XMP sidecars are written for both files
- GUI and CLI interfaces
- 3-bin (`low/medium/high`) or 5-bin (`poor/below_average/average/good/excellent`) scoring

## Quick Start

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

### Windows Batch Launchers

Two `.bat` files are included for convenience:

- **`run_inbox.bat`** — Runs IQ + Aesthetics on the `_Inbox/` folder (fast, no RAM+ tagging)
- **`run_all.bat`** — Runs all three features on the entire photo library

### GUI

```bash
python stag_gui.py
```

Select a directory, check the features you want, and click Run.

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

Uses an InceptionV2 backbone trained on the [AVA dataset](https://github.com/mtobeiyf/ava_downloader) (255k+ photos with human aesthetic ratings) via [PyIQA](https://github.com/chaofengc/IQA-PyTorch). Predicts aesthetic quality on a 1-10 scale, then normalises to 0-1.

The aesthetic score is mapped to XMP star ratings using percentile-based thresholds tuned to NIMA's real-world score distribution:

| Score | Stars | Tag (3-bin) |
|-------|-------|-------------|
| < 0.28 | 1 | `aes\|low` |
| 0.28 - 0.33 | 2 | `aes\|low` |
| 0.33 - 0.40 | 3 | `aes\|medium` |
| 0.40 - 0.50 | 4 | `aes\|medium` |
| > 0.50 | 5 | `aes\|high` |

### RAW+JPG Deduplication

When a directory contains both `P1234567.JPG` and `P1234567.ORF` (or any RAW variant), STAG+ only processes the JPG — it's much faster to decode than the RAW file. XMP sidecars are written for **both** the JPG and the RAW file, so darktable sees the tags on either.

This typically cuts processing time by ~50% for cameras shooting RAW+JPG.

## Performance

Approximate per-image times on CPU:

| Feature | Time | Notes |
|---------|------|-------|
| BRISQUE IQ | ~0.2s | No model download needed |
| NIMA Aesthetics | ~0.4s | 208 MB model (auto-downloaded once) |
| RAM+ Tagging | ~3-5s | 3.2 GB model (auto-downloaded once) |

For a 13k image library: IQ+AES takes ~2-3 hours. All three features takes ~12-20 hours (run overnight).

## Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch + torchvision (CPU is fine, GPU optional)
- See `requirements.txt` for full list

Models are downloaded automatically on first run:
- **RAM+** (~3.2 GB) from HuggingFace — for content tagging
- **NIMA** (~208 MB) from HuggingFace — for aesthetic scoring
- **BRISQUE** needs no model download

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
- NIMA aesthetic scoring via [PyIQA](https://github.com/chaofengc/IQA-PyTorch) (based on [NIMA paper](https://arxiv.org/abs/1709.05424), InceptionV2 trained on AVA dataset)

## License

Apache-2.0 — same as upstream STAG.
