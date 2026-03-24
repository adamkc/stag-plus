#!/usr/bin/env python3
"""
Quick test for BRISQUE IQ scoring.
Only requires: numpy, scipy, Pillow (no torch, no bs4).

Usage:
    python test_brisque.py [directory_or_image]

Defaults to scanning Z:/PhotoEdits/Exports/ if no argument given.
"""

import os
import sys
import glob
import time
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import skew, kurtosis
from PIL import Image


# --- BRISQUE implementation (copied from stag.py for standalone use) ---

def _mscn(img_grey, kernel_size=7):
    C = 1.0 / 255
    mu = gaussian_filter(img_grey.astype(np.float64), sigma=kernel_size / 6)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(
        gaussian_filter(img_grey.astype(np.float64) ** 2, sigma=kernel_size / 6) - mu_sq
    ))
    return (img_grey - mu) / (sigma + C)


def _compute_brisque_features(img_grey):
    features = []
    for scale in [1.0, 0.5]:
        img_s = zoom(img_grey, scale, order=1) if scale != 1.0 else img_grey
        mscn_c = _mscn(img_s)
        features.extend([
            np.mean(mscn_c), np.var(mscn_c),
            skew(mscn_c.ravel()), kurtosis(mscn_c.ravel())
        ])
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            shifted = np.roll(np.roll(mscn_c, dy, axis=0), dx, axis=1)
            pair = mscn_c * shifted
            features.extend([
                np.mean(pair), np.var(pair),
                skew(pair.ravel()), kurtosis(pair.ravel())
            ])
    return np.array(features, dtype=np.float64)


def compute_brisque_score(pil_image):
    img = pil_image.convert("L")
    max_dim = 512
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    feats = _compute_brisque_features(np.array(img, dtype=np.float64))

    mscn_var_s1 = feats[1]
    mscn_var_s2 = feats[21]
    pair_kurt_s1 = np.mean([feats[7], feats[11], feats[15], feats[19]])
    pair_kurt_s2 = np.mean([feats[27], feats[31], feats[35], feats[39]])
    mscn_skew_s1 = abs(feats[2])
    mscn_kurt_s1 = feats[3]
    pair_var_s1 = np.mean([feats[5], feats[9], feats[13], feats[17]])

    raw_score = (
        (1.0 - np.clip(mscn_var_s1, 0, 1)) * 25
        + (1.0 - np.clip(mscn_var_s2, 0, 1)) * 10
        + np.clip(pair_kurt_s1, 0, 50) * 0.3
        + np.clip(pair_kurt_s2, 0, 50) * 0.15
        + mscn_skew_s1 * 3.0
        + np.clip(abs(mscn_kurt_s1) - 1.0, 0, 10) * 1.0
        + (1.0 - np.clip(pair_var_s1 * 5, 0, 1)) * 5
    )

    normalised = 1.0 - np.clip((raw_score - 15.0) / 35.0, 0.0, 1.0)
    return float(normalised)


def score_to_bin(score, n_bins=3):
    if n_bins == 5:
        names = ["poor", "below_average", "average", "good", "excellent"]
    else:
        names = ["low", "medium", "high"]
    return names[min(int(score * n_bins), n_bins - 1)]


def score_to_stars(score):
    return max(1, min(5, int(score * 5) + 1))


# --- Main ---

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "Z:/PhotoEdits/Exports"

    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = []
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.tiff"):
            files.extend(glob.glob(os.path.join(target, "**", ext), recursive=True))
        files = sorted(files)[:50]  # cap at 50 for testing
    else:
        print(f"Not found: {target}")
        return

    if not files:
        print(f"No images found in {target}")
        return

    print(f"BRISQUE IQ Test — {len(files)} images from {target}\n")
    print(f"{'File':<35s} {'Score':>6s}  {'3-bin':<8s} {'5-bin':<15s} {'Stars':>5s}  {'Time':>5s}")
    print("-" * 85)

    scores = []
    for f in files:
        t0 = time.time()
        try:
            score = compute_brisque_score(Image.open(f))
        except Exception as e:
            print(f"{os.path.basename(f):<35s}  ERROR: {e}")
            continue
        dt = time.time() - t0
        scores.append(score)
        print(f"{os.path.basename(f):<35s} {score:>6.3f}  {score_to_bin(score, 3):<8s} "
              f"{score_to_bin(score, 5):<15s} {'*' * score_to_stars(score):>5s}  {dt:.2f}s")

    if scores:
        print("-" * 85)
        print(f"Range: {min(scores):.3f} - {max(scores):.3f}  "
              f"Mean: {np.mean(scores):.3f}  Spread: {max(scores) - min(scores):.3f}")


if __name__ == "__main__":
    main()
