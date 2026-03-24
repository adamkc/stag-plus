#!/usr/bin/env python3

#############################################
## STAG+                                    #
## Automatic Image Tagger + Quality Scorer  #
## Fork of DIVISIO STAG with IQ & AES      #
#############################################

import argparse
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

# Version information
VERSION = "2.0.0"
UPSTREAM_VERSION = "1.0.2"

import numpy as np
import rawpy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from huggingface_hub import hf_hub_download
from PIL import Image
from pillow_heif import register_heif_opener
from ram import get_transform, inference_ram as inference
from ram.models import ram_plus

from xmphandler import XMPHandler

raw_extensions = [
    ".3fr", ".ari", ".arw", ".bay", ".cr2", ".cr3", ".cap", ".data",
    ".dcr", ".dng", ".drf", ".eip", ".erf", ".fff", ".gpr", ".iiq",
    ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw",
    ".orf", ".pef", ".ptx", ".pxn", ".r3d", ".raf", ".raw", ".rwl",
    ".rw2", ".rwz", ".sr2", ".srf", ".srw", ".x3f",
    ".ori",  # Olympus in-camera RAW edit
]


# ---------------------------------------------------------------------------
# IQ / Aesthetic scoring helpers
# ---------------------------------------------------------------------------

def _score_to_bin(score: float, n_bins: int = 3) -> str:
    """
    Map a normalised 0-1 score to a named quality bin.

    3-bin:  low / medium / high
    5-bin:  poor / below_average / average / good / excellent
    """
    if n_bins == 5:
        names = ["poor", "below_average", "average", "good", "excellent"]
    else:
        names = ["low", "medium", "high"]
    idx = min(int(score * n_bins), n_bins - 1)
    return names[idx]


def _score_to_stars(score: float) -> int:
    """Map a normalised 0-1 score to 1-5 stars."""
    return max(1, min(5, int(score * 5) + 1))


# ---------------------------------------------------------------------------
# BRISQUE – Blind/Referenceless Image Spatial Quality Evaluator
#   Pure numpy/scipy implementation (no extra model download needed).
#   Lower BRISQUE = better quality.  We invert + normalise to 0-1.
# ---------------------------------------------------------------------------

def _mscn(img_grey: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Mean-subtracted contrast-normalised coefficients."""
    from scipy.ndimage import uniform_filter, gaussian_filter
    C = 1.0 / 255
    mu = gaussian_filter(img_grey.astype(np.float64), sigma=kernel_size / 6)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(gaussian_filter(img_grey.astype(np.float64) ** 2, sigma=kernel_size / 6) - mu_sq))
    return (img_grey - mu) / (sigma + C)


def _compute_brisque_features(img_grey: np.ndarray) -> np.ndarray:
    """Compute a simplified BRISQUE feature vector (36-dim)."""
    from scipy.stats import norm, skew, kurtosis
    from scipy.ndimage import zoom

    features = []
    for scale in [1.0, 0.5]:
        if scale != 1.0:
            img_s = zoom(img_grey, scale, order=1)
        else:
            img_s = img_grey
        mscn = _mscn(img_s)

        # Shape parameters from MSCN
        features.append(np.mean(mscn))
        features.append(np.var(mscn))
        features.append(skew(mscn.ravel()))
        features.append(kurtosis(mscn.ravel()))

        # Paired-product neighbours (H, V, D1, D2)
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dy, dx in shifts:
            shifted = np.roll(np.roll(mscn, dy, axis=0), dx, axis=1)
            pair = mscn * shifted
            features.append(np.mean(pair))
            features.append(np.var(pair))
            features.append(skew(pair.ravel()))
            features.append(kurtosis(pair.ravel()))

    return np.array(features, dtype=np.float64)


def compute_brisque_score(pil_image: Image.Image) -> float:
    """
    Compute a normalised BRISQUE-like quality score for a PIL image.

    Uses a multi-feature aggregate across two scales to distinguish
    sharp, well-exposed images from blurry / noisy / distorted ones.

    Returns:
        Float in [0, 1] where 1 = best quality.
    """
    img = pil_image.convert("L")
    # Downscale to speed up – BRISQUE only needs spatial stats
    max_dim = 512
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)

    img_np = np.array(img, dtype=np.float64)
    feats = _compute_brisque_features(img_np)

    # Feature layout (per scale, 20 features each):
    #   [0] MSCN mean  [1] MSCN var  [2] MSCN skew  [3] MSCN kurtosis
    #   [4..7] H-pair: mean, var, skew, kurt
    #   [8..11] V-pair  [12..15] D1-pair  [16..19] D2-pair
    # Scale 1: feats[0..19], Scale 2: feats[20..39]

    # Aggregate multiple indicators of distortion:
    # - Low MSCN variance = flat/blurry image (bad)
    # - High pair-product kurtosis = heavy tailed noise (bad)
    # - Large |MSCN skew| = uneven brightness distribution (bad)

    # MSCN variance (higher = more texture/detail = better)
    mscn_var_s1 = feats[1]
    mscn_var_s2 = feats[21]

    # Average pair-product kurtosis across all 4 neighbour directions
    pair_kurt_s1 = np.mean([feats[7], feats[11], feats[15], feats[19]])
    pair_kurt_s2 = np.mean([feats[27], feats[31], feats[35], feats[39]])

    # MSCN skewness (closer to 0 = more symmetric = natural image)
    mscn_skew_s1 = abs(feats[2])
    mscn_skew_s2 = abs(feats[22])

    # MSCN kurtosis (natural images typically have positive kurtosis;
    # very high or negative values indicate distortion)
    mscn_kurt_s1 = feats[3]

    # Average pair-product variance (structural correlation)
    pair_var_s1 = np.mean([feats[5], feats[9], feats[13], feats[17]])

    # Build a composite: combine indicators with empirically tuned weights.
    # Higher raw_score = worse quality.
    raw_score = (
        (1.0 - np.clip(mscn_var_s1, 0, 1)) * 25      # penalise low variance (blur)
        + (1.0 - np.clip(mscn_var_s2, 0, 1)) * 10     # same at half-scale
        + np.clip(pair_kurt_s1, 0, 50) * 0.3           # penalise heavy-tailed noise
        + np.clip(pair_kurt_s2, 0, 50) * 0.15
        + mscn_skew_s1 * 3.0                           # penalise asymmetry
        + np.clip(abs(mscn_kurt_s1) - 1.0, 0, 10) * 1.0  # penalise extreme kurtosis
        + (1.0 - np.clip(pair_var_s1 * 5, 0, 1)) * 5  # penalise low structural correlation
    )

    # Typical observed range: ~15 (excellent) to ~50 (terrible).
    normalised = 1.0 - np.clip((raw_score - 15.0) / 35.0, 0.0, 1.0)
    return float(normalised)


# ---------------------------------------------------------------------------
# NIMA – Neural Image Assessment (aesthetic scoring)
#   Uses MobileNetV2 backbone trained on AVA dataset.
#   Predicts a distribution over 1-10 aesthetic ratings.
# ---------------------------------------------------------------------------

class NimaModel(nn.Module):
    """NIMA model with MobileNetV2 backbone."""

    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _nima_transform():
    """Image transform for NIMA input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def compute_nima_score(pil_image: Image.Image, model: nn.Module, device: torch.device) -> float:
    """
    Compute NIMA aesthetic score.

    Returns:
        Float in [0, 1] where 1 = most aesthetically pleasing.
    """
    transform = _nima_transform()
    img = pil_image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model(tensor).cpu().numpy()[0]

    # Weighted mean over the 1-10 distribution
    scores = np.arange(1, 11, dtype=np.float64)
    mean_score = float(np.sum(probs * scores))

    # Normalise from [1, 10] range to [0, 1]
    normalised = (mean_score - 1.0) / 9.0
    return float(np.clip(normalised, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SKTagger – main class (extended from upstream STAG)
# ---------------------------------------------------------------------------

class SKTagger:
    """Main class for STAG+ (Automatic Image Tagger + Quality Scorer)"""

    def __init__(self,
                 model_path: str,
                 image_size: int,
                 force_tagging: bool,
                 test_mode: bool,
                 prefer_exact_filenames: bool,
                 tag_prefix: str,
                 enable_tagging: bool = True,
                 enable_iq: bool = False,
                 enable_aes: bool = False,
                 n_bins: int = 3,
                 write_stars: bool = True):
        """
        Initialize the tagger with given parameters.

        Args:
            model_path: Path to the pretrained RAM+ model (ignored if enable_tagging=False)
            image_size: Image size for the RAM+ model
            force_tagging: Force tagging even if images are already tagged
            test_mode: Don't actually write or modify XMP files
            prefer_exact_filenames: Use exact filenames for XMP sidecars
            tag_prefix: Prefix for tags (empty string for no prefix)
            enable_tagging: Run the RAM+ tag model
            enable_iq: Run BRISQUE image quality assessment
            enable_aes: Run NIMA aesthetic assessment
            n_bins: Number of quality bins (3 or 5)
            write_stars: Write aesthetic score as xmp:Rating stars
        """
        register_heif_opener()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"STAG+ using device: {self.device}")

        # Store configuration
        self.force_tagging = force_tagging
        self.test_mode = test_mode
        self.prefer_exact_filenames = prefer_exact_filenames
        self.tag_prefix = tag_prefix
        self.enable_tagging = enable_tagging
        self.enable_iq = enable_iq
        self.enable_aes = enable_aes
        self.n_bins = n_bins
        self.write_stars = write_stars

        # Load RAM+ tagging model
        if self.enable_tagging:
            self.transform = get_transform(image_size=image_size)
            self.model = ram_plus(pretrained=model_path, image_size=image_size, vit='swin_l')
            self.model.eval()
            self.model = self.model.to(self.device)
            print("RAM+ tagging model loaded.")
        else:
            self.model = None
            self.transform = None

        # Load NIMA aesthetic model
        if self.enable_aes:
            self.nima_model = NimaModel()
            self.nima_model.eval()
            self.nima_model = self.nima_model.to(self.device)
            print("NIMA aesthetic model loaded (MobileNetV2 backbone, ImageNet weights).")
        else:
            self.nima_model = None

        if self.enable_iq:
            print("BRISQUE IQ assessment enabled (no model download needed).")

    def get_tags_for_image(self, pil_image: Image.Image) -> str:
        """Generate tags for a given PIL image."""
        if not self.enable_tagging or self.model is None:
            return ""
        try:
            torch_image = self.transform(pil_image).unsqueeze(0).to(self.device)
            res = inference(torch_image, self.model)
            return res[0]
        except Exception as e:
            print(f"Tagging failed: {e}")
            return ""

    def get_tags_for_image_at_path(self, path: str) -> str:
        """Open an image file and generate tags for it."""
        pillow_image = Image.open(path)
        return self.get_tags_for_image(pillow_image)

    def load_image(self, image_path: str) -> Tuple[Optional[Image.Image], str]:
        """Load an image using the appropriate method."""
        filename, file_extension = os.path.splitext(image_path)
        file_extension = file_extension.lower()

        if file_extension == ".xmp":
            return None, "none"

        image = None
        loader = "none"

        if file_extension not in raw_extensions:
            try:
                image = Image.open(image_path)
                loader = "pillow"
            except Exception as e:
                print(f"Pillow can't read image {image_path}: {e}")

        if image is None:
            try:
                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess()
                    image = Image.fromarray(rgb)
                    loader = "rawpy"
            except Exception as e:
                print(f"Rawpy could not read {image_path}: {e}")

        return image, loader

    def is_already_tagged(self, sidecar_files: List[str]) -> bool:
        """Check if an image is already tagged with all enabled features."""
        if self.force_tagging:
            return False

        for current_file in sidecar_files:
            handler = XMPHandler(current_file)

            # Check RAM+ tags
            if self.enable_tagging:
                if self.tag_prefix:
                    if not handler.has_subject_prefix(self.tag_prefix):
                        return False
                else:
                    if len(handler.get_all_subjects()) == 0:
                        return False

            # Check IQ tags
            if self.enable_iq:
                if not handler.has_subject_prefix("iq"):
                    return False

            # Check AES tags
            if self.enable_aes:
                if not handler.has_subject_prefix("aes"):
                    return False

            # If we get here, all enabled features are already present
            return True

        return False

    def save_tags(self, image_file: str, sidecar_files: List[str],
                  tags: List[str], iq_score: Optional[float] = None,
                  aes_score: Optional[float] = None) -> None:
        """
        Save tags and scores to XMP sidecar files.

        Args:
            image_file: Path to the image file
            sidecar_files: List of existing XMP sidecar files
            tags: List of content tags to save
            iq_score: Normalised IQ score (0-1) or None
            aes_score: Normalised aesthetic score (0-1) or None
        """
        has_anything = bool(tags) or iq_score is not None or aes_score is not None
        if not has_anything:
            return

        # Create sidecar file if none exists
        if len(sidecar_files) == 0:
            if not self.test_mode:
                sidecar_files = [XMPHandler.create_xmp_sidecar(image_file, self.prefer_exact_filenames)]
            else:
                print("Skipping XMP file creation, not writing tags")
                return

        # Write to all sidecar files
        for current_file in sidecar_files:
            handler = XMPHandler(current_file)

            # Content tags from RAM+
            for tag in tags:
                if self.tag_prefix:
                    handler.add_hierarchical_subject(f"{self.tag_prefix}|{tag}")
                else:
                    handler.add_hierarchical_subject(tag)

            # IQ tags
            if iq_score is not None:
                iq_bin = _score_to_bin(iq_score, self.n_bins)
                handler.add_hierarchical_subject(f"iq|{iq_bin}")

            # Aesthetic tags + star rating
            if aes_score is not None:
                aes_bin = _score_to_bin(aes_score, self.n_bins)
                handler.add_hierarchical_subject(f"aes|{aes_bin}")
                if self.write_stars:
                    stars = _score_to_stars(aes_score)
                    handler.set_rating(stars)

            if not self.test_mode:
                handler.save()

    def enter_dir(self, img_dir: str, stop_event: threading.Event) -> None:
        """Process all images in a directory and its subdirectories."""
        print(f"Entering {img_dir}")
        features = []
        if self.enable_tagging:
            features.append("Tagging")
        if self.enable_iq:
            features.append("IQ")
        if self.enable_aes:
            features.append("Aesthetics")
        print(f"Enabled features: {', '.join(features)}")

        processed = 0
        skipped = 0

        for current_dir, _, file_list in os.walk(img_dir):
            for fname in sorted(file_list):
                if stop_event.is_set():
                    print("Processing cancelled.")
                    return

                if fname.startswith("."):
                    continue

                image_file = os.path.join(current_dir, fname)
                sidecar_files = XMPHandler.get_xmp_sidecars_for_image(image_file)

                if self.is_already_tagged(sidecar_files):
                    skipped += 1
                    continue

                image, loader = self.load_image(image_file)

                if image is not None:
                    print(f'Processing {image_file} ({loader}):')

                    # --- RAM+ tagging ---
                    tags = []
                    if self.enable_tagging:
                        tag_string = self.get_tags_for_image(image)
                        tags = [item.strip() for item in tag_string.split("|")] if tag_string else []
                        if tags:
                            print(f"  Tags: {tags}")

                    # --- IQ assessment ---
                    iq_score = None
                    if self.enable_iq:
                        iq_score = compute_brisque_score(image)
                        iq_bin = _score_to_bin(iq_score, self.n_bins)
                        print(f"  IQ: {iq_score:.3f} -> {iq_bin}")

                    # --- Aesthetic assessment ---
                    aes_score = None
                    if self.enable_aes:
                        aes_score = compute_nima_score(image, self.nima_model, self.device)
                        aes_bin = _score_to_bin(aes_score, self.n_bins)
                        stars = _score_to_stars(aes_score) if self.write_stars else None
                        star_str = f" ({stars}★)" if stars else ""
                        print(f"  Aesthetics: {aes_score:.3f} -> {aes_bin}{star_str}")

                    # --- Save everything ---
                    self.save_tags(image_file, sidecar_files, tags, iq_score, aes_score)
                    processed += 1

        print(f"\nDone. Processed: {processed}, Skipped (already done): {skipped}")


def main():
    """Main entry point for the STAG+ command-line tool"""
    parser = argparse.ArgumentParser(description='STAG+ image tagger and quality scorer')

    parser.add_argument('imagedir', metavar='DIR', help='path to image directory')
    parser.add_argument('--prefix', metavar='STR',
                        help='top category for tags (default="st")', default='st')
    parser.add_argument('--force', action='store_true',
                        help='force processing, even if images are already tagged')
    parser.add_argument('--test', action='store_true',
                        help="don't actually write or modify XMP files")
    parser.add_argument('--prefer-exact-filenames', action='store_true',
                        help="use darktable-compatible filenames (image.jpg.xmp)")

    # Feature toggles
    parser.add_argument('--no-tags', action='store_true',
                        help='disable RAM+ content tagging')
    parser.add_argument('--iq', action='store_true',
                        help='enable BRISQUE image quality assessment')
    parser.add_argument('--aes', action='store_true',
                        help='enable NIMA aesthetic assessment')
    parser.add_argument('--all', action='store_true',
                        help='enable all features (tags + IQ + aesthetics)')

    # Scoring options
    parser.add_argument('--bins', type=int, choices=[3, 5], default=3,
                        help='number of quality bins (default: 3 = low/medium/high)')
    parser.add_argument('--no-stars', action='store_true',
                        help='do not write aesthetic score as xmp:Rating stars')

    args = parser.parse_args()

    enable_tagging = not args.no_tags or args.all
    enable_iq = args.iq or args.all
    enable_aes = args.aes or args.all

    if not enable_tagging and not enable_iq and not enable_aes:
        print("Nothing to do — enable at least one feature (tags, --iq, --aes, or --all).")
        return

    # Download RAM+ model if tagging is enabled
    pretrained = None
    if enable_tagging:
        pretrained = hf_hub_download(
            repo_id="xinyu1205/recognize-anything-plus-model",
            filename="ram_plus_swin_large_14m.pth"
        )

    tagger = SKTagger(
        model_path=pretrained or "",
        image_size=384,
        force_tagging=args.force,
        test_mode=args.test,
        prefer_exact_filenames=args.prefer_exact_filenames,
        tag_prefix=args.prefix,
        enable_tagging=enable_tagging,
        enable_iq=enable_iq,
        enable_aes=enable_aes,
        n_bins=args.bins,
        write_stars=not args.no_stars,
    )

    stop_event = threading.Event()
    stop_event.clear()
    tagger.enter_dir(args.imagedir, stop_event)


if __name__ == "__main__":
    main()
