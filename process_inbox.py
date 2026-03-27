"""
Inbox Processor — Combined STAG+ scoring + date-based sorting + dedup.

Steps:
  1. Run STAG+ (tags, IQ, aesthetics) on _Inbox/
  2. Move files to YYYY/YYYY-MM/ folders by file modification date
  3. Skip duplicates (same filename + same size already exists)
  4. Move XMP sidecars alongside their parent files

USAGE:
    python process_inbox.py              # Dry run
    python process_inbox.py --execute    # Score, sort, and move for real
    python process_inbox.py --skip-stag  # Skip STAG+ scoring, just sort
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path("Z:/PhotoEdits")
INBOX = ROOT / "_Inbox"
STAG_DIR = Path(__file__).parent
STAG_PY = STAG_DIR / "stag.py"
VENV_PYTHON = STAG_DIR / ".venv" / "Scripts" / "python.exe"

EXECUTE = "--execute" in sys.argv
SKIP_STAG = "--skip-stag" in sys.argv

RAW_EXTS = {
    ".orf", ".ori", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf",
    ".rw2", ".3fr", ".ari", ".bay", ".cap", ".data", ".dcr", ".drf",
    ".eip", ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc",
    ".mef", ".mos", ".mrw", ".nrw", ".pef", ".ptx", ".pxn", ".r3d",
    ".raw", ".rwl", ".rwz", ".sr2", ".srf", ".srw", ".x3f"
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heif", ".heic"} | RAW_EXTS
VIDEO_EXTS = {".mov", ".mp4", ".avi"}
ALL_EXTS = IMG_EXTS | VIDEO_EXTS | {".xmp"}

stats = {"scored": 0, "moved": 0, "skipped_dupe": 0, "skipped_xmp": 0, "errors": 0}


def get_date_dest(filepath):
    """Get YYYY/YYYY-MM/ destination based on file modification time."""
    mtime = datetime.fromtimestamp(filepath.stat().st_mtime)

    # For XMP sidecars, use the parent image's date
    if filepath.suffix.lower() == ".xmp":
        parent_name = filepath.stem  # e.g., P1234.ORF from P1234.ORF.xmp
        parent_file = filepath.parent / parent_name
        if parent_file.exists():
            mtime = datetime.fromtimestamp(parent_file.stat().st_mtime)

    year = mtime.strftime("%Y")
    month = mtime.strftime("%Y-%m")
    return ROOT / year / month / filepath.name


def is_duplicate(src, dest):
    """Check if dest exists and is same size (likely same file)."""
    if not dest.exists():
        return False
    return dest.stat().st_size == src.stat().st_size


def resolve_conflict(dest):
    """Add _1, _2, etc. suffix for different-size conflicts."""
    base = dest.stem
    ext = dest.suffix
    parent = dest.parent
    counter = 1
    while True:
        new_path = parent / f"{base}_{counter}{ext}"
        if not new_path.exists():
            return new_path
        counter += 1


def run_stag():
    """Run STAG+ on the inbox."""
    print("=" * 60)
    print("  Step 1: Running STAG+ on _Inbox")
    print("=" * 60)
    print()

    cmd = [
        str(VENV_PYTHON), str(STAG_PY),
        str(INBOX),
        "--all", "--prefer-exact-filenames"
    ]
    result = subprocess.run(cmd)
    print()
    if result.returncode != 0:
        print("WARNING: STAG+ exited with errors. Continuing with sort...")
    return result.returncode == 0


def sort_inbox():
    """Move files from _Inbox to date-based folders."""
    print("=" * 60)
    print(f"  Step 2: Sorting _Inbox ({'EXECUTE' if EXECUTE else 'DRY RUN'})")
    print("=" * 60)
    print()

    # Collect all files
    files = []
    for f in INBOX.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALL_EXTS:
            files.append(f)

    if not files:
        print("  No files to process in _Inbox/")
        return

    # Sort: process images/videos first, then XMPs (so XMPs follow their parents)
    images = [f for f in files if f.suffix.lower() != ".xmp"]
    xmps = [f for f in files if f.suffix.lower() == ".xmp"]
    files = sorted(images) + sorted(xmps)

    total = len(files)
    print(f"  Found {total} files to sort")
    print()

    # Track where we moved things so XMPs can follow
    moved_map = {}  # original_path -> dest_path

    for i, fpath in enumerate(files, 1):
        rel = fpath.relative_to(INBOX)
        ext = fpath.suffix.lower()
        dest = get_date_dest(fpath)

        # For XMP sidecars, try to follow the parent image
        if ext == ".xmp":
            parent_name = fpath.stem  # e.g., P1234.ORF
            parent_src = fpath.parent / parent_name
            if parent_src in moved_map:
                # Follow parent to its destination folder
                parent_dest = moved_map[parent_src]
                dest = parent_dest.parent / fpath.name

        if is_duplicate(fpath, dest):
            print(f"  [{i}/{total}] SKIP (dupe): {rel}")
            stats["skipped_dupe"] += 1
            continue

        if dest.exists():
            # Different file, same name — rename
            dest = resolve_conflict(dest)
            print(f"  [{i}/{total}] RENAME: {rel} -> {dest.relative_to(ROOT)}")
        else:
            print(f"  [{i}/{total}] MOVE: {rel} -> {dest.relative_to(ROOT)}")

        if EXECUTE:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fpath), str(dest))
                moved_map[fpath] = dest
                stats["moved"] += 1
            except Exception as e:
                print(f"  [{i}/{total}] ERROR: {e}")
                stats["errors"] += 1
        else:
            moved_map[fpath] = dest
            stats["moved"] += 1

    # Clean up empty dirs in _Inbox
    if EXECUTE:
        for dirpath, dirnames, filenames in os.walk(INBOX, topdown=False):
            dp = Path(dirpath)
            if dp == INBOX:
                continue
            if not any(dp.iterdir()):
                dp.rmdir()
                print(f"  Removed empty: {dp.relative_to(ROOT)}")


def main():
    if not INBOX.exists():
        print(f"ERROR: _Inbox not found at {INBOX}")
        sys.exit(1)

    # Count files
    file_count = sum(1 for f in INBOX.rglob("*") if f.is_file() and f.suffix.lower() in ALL_EXTS)
    if file_count == 0:
        print("_Inbox is empty — nothing to process.")
        sys.exit(0)

    print()
    print(f"  _Inbox contains {file_count} files")
    print()

    # Step 1: STAG+ scoring
    if not SKIP_STAG and EXECUTE:
        run_stag()
        print()

    # Step 2: Sort into date folders
    sort_inbox()

    # Summary
    print()
    print("=" * 60)
    verb = "Moved" if EXECUTE else "Would move"
    print(f"  {verb}:           {stats['moved']}")
    print(f"  Skipped (dupes):  {stats['skipped_dupe']}")
    print(f"  Errors:           {stats['errors']}")
    print("=" * 60)

    if not EXECUTE:
        print()
        print("  DRY RUN — no files were moved.")
        print("  Run with --execute to process for real.")
    else:
        print()
        print("  Done! Import new date folders in darktable.")


if __name__ == "__main__":
    main()
