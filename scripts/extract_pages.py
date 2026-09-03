#!/usr/bin/env python3
"""
Stage 1 — extract page images from a comic source.

Supports: .pdf (via pdftoppm), .cbz/.zip, .cbr/.cb7/.7z (via 7z),
and plain directories of images.

Output: <outdir>/pages/page-0001.png ... in exact reading order,
plus <outdir>/manifest.json describing the source and page list.

Usage:
  python3 extract_pages.py SOURCE OUTDIR [--dpi 200] [--first N] [--last N]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def natural_key(s):
    """Sort '10' after '9' — critical for preserving page order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{p.stderr[:2000]}")
    return p.stdout


def from_pdf(src, pages_dir, dpi, first, last):
    cmd = ["pdftoppm", "-r", str(dpi), "-png"]
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    cmd += [src, os.path.join(pages_dir, "raw")]
    run(cmd)


def from_archive(src, pages_dir):
    tmp = os.path.join(pages_dir, "_unpack")
    os.makedirs(tmp, exist_ok=True)
    ext = os.path.splitext(src)[1].lower()
    if ext in (".cbz", ".zip") and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
    else:
        run(["7z", "x", "-y", f"-o{tmp}", src])
    found = []
    for root, _, files in os.walk(tmp):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                found.append(os.path.join(root, fn))
    found.sort(key=lambda p: natural_key(os.path.relpath(p, tmp)))
    for i, path in enumerate(found, 1):
        dst = os.path.join(pages_dir, f"raw-{i:04d}{os.path.splitext(path)[1].lower()}")
        shutil.move(path, dst)
    shutil.rmtree(tmp, ignore_errors=True)


def from_dir(src, pages_dir):
    found = [
        os.path.join(src, fn)
        for fn in os.listdir(src)
        if os.path.splitext(fn)[1].lower() in IMG_EXT
    ]
    found.sort(key=lambda p: natural_key(os.path.basename(p)))
    for i, path in enumerate(found, 1):
        shutil.copy(path, os.path.join(pages_dir, f"raw-{i:04d}{os.path.splitext(path)[1].lower()}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("outdir")
    ap.add_argument("--dpi", type=int, default=200,
                    help="render resolution for PDF sources (200 is a good OCR/quality balance)")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=0)
    args = ap.parse_args()

    src = os.path.abspath(args.source)
    if not os.path.exists(src):
        sys.exit(f"source not found: {src}")

    pages_dir = os.path.join(args.outdir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    for fn in os.listdir(pages_dir):
        os.remove(os.path.join(pages_dir, fn))

    ext = os.path.splitext(src)[1].lower()
    if os.path.isdir(src):
        kind = "dir"
        from_dir(src, pages_dir)
    elif ext == ".pdf":
        kind = "pdf"
        from_pdf(src, pages_dir, args.dpi, args.first, args.last)
    else:
        kind = "archive"
        from_archive(src, pages_dir)

    raw = sorted(
        [f for f in os.listdir(pages_dir) if f.startswith("raw")],
        key=natural_key,
    )
    if not raw:
        sys.exit("no page images produced")

    pages = []
    for i, fn in enumerate(raw, 1):
        new = f"page-{i:04d}{os.path.splitext(fn)[1].lower()}"
        os.rename(os.path.join(pages_dir, fn), os.path.join(pages_dir, new))
        pages.append(new)

    manifest = {
        "source": src,
        "source_kind": kind,
        "dpi": args.dpi if kind == "pdf" else None,
        "page_count": len(pages),
        "pages": pages,
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"extracted {len(pages)} pages -> {pages_dir}")


if __name__ == "__main__":
    main()
