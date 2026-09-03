#!/usr/bin/env python3
"""
Stage 5 — package the rendered pages back into a deliverable.

Produces a PDF (via img2pdf, lossless, one page per image) and/or a CBZ.

Usage:
  python3 assemble_output.py OUTDIR --title "My Title" [--pdf] [--cbz]
                             [--dest DIR] [--quality 85]
"""
import argparse
import os
import re
import subprocess
import sys
import zipfile


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def safe_name(s):
    return re.sub(r"[^\w\-. ]+", "_", s).strip() or "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--title", default="translated")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--cbz", action="store_true")
    ap.add_argument("--dest", default=None,
                    help="where to write deliverables (default: OUTDIR)")
    ap.add_argument("--quality", type=int, default=85,
                    help="JPEG quality when downsizing for the PDF; "
                         "0 = keep PNGs lossless")
    args = ap.parse_args()

    if not (args.pdf or args.cbz):
        args.pdf = True

    rendered = os.path.join(args.outdir, "rendered")
    if not os.path.isdir(rendered):
        sys.exit("rendered/ missing — run render_pages.py first")

    files = sorted(
        [f for f in os.listdir(rendered) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=natural_key,
    )
    if not files:
        sys.exit("no rendered pages found")

    dest = args.dest or args.outdir
    os.makedirs(dest, exist_ok=True)
    base = safe_name(args.title)
    paths = [os.path.join(rendered, f) for f in files]

    if args.pdf:
        pdf_inputs = paths
        if args.quality > 0:
            from PIL import Image
            tmp = os.path.join(args.outdir, "_pdf_jpg")
            os.makedirs(tmp, exist_ok=True)
            pdf_inputs = []
            for p in paths:
                out = os.path.join(tmp, os.path.splitext(os.path.basename(p))[0] + ".jpg")
                Image.open(p).convert("RGB").save(out, "JPEG",
                                                  quality=args.quality,
                                                  optimize=True)
                pdf_inputs.append(out)
        pdf_path = os.path.join(dest, base + ".pdf")
        r = subprocess.run(["img2pdf", "--output", pdf_path] + pdf_inputs,
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"img2pdf failed: {r.stderr[:1000]}")
        print(f"PDF  -> {pdf_path} ({os.path.getsize(pdf_path)/1e6:.1f} MB)")

    if args.cbz:
        cbz_path = os.path.join(dest, base + ".cbz")
        with zipfile.ZipFile(cbz_path, "w", zipfile.ZIP_DEFLATED) as z:
            for i, p in enumerate(paths, 1):
                z.write(p, f"{i:04d}{os.path.splitext(p)[1].lower()}")
        print(f"CBZ  -> {cbz_path} ({os.path.getsize(cbz_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
