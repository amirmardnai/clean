#!/usr/bin/env python3
"""
Stage 2 — OCR every page and emit an editable translation worksheet.

Uses apple-vision (Apple Vision framework OCR) which is available on this
device and handles comic lettering far better than tesseract.

Vision returns bboxes as [x, y, w, h] normalized with the ORIGIN AT
BOTTOM-LEFT. This script converts them to top-left pixel rectangles so
downstream rendering is straightforward.

Outputs <outdir>/worksheet.json:
  {"pages":[{"page":1,"file":"page-0001.png","width":W,"height":H,
             "items":[{"id":"p1-i1","box":[x,y,w,h],"en":"...","fa":"",
                       "kind":"dialogue"}]}]}

Fill in every "fa" field with the Persian translation, then run
render_pages.py. Leave "fa" empty to keep a region untouched.

Usage:
  python3 ocr_pages.py OUTDIR [--min-conf 0.35] [--merge-gap 0.02]
                              [--lang en-US] [--level accurate]
"""
import argparse
import json
import os
import subprocess
import sys

# Lines matching these are scanlation-group watermarks / site credits, not
# story text. They get tagged kind="credit" so you can skip translating them.
CREDIT_HINTS = (
    "scans", "scan", ".net", ".com", ".org", "www.", "http",
    "translator", "translation", "proofread", "cleaner", "redraw",
    "typesetter", "raw provider", "join us", "discord", "patreon",
    "read this work", "our website", "motivate us", "follow us",
    "team", "uploaded", "credit",
)


def looks_like_credit(text):
    low = text.lower()
    return any(h in low for h in CREDIT_HINTS)


def ocr_one(path, lang, level):
    cmd = ["apple-vision", "ocr", path, "-q", "--level", level]
    if lang:
        cmd += ["--lang", lang]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  ! OCR failed for {os.path.basename(path)}: {p.stderr[:200]}",
              file=sys.stderr)
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        print(f"  ! unparsable OCR output for {os.path.basename(path)}",
              file=sys.stderr)
        return None


def to_pixel_box(bbox, w, h):
    """Vision [x,y,w,h] bottom-left normalized -> [x,y,w,h] top-left pixels."""
    bx, by, bw, bh = bbox
    return [
        int(round(bx * w)),
        int(round((1.0 - (by + bh)) * h)),
        int(round(bw * w)),
        int(round(bh * h)),
    ]


def merge_lines(lines, page_h, gap_ratio):
    """
    Group OCR lines that belong to the same speech bubble.

    Two consecutive lines merge when they overlap horizontally and the
    vertical gap between them is smaller than gap_ratio * page_height.
    Comic bubbles are tightly leaded, so this is reliable in practice.
    """
    lines = sorted(lines, key=lambda l: (l["box"][1], l["box"][0]))
    groups = []
    for ln in lines:
        placed = False
        for g in groups:
            gx, gy, gw, gh = g["box"]
            lx, ly, lw, lh = ln["box"]
            h_overlap = min(gx + gw, lx + lw) - max(gx, lx)
            narrower = min(gw, lw)
            v_gap = ly - (gy + gh)
            if h_overlap > 0.25 * narrower and -0.5 * gh <= v_gap <= gap_ratio * page_h:
                nx, ny = min(gx, lx), min(gy, ly)
                nx2, ny2 = max(gx + gw, lx + lw), max(gy + gh, ly + lh)
                g["box"] = [nx, ny, nx2 - nx, ny2 - ny]
                g["lines"].append(ln)
                placed = True
                break
        if not placed:
            groups.append({"box": list(ln["box"]), "lines": [ln]})
    for g in groups:
        g["lines"].sort(key=lambda l: (l["box"][1], l["box"][0]))
    groups.sort(key=lambda g: (g["box"][1], g["box"][0]))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--min-conf", type=float, default=0.35)
    ap.add_argument("--merge-gap", type=float, default=0.02,
                    help="max vertical gap between lines of one bubble, as a "
                         "fraction of page height")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--level", default="accurate", choices=["fast", "accurate"])
    args = ap.parse_args()

    mpath = os.path.join(args.outdir, "manifest.json")
    if not os.path.exists(mpath):
        sys.exit("manifest.json missing — run extract_pages.py first")
    with open(mpath) as f:
        manifest = json.load(f)

    pages_dir = os.path.join(args.outdir, "pages")
    out_pages = []
    total = 0

    for idx, fn in enumerate(manifest["pages"], 1):
        path = os.path.join(pages_dir, fn)
        res = ocr_one(path, args.lang, args.level)
        if res is None:
            out_pages.append({"page": idx, "file": fn, "width": 0,
                              "height": 0, "items": [], "ocr_failed": True})
            continue

        w = res["image"]["width"]
        h = res["image"]["height"]
        lines = []
        for b in res.get("blocks", []):
            text = (b.get("text") or "").strip()
            if not text:
                continue
            if b.get("confidence", 0) < args.min_conf:
                continue
            lines.append({"box": to_pixel_box(b["bbox"], w, h),
                          "text": text,
                          "conf": round(b.get("confidence", 0), 3)})

        items = []
        for gi, g in enumerate(merge_lines(lines, h, args.merge_gap), 1):
            joined = " ".join(l["text"] for l in g["lines"])
            items.append({
                "id": f"p{idx}-i{gi}",
                "box": g["box"],
                "en": joined,
                "fa": "",
                "kind": "credit" if looks_like_credit(joined) else "dialogue",
                "conf": min(l["conf"] for l in g["lines"]),
                "line_count": len(g["lines"]),
            })

        total += sum(1 for i in items if i["kind"] == "dialogue")
        out_pages.append({"page": idx, "file": fn, "width": w, "height": h,
                          "items": items})
        print(f"  page {idx:>4}/{len(manifest['pages'])}: "
              f"{len(items)} regions")

    worksheet = {"source": manifest["source"], "page_count": len(out_pages),
                 "pages": out_pages}
    wpath = os.path.join(args.outdir, "worksheet.json")
    with open(wpath, "w") as f:
        json.dump(worksheet, f, ensure_ascii=False, indent=2)

    print(f"\nworksheet -> {wpath}")
    print(f"{total} dialogue regions to translate "
          f"(credit/watermark regions tagged separately)")


if __name__ == "__main__":
    main()
