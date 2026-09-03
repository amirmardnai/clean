#!/usr/bin/env python3
"""
Stage 4 — mask the original lettering and typeset the Persian text.

For each worksheet item that has a non-empty "fa" value:
  1. sample the colour around the text box and flood the box with it
     (speech bubbles are flat white/near-white, so this reads as a clean erase)
  2. shape the Persian text (arabic_reshaper) and apply the bidi algorithm
  3. word-wrap and auto-size the font so the text fills the box
  4. draw it centred, top-to-bottom, right-to-left

Items with kind == "credit" are skipped unless --include-credits is passed.

Usage:
  python3 render_pages.py OUTDIR [--font PATH] [--pad 6] [--min-size 11]
                                 [--max-size 90] [--grow 1.15]
                                 [--include-credits] [--debug-boxes]
"""
import argparse
import json
import os
import sys
from collections import Counter

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    sys.exit("missing deps: pip install --break-system-packages "
             "arabic-reshaper python-bidi")

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FONT = os.path.join(HERE, "..", "assets", "fonts", "Vazirmatn-Bold.ttf")

_reshaper = arabic_reshaper.ArabicReshaper(
    arabic_reshaper.config_for_true_type_render
) if hasattr(arabic_reshaper, "config_for_true_type_render") else None


def shape(text):
    """Logical Persian string -> visually ordered, contextually joined string."""
    reshaped = (_reshaper.reshape(text) if _reshaper
                else arabic_reshaper.reshape(text))
    return get_display(reshaped)


def sample_fill(img, box, ring=4):
    """
    Most common colour in a thin ring just outside the text box.
    That is the bubble interior, which is what the text should be erased to.
    """
    x, y, w, h = box
    W, H = img.size
    x0, y0 = max(0, x - ring), max(0, y - ring)
    x1, y1 = min(W, x + w + ring), min(H, y + h + ring)
    if x1 <= x0 or y1 <= y0:
        return (255, 255, 255)
    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
    px = crop.load()
    cw, ch = crop.size
    votes = Counter()
    for i in range(cw):
        for j in range(min(ring, ch)):
            votes[px[i, j]] += 1
            votes[px[i, ch - 1 - j]] += 1
    for j in range(ch):
        for i in range(min(ring, cw)):
            votes[px[i, j]] += 1
            votes[px[cw - 1 - i, j]] += 1
    if not votes:
        return (255, 255, 255)
    # quantise slightly so near-identical scan noise collapses to one vote
    coarse = Counter()
    for (r, g, b), n in votes.items():
        coarse[(r // 8 * 8, g // 8 * 8, b // 8 * 8)] += n
    r, g, b = coarse.most_common(1)[0][0]
    return (min(255, r + 4), min(255, g + 4), min(255, b + 4))


def text_colour(fill):
    """Black on light backgrounds, white on dark ones."""
    lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    return (20, 20, 20) if lum > 128 else (240, 240, 240)


def wrap_to_width(words, font, max_w, draw):
    """Greedy wrap on logical words, measured in shaped form."""
    lines, cur = [], []
    for word in words:
        trial = cur + [word]
        w = draw.textlength(shape(" ".join(trial)), font=font)
        if w <= max_w or not cur:
            cur = trial
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def fit_text(draw, text, box, font_path, pad, min_size, max_size, grow):
    """
    Find the largest font size at which the wrapped text fits the box.
    If even min_size does not fit, the box is allowed to grow by `grow`.
    Returns (font, lines, line_h, box).
    """
    words = text.split()
    for attempt in range(2):
        x, y, w, h = box
        avail_w = max(8, w - 2 * pad)
        avail_h = max(8, h - 2 * pad)
        lo, hi, best = min_size, max_size, None
        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(font_path, mid)
            lines = wrap_to_width(words, font, avail_w, draw)
            asc, desc = font.getmetrics()
            line_h = int((asc + desc) * 1.16)
            if line_h * len(lines) <= avail_h:
                best = (font, lines, line_h)
                lo = mid + 1
            else:
                hi = mid - 1
        if best:
            return best[0], best[1], best[2], box
        if attempt == 0 and grow > 1.0:
            nw, nh = int(w * grow), int(h * grow)
            box = [x - (nw - w) // 2, y - (nh - h) // 2, nw, nh]
    font = ImageFont.truetype(font_path, min_size)
    lines = wrap_to_width(words, font, max(8, box[2] - 2 * pad), draw)
    asc, desc = font.getmetrics()
    return font, lines, int((asc + desc) * 1.16), box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--font", default=os.path.normpath(DEFAULT_FONT))
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--min-size", type=int, default=11)
    ap.add_argument("--max-size", type=int, default=90)
    ap.add_argument("--grow", type=float, default=1.15,
                    help="how much a box may expand when text will not fit")
    ap.add_argument("--include-credits", action="store_true")
    ap.add_argument("--debug-boxes", action="store_true",
                    help="outline every replaced region, for QA passes")
    args = ap.parse_args()

    if not os.path.exists(args.font):
        sys.exit(f"font not found: {args.font}")

    wpath = os.path.join(args.outdir, "worksheet.json")
    if not os.path.exists(wpath):
        sys.exit("worksheet.json missing — run ocr_pages.py first")
    with open(wpath) as f:
        ws = json.load(f)

    pages_dir = os.path.join(args.outdir, "pages")
    out_dir = os.path.join(args.outdir, "rendered")
    os.makedirs(out_dir, exist_ok=True)

    done = skipped = 0
    for page in ws["pages"]:
        src = os.path.join(pages_dir, page["file"])
        if not os.path.exists(src):
            print(f"  ! missing {page['file']}", file=sys.stderr)
            continue
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)

        for item in page.get("items", []):
            fa = (item.get("fa") or "").strip()
            if not fa:
                skipped += 1
                continue
            if item.get("kind") == "credit" and not args.include_credits:
                skipped += 1
                continue

            fill = sample_fill(img, item["box"])
            font, lines, line_h, box = fit_text(
                draw, fa, list(item["box"]), args.font, args.pad,
                args.min_size, args.max_size, args.grow)

            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], fill=fill)
            colour = text_colour(fill)
            block_h = line_h * len(lines)
            cy = y + max(0, (h - block_h) // 2)
            for line in lines:
                vis = shape(line)
                lw = draw.textlength(vis, font=font)
                draw.text((x + (w - lw) / 2, cy), vis, font=font, fill=colour)
                cy += line_h
            if args.debug_boxes:
                draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
            done += 1

        out = os.path.join(out_dir, os.path.splitext(page["file"])[0] + ".png")
        img.save(out)

    print(f"rendered {ws['page_count']} pages -> {out_dir}")
    print(f"{done} regions typeset, {skipped} left untouched")


if __name__ == "__main__":
    main()
