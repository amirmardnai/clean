#!/usr/bin/env python3
"""
Comic-style lettering typesetter.

`render_pages.py` handles the ordinary case: text inside a speech bubble, so
plain dark text on the bubble fill is right. This script handles the other
case — free-floating lettering over artwork (shouts, narration, SFX), which
needs a stroked outline to stay readable, and usually leans italic.

Differences that matter for matching the original lettering:
  * one line per original OCR box, so each translated line lands where its
    source line was instead of being re-wrapped into a block;
  * stroke width proportional to font size, which is how letterers scale it;
  * italic applied as a shear on a rendered tile, so any font can lean
    without needing an italic cut;
  * rendered at 4x and downsampled, because a stroked glyph at final size
    has visibly ragged edges otherwise.

Usage
-----
    python3 typeset_lettering.py IN OUT --line 'x,y,w,h:متن' [--line ...]
                                        [--font PATH] [--fill 000000]
                                        [--stroke ffffff] [--stroke-ratio 0.115]
                                        [--slant 0.16] [--supersample 4]

Colours are hex RGB. Pass --stroke none to draw without an outline.
"""
from __future__ import annotations

import argparse
import sys

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    sys.exit("missing deps: pip install --break-system-packages "
             "arabic-reshaper python-bidi")

from PIL import Image, ImageDraw, ImageFont

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FONT = os.path.normpath(
    os.path.join(HERE, "..", "assets", "fonts", "Vazirmatn-ExtraBold.ttf"))

_reshaper = (arabic_reshaper.ArabicReshaper(arabic_reshaper.config_for_true_type_render)
             if hasattr(arabic_reshaper, "config_for_true_type_render") else None)


def shape(text):
    """Logical string -> visually ordered, contextually joined string."""
    reshaped = (_reshaper.reshape(text) if _reshaper
                else arabic_reshaper.reshape(text))
    return get_display(reshaped)


def hex_rgb(s):
    if s is None or s.lower() == "none":
        return None
    s = s.lstrip("#")
    if len(s) != 6:
        raise argparse.ArgumentTypeError(f"want RRGGBB, got {s!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def fit_size(draw, shaped, box_w, box_h, font_path, max_size=600):
    """Largest font size whose single line still fits the box."""
    size = 8
    while size < max_size:
        f = ImageFont.truetype(font_path, size + 1)
        asc, desc = f.getmetrics()
        if draw.textlength(shaped, font=f) > box_w * 0.97 or (asc + desc) > box_h * 1.02:
            break
        size += 1
    return size


def render_line(text, box, font_path, fill, stroke, stroke_ratio, slant, ss,
                probe_draw):
    """Return (tile RGBA, paste_xy) for one line, already sheared."""
    l, t, w, h = box
    bw, bh = w * ss, h * ss
    shaped = shape(text)

    size = fit_size(probe_draw, shaped, bw, bh, font_path)
    font = ImageFont.truetype(font_path, size)
    asc, desc = font.getmetrics()
    tw = probe_draw.textlength(shaped, font=font)

    sw = max(2, int(round(size * stroke_ratio))) if stroke else 0
    pad = int(size * 0.45) + sw
    tile = Image.new("RGBA", (int(tw) + pad * 2, asc + desc + pad * 2), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    if stroke:
        td.text((pad, pad), shaped, font=font, fill=fill + (255,),
                stroke_width=sw, stroke_fill=stroke + (255,))
    else:
        td.text((pad, pad), shaped, font=font, fill=fill + (255,))

    if slant:
        neww = tile.width + int(abs(slant) * tile.height)
        tile = tile.transform(
            (neww, tile.height), Image.AFFINE,
            (1, slant, -slant * tile.height if slant > 0 else 0, 0, 1, 0),
            resample=Image.BICUBIC)

    x = l * ss + (bw - tile.width) // 2
    y = t * ss + (bh - tile.height) // 2
    return tile, (int(x), int(y)), size, sw


def parse_line(spec):
    """'x,y,w,h:text' -> ((x,y,w,h), text)"""
    geom, sep, text = spec.partition(":")
    if not sep:
        sys.exit(f"bad --line {spec!r}, want 'x,y,w,h:text'")
    nums = [int(round(float(v))) for v in geom.split(",")]
    if len(nums) != 4:
        sys.exit(f"bad box in {spec!r}, want x,y,w,h")
    return tuple(nums), text


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--line", action="append", required=True,
                    help="'x,y,w,h:text' — repeat once per original text line")
    ap.add_argument("--font", default=DEFAULT_FONT)
    ap.add_argument("--fill", type=hex_rgb, default=(0, 0, 0))
    ap.add_argument("--stroke", type=hex_rgb, default=(255, 255, 255),
                    help="outline colour, or 'none'")
    ap.add_argument("--stroke-ratio", type=float, default=0.115,
                    help="outline width as a fraction of font size")
    ap.add_argument("--slant", type=float, default=0.16,
                    help="italic shear; 0 for upright")
    ap.add_argument("--supersample", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.font):
        sys.exit(f"font not found: {args.font}")

    base = Image.open(args.src).convert("RGB")
    W, H = base.size
    ss = max(1, args.supersample)

    layer = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    probe = ImageDraw.Draw(layer)

    for spec in args.line:
        box, text = parse_line(spec)
        if not text.strip():
            continue
        tile, xy, size, sw = render_line(
            text, box, args.font, args.fill, args.stroke, args.stroke_ratio,
            args.slant, ss, probe)
        layer.alpha_composite(tile, xy)
        print(f"  {text[:28]!r:32s} box={box} size={size} stroke={sw}")

    if ss > 1:
        layer = layer.resize((W, H), Image.LANCZOS)
    out = base.convert("RGBA")
    out.alpha_composite(layer)
    out.convert("RGB").save(args.dst)
    print(f"{args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
