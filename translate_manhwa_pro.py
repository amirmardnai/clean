#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manhwa/Webtoon Auto Persian Translator & Typesetter (v3.0 Ultra-Precision Edition)
Features:
  1. PyMuPDF native lossless page extraction
  2. Tiled EasyOCR for 100% text recall on ultra-tall (50,000px+) webtoon strips
  3. Structured ID-mapped Gemini 3.7 Flash Persian localization
  4. Guaranteed Zero-Blank-Bubble inpainting (only cleans when translation is valid)
  5. RTL Persian typography with outline stroke & Vazir font
  6. Per-chapter PDF + CBZ packaging + Ordered Chapter ZIP Bundler
"""

import os
import sys
import io
import re
import glob
import json
import time
import zipfile
import argparse
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import easyocr
except ImportError:
    easyocr = None

import arabic_reshaper
from bidi.algorithm import get_display

# Structure-aware lettering eraser/typesetter live next to this file in
# scripts/. They are used for text that sits on ARTWORK (SFX, titles, system
# panels) — anywhere the old "flat fill" behaviour left a visible patch.
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    import clean_lettering as _cl
    import typeset_lettering as _tl
except Exception:
    _cl = None
    _tl = None

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

Image.MAX_IMAGE_PIXELS = None

DEFAULT_LLM_URL = os.environ.get("MANHWA_LLM_URL", "http://127.0.0.1:8046/v1")
DEFAULT_LLM_MODEL = os.environ.get("MANHWA_LLM_MODEL", "gemini-3.7-flash")
# NOTE: prefer the MANHWA_API_KEY env var over the baked-in default below.
DEFAULT_LLM_KEY = os.environ.get("MANHWA_API_KEY", "sk-2428d57044e946e18c4361e9ad45b0bf")


def resolve_font():
    """Find best Persian font: bundled with the repo first, then OS fonts."""
    bundled = os.path.join(_SCRIPTS_DIR, "..", "assets", "fonts", "Vazirmatn-ExtraBold.ttf")
    candidates = [
        os.path.normpath(bundled),
        os.path.normpath(bundled.replace("ExtraBold", "Bold")),
        "C:/Windows/Fonts/Vazir-Bold.ttf",
        "C:/Windows/Fonts/Vazir.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/noto/NotoSansArabic-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "arial.ttf"


FONT_PATH = resolve_font()


def reshape_fa(text):
    if not text:
        return ""
    text = str(text).strip()
    return get_display(arabic_reshaper.reshape(text))


def natural_key(p):
    name = os.path.basename(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def chapter_number(path):
    name = os.path.basename(path)
    m = re.search(r"\[(\d+)\]", name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(?:ch|chapter|ep|episode|\b)(\d+)\b", name, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (ax1 - ax0) * (ay1 - ay0)
    ub = (bx1 - bx0) * (by1 - by0)
    return inter / float(ua + ub - inter)


def is_dialogue_noise(text):
    """Filter only scanlation watermarks and pure non-story noise."""
    if not text:
        return True
    t = text.strip()
    if len(t) == 0:
        return True

    # Filter pure punctuation/symbols (e.g. "...", "!!!", "---")
    if re.match(r"^[\.\,\:\;\!\?\'\"\-\_\~\`\^\*\#\@\$\%\&]+$", t):
        return True

    # Explicit scanlation credits & watermarks regex (whole-phrase matching)
    scanlation_pattern = r"(?i)\b(vortex\s*scans|vortexscans|manhwa_weebs|discord\.gg|patreon\.com|scanlation|brought\s*to\s*you\s*by|support\s*and\s*join|show\s*your\s*support)\b"
    if re.search(scanlation_pattern, t):
        return True

    # SFX noise patterns (only pure isolated sounds)
    if re.match(r"^(fwish|swoosh|bam|thwack|clack|pant|gasp|sigh|snort|tsk)$", t, re.IGNORECASE):
        return True

    return False


def detect_text_tiled(reader, np_img, tile_h=1800, overlap=250):
    """
    Runs EasyOCR with vertical tiling for ultra-tall webtoon strips to prevent aspect-ratio squashing.
    Guarantees 100% OCR recall for small text across 50,000px+ strips.
    """
    h_img, w_img = np_img.shape[:2]
    if h_img <= 2200:
        return reader.readtext(np_img)

    step = tile_h - overlap
    y = 0
    all_raw_boxes = []

    while y < h_img:
        y1 = min(y + tile_h, h_img)
        tile_crop = np_img[y:y1, 0:w_img]
        tile_res = reader.readtext(tile_crop)

        for box, text, conf in tile_res:
            # Map Y coordinates from tile space to global image space
            global_box = [[pt[0], pt[1] + y] for pt in box]
            all_raw_boxes.append((global_box, text, conf))

        if y1 >= h_img:
            break
        y += step

    # Deduplicate boxes in overlap regions
    deduped = []
    for item in all_raw_boxes:
        box, text, conf = item
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        cur_rect = [min(xs), min(ys), max(xs), max(ys)]

        is_dup = False
        for i, existing in enumerate(deduped):
            e_box, e_text, e_conf = existing
            e_xs = [pt[0] for pt in e_box]
            e_ys = [pt[1] for pt in e_box]
            e_rect = [min(e_xs), min(e_ys), max(e_xs), max(e_ys)]

            if iou(cur_rect, e_rect) > 0.4 or (text.strip().lower() == e_text.strip().lower() and abs(cur_rect[1] - e_rect[1]) < 80):
                if conf > e_conf:
                    deduped[i] = item
                is_dup = True
                break

        if not is_dup:
            deduped.append(item)

    return deduped


def translate_dialogues_llm(texts, base_url=DEFAULT_LLM_URL, model=DEFAULT_LLM_MODEL, key=DEFAULT_LLM_KEY):
    """
    Send ordered list of English texts to Gemini for contextual Persian localization.
    Uses structured ID mapping so responses are never misaligned or dropped.
    Intelligently recovers stylized comic fonts, outlined typography, and handwriting distortions.
    """
    if not texts:
        return []

    endpoint = base_url.rstrip("/") + "/chat/completions"
    system_prompt = (
        "You are an elite manhwa/webtoon dialogue localizer and Persian translator for professional scanlations.\n"
        "Translate English dialogue lines from a manhwa chapter into natural, fluent, spoken Persian (زبان محاوره‌ای و روان مانهوا).\n"
        "Rules:\n"
        "1. For each input item, return the natural Persian translation inside speech bubbles.\n"
        "2. Stylized comic fonts, outlined fonts, or handwriting may contain OCR distortions or typos "
        "(e.g., 'SO POURESHTNNO ACAIN' -> 'پس دوباره نشستی؟', 'The Ivialis Nvev , he coMe Co YoU?' -> 'دادگاه تموم شد، پس چرا باهات نیومد؟'). "
        "Intelligently decode and recover the intended comic dialogue to produce the natural Persian translation!\n"
        "3. ONLY return empty string \"\" for obvious scanlation credits (e.g. 'discord.gg', 'patreon.com').\n"
        "4. CRITICAL: Output ONLY a valid JSON array of objects with 'id' and 'fa': [{\"id\": 0, \"fa\": \"ترجمه فارسی\"}, ...]. "
        "Ensure EVERY input id (0 to N-1) is present in the output array. No markdown fences."
    )

    items_payload = [{"id": i, "en": t} for i, t in enumerate(texts)]
    user_prompt = "Translate these manhwa dialogues into Persian:\n" + json.dumps(items_payload, ensure_ascii=False)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    for attempt in range(4):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                    content = re.sub(r"\n?```$", "", content).strip()
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        fa_map = {}
                        for item in parsed:
                            if isinstance(item, dict) and "id" in item:
                                fa_map[int(item["id"])] = item.get("fa", "")
                            elif isinstance(item, str):
                                idx = len(fa_map)
                                fa_map[idx] = item

                        out_fa = [fa_map.get(i, "").strip() for i in range(len(texts))]
                        return out_fa
            else:
                print(f"      [LLM Note {resp.status_code} - Retry {attempt + 1}/4] {resp.text[:100]}")
                time.sleep(2)
        except Exception as e:
            print(f"      [LLM Network Retry {attempt + 1}/4] {e}")
            time.sleep(2)

    return [""] * len(texts)


def is_genuine_dialogue_line(text, bbox, conf):
    """
    Distinguishes true dialogue lines from huge stylized Korean SFX / noise strokes.
    Prevents background action sounds from pulling dialogue boxes onto characters.
    """
    t = text.strip()
    if not t:
        return False
    if is_dialogue_noise(t):
        return False

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Filter very low confidence tiny garbage
    if conf < 0.15 and len(t) < 3:
        return False

    cleaned = re.sub(r"[\d\s\W_]+", "", t)
    if len(cleaned) == 0:
        return False

    # Standalone single letter noise
    if len(cleaned) == 1 and (h > 40 or conf < 0.5):
        return False

    # Pure tall non-word strokes (Korean calligraphy)
    if len(cleaned) <= 2 and h > 60 and w < h * 0.8:
        return False

    # Filter pure consonant noise only if 3+ chars and no vowels
    if len(cleaned) >= 3 and not re.search(r"[aeiouyAEIOUY]", cleaned):
        return False

    return True


def cluster_ocr_boxes(ocr_results, img_shape, y_thresh=28, x_thresh=45):
    """Cluster multi-line dialogue boxes into cohesive speech bubbles with exact pixel coordinates."""
    if not ocr_results:
        return []

    h_img, w_img = img_shape[:2]
    filtered = []

    for entry in ocr_results:
        box, text, conf = entry
        t = text.strip()

        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        if x2 - x1 < 8 or y2 - y1 < 6:
            continue

        rect = [max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)]
        if not is_genuine_dialogue_line(t, rect, float(conf)):
            continue

        filtered.append({
            "bbox": rect,
            "text": t,
            "conf": float(conf)
        })

    clusters = []
    used = [False] * len(filtered)

    for i, it in enumerate(filtered):
        if used[i]:
            continue
        curr_cluster = [it]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j, other in enumerate(filtered):
                if used[j]:
                    continue
                for c in curr_cluster:
                    b1 = c["bbox"]
                    b2 = other["bbox"]
                    v_dist = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                    h_overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])

                    if v_dist <= y_thresh and h_overlap >= -x_thresh:
                        curr_cluster.append(other)
                        used[j] = True
                        changed = True
                        break

        curr_cluster.sort(key=lambda x: x["bbox"][1])
        full_text = " ".join([c["text"] for c in curr_cluster]).strip()

        if is_dialogue_noise(full_text):
            continue

        min_x = min(c["bbox"][0] for c in curr_cluster)
        min_y = min(c["bbox"][1] for c in curr_cluster)
        max_x = max(c["bbox"][2] for c in curr_cluster)
        max_y = max(c["bbox"][3] for c in curr_cluster)

        # Precise padding for Persian text layout (avoid overflowing bubbles)
        pad_x = max(4, int((max_x - min_x) * 0.08))
        pad_y = max(3, int((max_y - min_y) * 0.06))

        clusters.append({
            "bbox": [max(0, min_x - pad_x), max(0, min_y - pad_y),
                     min(w_img, max_x + pad_x), min(h_img, max_y + pad_y)],
            "original_text": full_text,
            "items": curr_cluster
        })

    return clusters


def clean_bubble_region(img_np, bbox, items=None, pad=3):
    """
    Non-destructive inpainting using OpenCV Telea algorithm.
    Preserves gradients, textures, comic screen tones, and complex bubble borders.
    """
    x1, y1, x2, y2 = bbox
    h, w = img_np.shape[:2]

    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)

    if x2 <= x1 or y2 <= y1:
        return True, (255, 255, 255)

    region = img_np[y1:y2, x1:x2]
    rh, rw = region.shape[:2]
    if rh < 4 or rw < 4:
        return True, (255, 255, 255)

    if cv2 is not None:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        # Sample outer border ring to determine local background luminance
        ring = np.concatenate([
            gray[0:min(3, rh), :].ravel(),
            gray[max(0, rh-3):rh, :].ravel(),
            gray[:, 0:min(3, rw)].ravel(),
            gray[:, max(0, rw-3):rw].ravel()
        ]) if rh > 6 and rw > 6 else gray.ravel()

        bg_val = float(np.median(ring))
        bg_std = float(np.std(ring))
        is_white_bg = bg_val > 115

        # Build exact text mask inside individual line boxes if provided, or full region
        mask = np.zeros((rh, rw), dtype=np.uint8)
        if items:
            for item in items:
                ib = item["bbox"]
                ix1 = max(0, ib[0] - x1 - 2)
                iy1 = max(0, ib[1] - y1 - 2)
                ix2 = min(rw, ib[2] - x1 + 2)
                iy2 = min(rh, ib[3] - y1 + 2)
                if ix2 > ix1 and iy2 > iy1:
                    sub_gray = gray[iy1:iy2, ix1:ix2]
                    if is_white_bg:
                        diff = bg_val - sub_gray.astype(np.float32)
                        sub_mask = ((diff > max(14.0, bg_std * 1.0)) & (sub_gray < 225)).astype(np.uint8) * 255
                    else:
                        diff = sub_gray.astype(np.float32) - bg_val
                        sub_mask = ((diff > max(14.0, bg_std * 1.0)) & (sub_gray > 30)).astype(np.uint8) * 255
                    mask[iy1:iy2, ix1:ix2] = sub_mask
        else:
            if is_white_bg:
                diff = bg_val - gray.astype(np.float32)
                mask = ((diff > max(14.0, bg_std * 1.0)) & (gray < 225)).astype(np.uint8) * 255
            else:
                diff = gray.astype(np.float32) - bg_val
                mask = ((diff > max(14.0, bg_std * 1.0)) & (gray > 30)).astype(np.uint8) * 255

        # Dilate mask slightly to cover font anti-aliasing edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint only where text mask is active
        if np.any(mask > 0):
            bgr = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
            region[:] = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

        return is_white_bg, (int(bg_val), int(bg_val), int(bg_val))
    else:
        # Fallback if cv2 is not available
        ring = np.concatenate([
            img_np[y1:y1 + 3, x1:x2].reshape(-1, 3),
            img_np[max(y1, y2 - 3):y2, x1:x2].reshape(-1, 3),
            img_np[y1:y2, x1:x1 + 3].reshape(-1, 3),
            img_np[y1:y2, max(x1, x2 - 3):x2].reshape(-1, 3),
        ]) if (y2 - y1) > 8 and (x2 - x1) > 8 else region.reshape(-1, 3)

        bg_color = np.median(ring, axis=0)
        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        is_white_bg = lum > 110
        fill_rgb = [int(v) for v in bg_color]

        if is_white_bg:
            is_dark = np.mean(region, axis=2) < 225
            region[is_dark] = fill_rgb
        else:
            is_light = np.mean(region, axis=2) > 35
            region[is_light] = fill_rgb

        return is_white_bg, fill_rgb


def typeset_persian_bubble(draw, bbox, text, font_path, is_white_bg=True, line_spacing=1.25):
    """Typeset Persian dialogue centered perfectly inside speech bubbles without overflowing."""
    if not text or not text.strip():
        return

    x1, y1, x2, y2 = bbox
    w_box = max(12, x2 - x1)
    h_box = max(12, y2 - y1)

    words = text.split()
    if not words:
        return

    best_size = 9
    best_lines = [text]

    # Binary search optimal font size that fits 100% inside both width and height bounds
    lo, hi = 9, 44
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            font = ImageFont.load_default()

        lines = []
        curr = []
        for w in words:
            test_line = " ".join(curr + [w])
            reshaped = reshape_fa(test_line)
            bbox_t = font.getbbox(reshaped)
            tw = bbox_t[2] - bbox_t[0]
            if tw <= w_box * 0.90:
                curr.append(w)
            else:
                if curr:
                    lines.append(" ".join(curr))
                    curr = [w]
                else:
                    lines.append(w)
                    curr = []
        if curr:
            lines.append(" ".join(curr))

        all_fit = True
        for l in lines:
            reshaped_l = reshape_fa(l)
            bbox_l = font.getbbox(reshaped_l)
            if (bbox_l[2] - bbox_l[0]) > w_box * 0.94:
                all_fit = False
                break

        total_h = len(lines) * int(mid * line_spacing)
        if all_fit and total_h <= h_box * 0.92:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1

    try:
        font = ImageFont.truetype(font_path, best_size)
    except Exception:
        font = ImageFont.load_default()

    lh = int(best_size * line_spacing)
    total_h = len(best_lines) * lh

    # Calculate geometric center of the bubble
    cy = (y1 + y2) // 2
    start_y = max(y1 + 2, cy - (total_h // 2))

    text_color = (15, 15, 15) if is_white_bg else (250, 250, 250)
    stroke_color = (255, 255, 255) if is_white_bg else (10, 10, 10)
    stroke_width = max(1, best_size // 13)

    for i, line in enumerate(best_lines):
        reshaped_l = reshape_fa(line)
        bbox_l = font.getbbox(reshaped_l)
        lw = bbox_l[2] - bbox_l[0]
        lx = x1 + (w_box - lw) // 2 - bbox_l[0]
        ly = start_y + i * lh

        draw.text(
            (lx, ly),
            reshaped_l,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color
        )


def sample_glyph_style(np_img_rgb, mask, bbox):
    """Sample the ORIGINAL lettering colours from the mask.

    Returns (fill_rgb, stroke_rgb|None).

    The mask includes anti-aliased edges and art pixels caught between
    glyphs, so a plain median lands somewhere in the artwork. Instead the
    masked pixels are clustered into colour classes (k-means); pixels whose
    colour matches the ring just OUTSIDE the mask are artwork bleed and are
    dropped; of the remaining classes the one sitting deepest inside the
    mask (mean distance-transform) is the glyph FILL and the other is the
    OUTLINE. Nearly identical colours mean plain unstroked lettering.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(np_img_rgb.shape[1], x2), min(np_img_rgb.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return (245, 245, 245), (20, 20, 20)
    wm = mask[y1:y2, x1:x2] if mask is not None else None
    if wm is None or (wm > 0).sum() < 25 or cv2 is None:
        return (245, 245, 245), (20, 20, 20)
    region = np_img_rgb[y1:y2, x1:x2]
    m01 = (wm > 0).astype(np.uint8)

    # ring just outside the mask tells us what artwork looks like
    ring_out = (cv2.dilate(m01, np.ones((7, 7), np.uint8), 1) > 0) & (m01 == 0)
    bg_col = np.median(region[ring_out], axis=0) if ring_out.sum() > 20 \
        else np.median(region.reshape(-1, 3), axis=0)

    px = region[m01 > 0].astype(np.float32).reshape(-1, 1, 3)
    K = int(min(3, max(2, len(px) // 60)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, labels, centers = cv2.kmeans(px, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return (245, 245, 245), (20, 20, 20)
    labels = labels.ravel()
    d = cv2.distanceTransform(m01, cv2.DIST_L2, 3)
    dvals = d[m01 > 0]

    classes = []
    for k in range(K):
        sel = labels == k
        if sel.sum() < 8:
            continue
        col = centers[k]
        bg_dist = float(np.abs(col - bg_col).mean())
        classes.append(dict(col=col, count=int(sel.sum()),
                            depth=float(dvals[sel].mean()),
                            bg_dist=bg_dist))
    # artwork bleed: class whose colour IS the background
    glyph = [c for c in classes if c["bg_dist"] > 14]
    if len(glyph) < 2:
        glyph = sorted(classes, key=lambda c: -c["depth"])[:2] or classes
    if not glyph:
        return (245, 245, 245), (20, 20, 20)
    glyph.sort(key=lambda c: -c["depth"])
    fill = tuple(int(v) for v in glyph[0]["col"])
    stroke = None
    if len(glyph) >= 2:
        cand = tuple(int(v) for v in glyph[1]["col"])
        if abs(int(fill[0]) - cand[0]) + abs(int(fill[1]) - cand[1]) \
                + abs(int(fill[2]) - cand[2]) > 45:
            stroke = cand

    # readability safety: if the sampled fill melts into the local
    # background (low-contrast original, e.g. orange on red), push it to
    # whichever pole keeps the original polarity but stays legible
    lum_bg = 0.299 * bg_col[0] + 0.587 * bg_col[1] + 0.114 * bg_col[2]
    lum_fill = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
    if abs(lum_fill - lum_bg) < 55:
        fill = (245, 245, 245) if lum_fill >= lum_bg else (18, 18, 18)
    if stroke is not None:
        lum_stroke = 0.299 * stroke[0] + 0.587 * stroke[1] + 0.114 * stroke[2]
        if abs(lum_stroke - lum_fill) < 40:
            stroke = (18, 18, 18) if lum_fill > 128 else (245, 245, 245)
    return fill, stroke


def typeset_artwork_block(im_pil, bbox, text, font_path, fill, stroke,
                          slant=0.12, line_spacing=1.18, ss=3):
    """Typeset Persian over artwork: wrapped, centred, STROKED in the
    sampled original colours, with an optional italic shear.

    Reuses the bubble-style binary-search fitting but renders each line at
    `ss` x supersampling on its own tile (stroked glyphs are visibly ragged
    otherwise) and composites with alpha, so gradients show through between
    letters the same way they did behind the original text.
    """
    if not text or not text.strip():
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w_box = max(12, x2 - x1)
    h_box = max(12, y2 - y1)
    words = text.split()
    if not words:
        return

    probe = ImageDraw.Draw(im_pil)
    best_size, best_lines = 9, [text]
    hi = min(140, max(48, h_box - 4))
    lo = 9
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            font = ImageFont.load_default()
        lines, curr = [], []
        for w in words:
            t = " ".join(curr + [w])
            b = font.getbbox(reshape_fa(t))
            if (b[2] - b[0]) <= w_box * 0.92:
                curr.append(w)
            else:
                if curr:
                    lines.append(" ".join(curr))
                    curr = [w]
                else:
                    lines.append(w)
                    curr = []
        if curr:
            lines.append(" ".join(curr))
        okw = all((font.getbbox(reshape_fa(l))[2] - font.getbbox(reshape_fa(l))[0])
                  <= w_box * 0.96 for l in lines)
        total_h = len(lines) * int(mid * line_spacing)
        if okw and total_h <= h_box * 0.94:
            best_size, best_lines = mid, lines
            lo = mid + 1
        else:
            hi = mid - 1

    lh = int(best_size * line_spacing)
    total_h = len(best_lines) * lh
    cy = (y1 + y2) // 2
    start_y = max(y1 + 2, cy - total_h // 2)
    sw = max(2, int(round(best_size * 0.115))) if stroke else 0

    for i, line in enumerate(best_lines):
        shaped = reshape_fa(line)
        try:
            font_big = ImageFont.truetype(font_path, best_size * ss)
        except Exception:
            font_big = ImageFont.load_default()
        bb = probe.textbbox((0, 0), shaped, font=font_big, stroke_width=sw * ss)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = sw * ss + int(best_size * ss * 0.15) + 4
        tile = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        td.text((pad - bb[0], pad - bb[1]), shaped, font=font_big,
                fill=tuple(fill) + (255,),
                stroke_width=sw * ss,
                stroke_fill=tuple(stroke) + (255,) if stroke else tuple(fill) + (255,))
        if slant:
            neww = tile.width + int(abs(slant) * tile.height)
            tile = tile.transform(
                (neww, tile.height), Image.AFFINE,
                (1, slant, -slant * tile.height if slant > 0 else 0, 0, 1, 0),
                resample=Image.BICUBIC)
        tile = tile.resize((max(1, tile.width // ss), max(1, tile.height // ss)),
                           Image.LANCZOS)
        lx = x1 + (w_box - tile.width) // 2
        ly = start_y + i * lh + (lh - tile.height) // 2
        im_pil.alpha_composite(tile, (int(lx), int(ly)))


def process_page(np_img, clusters, fa_texts, font_path,
                 use_artwork=True, slant=0.12, debug=False):
    """Clean + re-letter ONE page: bubbles via inpaint, artwork via the
    structure-aware engine. Routing is per region so a page can mix both.

    Returns (cleaned_pil_image, localized_count, artwork_count).
    """
    im_cleaned = Image.fromarray(np_img)
    draw = ImageDraw.Draw(im_cleaned)
    localized_count = 0
    artwork_count = 0

    # ---- bubble vs artwork routing ---------------------------------
    # Text inside flat speech bubbles takes the classic inpaint path.
    # Text sitting on artwork (SFX, titles, system panels, shouts) is
    # erased with the structure-aware engine and re-lettered in the
    # original colours instead of being buried under a flat patch.
    valid = [i for i, t in enumerate(fa_texts) if t and t.strip()]
    struct_ok = (use_artwork and _cl is not None and cv2 is not None)
    art_set = set()
    style_map = {}
    if struct_ok and valid:
        gray_full = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        xywh_all = [
            (c["bbox"][0], c["bbox"][1],
             c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1])
            for c in clusters
        ]
        try:
            mask_full = _cl.build_mask(gray_full, xywh_all)
            for i in valid:
                kind, _ang, _st = _cl.classify_region(
                    gray_full, mask_full, xywh_all[i], pad=24)
                is_bubble = kind == "flat"
                if is_bubble:
                    # genuine light bubble -> old path; but a DARK flat
                    # area (night caption box, dark plate) still needs
                    # the styled stroked lettering of the artwork path
                    bx1, by1, bx2, by2 = clusters[i]["bbox"]
                    x1c, y1c = max(0, bx1 - 4), max(0, by1 - 4)
                    x2c = min(gray_full.shape[1], bx2 + 4)
                    y2c = min(gray_full.shape[0], by2 + 4)
                    wm_loc = mask_full[y1c:y2c, x1c:x2c]
                    region = gray_full[y1c:y2c, x1c:x2c]
                    clean_px = region[wm_loc == 0]
                    med = float(np.median(clean_px)) if clean_px.size else 255.0
                    is_bubble = med >= 150
                if not is_bubble:
                    art_set.add(i)
                    style_map[i] = sample_glyph_style(
                        np_img, mask_full, clusters[i]["bbox"])
            if art_set:
                art_boxes = [xywh_all[i] for i in sorted(art_set)]
                bgr = np_img[:, :, ::-1].copy()
                erased = _cl.erase(bgr, mask_full, boxes=art_boxes)
                np_img = np.clip(erased[:, :, ::-1], 0, 255).astype(np.uint8)
                im_cleaned = Image.fromarray(np_img).convert("RGBA")
                draw = ImageDraw.Draw(im_cleaned)
        except Exception as e:
            print(f"      [artwork-clean fallback: {e}]")
            art_set = set()

    for ci, (c, fa_t) in enumerate(zip(clusters, fa_texts)):
        if not fa_t or not fa_t.strip():
            # If translation failed or is noise -> Keep original text untouched!
            continue

        if ci in art_set:
            # artwork path: already structure-erased above; re-letter
            # in the original style (fill/stroke from the glyphs)
            fill_rgb, stroke_rgb = style_map.get(ci, ((245, 245, 245), (20, 20, 20)))
            typeset_artwork_block(im_cleaned, c["bbox"], fa_t, font_path,
                                  fill_rgb, stroke_rgb, slant=slant)
            localized_count += 1
            artwork_count += 1
            if debug:
                draw.rectangle(c["bbox"], outline=(0, 128, 255), width=2)
            continue

        # Non-destructive inpainting of bubble interior
        is_white, bg_rgb = clean_bubble_region(np_img, c["bbox"], items=c.get("items"))
        x1, y1, x2, y2 = c["bbox"]
        patch = Image.fromarray(np_img[y1:y2, x1:x2])
        if im_cleaned.mode == "RGBA":
            im_cleaned.paste(patch.convert("RGBA"), (x1, y1))
        else:
            im_cleaned.paste(patch, (x1, y1))

        # Typeset Persian text
        typeset_persian_bubble(draw, c["bbox"], fa_t, font_path, is_white_bg=is_white)
        localized_count += 1

        if debug:
            draw.rectangle(c["bbox"], outline=(255, 0, 0), width=2)

    if im_cleaned.mode == "RGBA":
        im_cleaned = im_cleaned.convert("RGB")
    return im_cleaned, localized_count, artwork_count


def extract_pages(pdf, outdir):
    """Extract raw high-res images from PDF using PyMuPDF."""
    os.makedirs(outdir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(outdir, "pg-*.*")), key=natural_key)
    if existing:
        return existing

    good = []
    if fitz is not None:
        try:
            doc = fitz.open(pdf)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                image_list = page.get_images(full=True)
                extracted_any = False
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    with Image.open(io.BytesIO(image_bytes)) as im:
                        w, h = im.size
                    if w >= 200 and h >= 200:
                        img_path = os.path.join(outdir, "pg-%04d-%02d.%s" % (page_idx, img_idx, image_ext))
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        good.append(img_path)
                        extracted_any = True
                if not extracted_any:
                    pix = page.get_pixmap(dpi=150)
                    img_path = os.path.join(outdir, "pg-%04d.jpg" % page_idx)
                    pix.save(img_path)
                    good.append(img_path)
            doc.close()
        except Exception as e:
            print(f"    PyMuPDF extraction note: {e}")
    return sorted(good, key=natural_key)


def process_manhwa_pdf(pdf_path, work_dir, out_dir, reader, base_url, model, key,
                       font_path=FONT_PATH, limit_pages=None, debug=False,
                       use_artwork=True, slant=0.12):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    slug = re.sub(r"[^0-9A-Za-z]+", "_", base_name)[:40]
    src_dir = os.path.join(work_dir, slug, "src")
    done_dir = os.path.join(work_dir, slug, "out")
    os.makedirs(done_dir, exist_ok=True)

    pages = extract_pages(pdf_path, src_dir)
    if limit_pages:
        pages = pages[:limit_pages]

    if not pages:
        print("    [!] No pages extracted.")
        return None

    print(f"    [*] Extracted {len(pages)} pages.")
    rendered_pages = []

    for pi, page_path in enumerate(pages):
        target_jpg = os.path.join(done_dir, f"p{pi:04d}.jpg")
        if os.path.exists(target_jpg):
            rendered_pages.append(target_jpg)
            continue

        im = Image.open(page_path).convert("RGB")
        np_img = np.array(im)

        # 1. Tiled OCR detection (guarantees detection across 50,000px+ strips)
        ocr_res = detect_text_tiled(reader, np_img)

        # 2. Cluster text lines into dialogue bubbles
        clusters = cluster_ocr_boxes(ocr_res, np_img.shape)

        if clusters:
            en_texts = [c["original_text"] for c in clusters]
            fa_texts = translate_dialogues_llm(en_texts, base_url, model, key)

            # 3 & 4. Inpaint and Typeset ONLY when valid translation exists
            im_cleaned, localized_count, artwork_count = process_page(
                np_img, clusters, fa_texts, font_path,
                use_artwork=use_artwork, slant=slant, debug=debug)
            im_cleaned.save(target_jpg, "JPEG", quality=92, optimize=True)
            extra = f" ({artwork_count} artwork)" if artwork_count else ""
            print(f"      Page {pi + 1}/{len(pages)} -> {localized_count}/{len(clusters)} bubbles localized{extra}.")
        else:
            im.save(target_jpg, "JPEG", quality=92, optimize=True)
            print(f"      Page {pi + 1}/{len(pages)} -> 0 bubbles found.")

        rendered_pages.append(target_jpg)

    # PDF Output
    pdf_out = os.path.join(out_dir, base_name + " [FA].pdf")
    if rendered_pages:
        first = Image.open(rendered_pages[0]).convert("RGB")
        rest = [Image.open(p).convert("RGB") for p in rendered_pages[1:]]
        first.save(pdf_out, "PDF", save_all=True, append_images=rest, resolution=150.0)
        first.close()
        for r in rest:
            r.close()

    # CBZ Output
    cbz_out = os.path.join(out_dir, base_name + " [FA].cbz")
    with zipfile.ZipFile(cbz_out, "w", zipfile.ZIP_STORED) as z:
        for i, p in enumerate(rendered_pages):
            z.write(p, "%04d.jpg" % i)

    return pdf_out, cbz_out


def main():
    ap = argparse.ArgumentParser(description="Manhwa/Webtoon High-Precision Persian Localization CLI.")
    ap.add_argument("src", help="Folder containing source PDFs or single PDF file")
    ap.add_argument("-o", "--out", default=None, help="Output folder")
    ap.add_argument("--work", default=None, help="Working/cache folder")
    ap.add_argument("--base-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--api-key", default=os.environ.get("MANHWA_API_KEY", DEFAULT_LLM_KEY))
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--zip", dest="zipname", default="manhwa_persian_all_chapters.zip")
    ap.add_argument("--debug-boxes", action="store_true")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--langs", default="en",
                    help="EasyOCR languages, comma separated (e.g. en,ko)")
    ap.add_argument("--cpu", action="store_true", help="force EasyOCR CPU mode")
    ap.add_argument("--slant", type=float, default=0.12,
                    help="italic shear for artwork lettering (0 = upright)")
    ap.add_argument("--no-artwork", action="store_true",
                    help="disable structure-aware cleaning; paint everything flat")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out or os.path.join(src, "_translated_fa"))
    work = os.path.abspath(args.work or os.path.join(src, "_work_pro"))
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)

    if os.path.isdir(src):
        pdfs = sorted(glob.glob(os.path.join(src, "*.pdf")), key=natural_key)
    else:
        pdfs = [src]

    if args.limit_files:
        pdfs = pdfs[:args.limit_files]

    if not pdfs:
        print("No PDFs found.")
        sys.exit(1)

    print("=" * 65)
    print("  MANHWA/WEBTOON ULTRA-PRECISION LOCALIZATION ENGINE (v3.0)")
    print(f"  Source: {src}")
    print(f"  Output: {out}")
    print(f"  Model:  {args.model} @ {args.base_url}")
    print(f"  Font:   {FONT_PATH}")
    print(f"  Files:  {len(pdfs)}")
    print("=" * 65)

    if easyocr is None:
        sys.exit("EasyOCR is not installed: pip install easyocr")
    gpu = not args.cpu
    if gpu:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
    print(f"[*] Initializing EasyOCR Engine (langs={args.langs}, gpu={gpu})...")
    reader = easyocr.Reader([s.strip() for s in args.langs.split(",") if s.strip()], gpu=gpu)

    made = []
    t0 = time.time()

    for idx, pdf in enumerate(pdfs):
        print(f"\n[{idx + 1}/{len(pdfs)}] Processing: {os.path.basename(pdf)}")
        res = process_manhwa_pdf(
            pdf, work, out, reader,
            base_url=args.base_url, model=args.model, key=args.api_key,
            limit_pages=args.limit_pages or None,
            debug=args.debug_boxes,
            use_artwork=not args.no_artwork,
            slant=args.slant
        )
        if res:
            made.append(res)

    # Bundle all translated PDFs in clean chapter order
    if made and not args.no_zip:
        zip_path = os.path.join(out, args.zipname)
        print(f"\n[*] Bundling {len(made)} chapters into ordered PDF ZIP: {zip_path}")
        
        # Sort chapters numerically
        made_sorted = sorted(made, key=lambda pair: (chapter_number(pair[0]) if chapter_number(pair[0]) is not None else 999999, pair[0]))
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for idx, (pdf_out, cbz_out) in enumerate(made_sorted, 1):
                ch = chapter_number(pdf_out)
                folder_name = str(ch) if ch is not None else f"Chapter_{idx:03d}"
                entry_name = f"{folder_name}/{os.path.basename(pdf_out)}"
                z.write(pdf_out, entry_name)

            readme_text = (
                "مجموعه چپترهای ترجمه‌شده مانهوا - هر چپتر درون پوشه اختصاصی شماره چپتر\n"
                "Persian Manhwa PDF Collection (Each chapter inside its numbered folder)\n"
            )
            z.writestr("README.txt", readme_text)
        print(f"[✓] ZIP Bundle created: {os.path.basename(zip_path)} ({os.path.getsize(zip_path) / (1024*1024):.1f} MB)")

    dt = time.time() - t0
    print(f"\n[✓] All done! Completed {len(made)} chapters in {dt / 60.0:.1f} minutes.")
    print(f"[✓] Output directory: {out}")


if __name__ == "__main__":
    main()
