#!/usr/bin/env python3
"""
Structure-aware lettering eraser (v2 — self-verifying).

The naive way to erase comic lettering is to flood the text box with the
surrounding colour. That is correct for flat speech bubbles and wrong for
everything else: text sitting on artwork, speed lines, gradients, halftone
or texture turns into a visible grey patch.

This module chooses a strategy per region, automatically, and then VERIFIES
the result against the surrounding artwork; if the fill is measurably poor
it retries with a different strategy instead of shipping a bad patch. The
output therefore no longer depends on luck:

  flat      -> nearly uniform interior (a normal speech bubble).
               Flood with the sampled colour + matched grain.
  periodic  -> background repeats on a lattice (halftone dots, patterns).
               Copy pixels shifted by the detected period, then seam-correct.
  striped   -> background has a constant direction AND roughly constant
               colour along it (speed lines, motion blur, gradients).
               Transplant real pixels along the streak direction.
  blocky    -> hard-edged shapes (armor plates, panel borders). Direction
               exists but colour is not constant along it. Propagate the
               nearest colour class into the hole (approximate Voronoi),
               which extends interrupted shape boundaries plausibly.
  fallback   -> busy, directionless artwork. Two-pass isotropic inpaint.

Every fill is finished with grain re-synthesis: real scanned/printed pages
have sensor noise and film grain; a statistically smooth patch betrays
itself even when its macrostructure is right. The grain parameters are
measured on the clean ring around the mask.

After the first erase, a leftover pass re-scans the original window for
text-like components that escaped the mask (e.g. a bright core whose
outline alone was caught) and re-erases once. This fixes the classic
"ghost letters" failure without ever touching artwork far from the text.

Public API
----------
    build_mask(gray, boxes, ...)          -> uint8 mask of glyphs + outline
    erase(bgr, mask, boxes=None, ...)     -> cleaned BGR float32
    erase_auto(bgr, boxes, ...)           -> convenience: mask + erase
    fill_quality(orig, out, mask, boxes)  -> objective metrics per region

CLI
---
    python3 clean_lettering.py IN OUT --boxes 'x,y,w,h;x,y,w,h' [--report]
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------- mask building


def _glyph_components(binary, hh, ww):
    """Keep glyph-shaped components; score how text-like they are."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    core = np.zeros_like(binary)
    heights, mids = [], []
    for i in range(1, n):
        _, y_, w_, h_, area = stats[i]
        if area < 16:
            continue
        if h_ > 0.9 * hh or w_ > 0.85 * ww:
            continue              # page-spanning blob, not a glyph
        if w_ * h_ and area / float(w_ * h_) < 0.06:
            continue              # hairline streak, not a glyph
        core[lab == i] = 1
        heights.append(h_)
        mids.append(y_ + h_ / 2.0)
    if len(heights) < 1:
        return core, 0.0
    frac = float(core.mean())
    if frac > 0.6:
        return core, 0.0          # flooded the window, not lettering
    med_h = float(np.median(heights))
    size_score = min(med_h / (0.3 * hh), 1.0)
    if len(heights) >= 2:
        consistency = 1.0 - min(float(np.std(heights)) / (med_h + 1e-6), 1.0)
        baseline = 1.0 - min(float(np.std(mids)) / (0.5 * hh + 1e-6), 1.0)
    else:
        consistency = baseline = 0.5
    score = len(heights) * size_score * (0.5 + 0.5 * consistency) \
        * (0.5 + 0.5 * baseline)
    return core, float(score)


def _stroke_width_estimate(sub, core):
    """How far the opposite-polarity outline extends around the cores.

    Estimates from the image itself instead of a fixed pixel count, so it
    adapts to 60-px SFX outlines and thin caption lettering alike.
    """
    if not core.any():
        return 0
    med = float(np.median(sub))
    light_core = float(np.median(sub[core > 0])) >= med
    dist = cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 3)
    max_d = int(min(40, dist.max()))
    if max_d < 2:
        return 2
    prev_frac = 1.0
    for d in range(2, max_d + 1):
        ring = (dist > d - 1) & (dist <= d)
        vals = sub[ring]
        if len(vals) < 40:
            break
        if light_core:
            frac = float((vals < med - 12).mean())   # dark outline fraction
        else:
            frac = float((vals > med + 12).mean())   # light outline fraction
        if frac < 0.18 and prev_frac < 0.25:
            return d
        prev_frac = frac
    return min(max_d, 6)


def build_mask(gray, boxes, pad=20, dark_max=78, stroke_px=7, halo_px=11,
               halo_min=None, close=15, dilate=5):
    """Mask the lettering inside each box: glyph cores plus their outline.

    Comic lettering is almost always a dark core with a light stroke, the
    inverse, or a gradient core with either. Both polarity candidates are
    scored on how text-like they look; when both look like text (gradient
    SFX), their union is used. Thresholds are RELATIVE to the window's own
    brightness distribution, so low-contrast lettering (orange on red) is
    found just like high-contrast lettering is.

    The stroke is then taken geometrically, as everything within an
    ESTIMATED outline width of a core — far more reliable than global
    thresholds over unknown artwork.
    """
    H, W = gray.shape[:2]
    mask = np.zeros((H, W), np.uint8)

    for (bx, by, bw, bh) in boxes:
        L, T = max(0, int(bx) - pad), max(0, int(by) - pad)
        R, B = min(W, int(bx + bw) + pad), min(H, int(by + bh) + pad)
        if R - L < 4 or B - T < 4:
            continue
        sub = gray[T:B, L:R]
        hh, ww = sub.shape

        med = float(np.median(sub))
        p10, p90 = np.percentile(sub, (10, 90))
        spread = float(p90 - p10)
        dev = max(14.0, 0.45 * spread)

        cand_dark, score_dark = _glyph_components((sub < med - dev).astype(np.uint8), hh, ww)
        cand_light, score_light = _glyph_components((sub > med + dev).astype(np.uint8), hh, ww)

        # union when both polarities look textual (gradient cores, split fill)
        if score_dark > 0 and score_light > 0 and \
                min(score_dark, score_light) > 0.7 * max(score_dark, score_light):
            overlap = cv2.dilate(cand_dark, np.ones((9, 9), np.uint8)).astype(bool) & cand_light.astype(bool)
            core = (cand_dark | cand_light).astype(np.uint8) if overlap.any() else \
                (cand_dark if score_dark >= score_light else cand_light)
        else:
            core = cand_dark if score_dark >= score_light else cand_light
        if core is None or not core.any():
            continue

        # ---- grow the seeds outward at a WEAKER threshold ----------------------
        # Lettering frequently splits across thresholds: a dark outline gets
        # caught decisively while a low-contrast core (orange on red) scores
        # like artwork. Weak components are adopted ONLY when they touch the
        # accepted mask and stay glyph-sized — artwork larger than ~2.5x the
        # median glyph can never be pulled in, however close it sits.
        dev_weak = max(10.0, 0.26 * spread)
        weak = ((sub > med + dev_weak) | (sub < med - dev_weak)).astype(np.uint8)
        n0, _, stats0, _ = cv2.connectedComponentsWithStats(core, 8)
        if n0 > 1:
            med_area = float(np.median([stats0[i, 4] for i in range(1, n0)]))
            max_add = max(2.5 * med_area, 120.0)
        else:
            med_area = float(core.sum())
            max_add = 3.0 * med_area
        for _ in range(3):
            nbr = cv2.dilate(core, np.ones((5, 5), np.uint8))
            cand_add = (weak & (nbr > 0) & (core == 0)).astype(np.uint8)
            if not cand_add.any():
                break
            na, lab, stats_a, _ = cv2.connectedComponentsWithStats(cand_add, 8)
            changed = False
            for i in range(1, na):
                x_, y_, w_, h_, area = stats_a[i]
                if area < 10 or area > max_add:
                    continue
                if w_ > 0.9 * ww or h_ > 0.9 * hh:
                    continue
                core[lab == i] = 1
                changed = True
            if not changed:
                break
        if core.mean() > 0.65:      # safety: never flood the window
            core = cand_dark if score_dark >= score_light else cand_light

        # outline ring, geometric, width estimated from the page itself
        ring_w = max(stroke_px, _stroke_width_estimate(sub, core))
        dist = cv2.distanceTransform((1 - core).astype(np.uint8), cv2.DIST_L2, 3)
        ring = (dist <= ring_w).astype(np.uint8)
        # plus pixels that want to be halo: near the core, deviating from the
        # window's own mid-brightness (RELATIVE thresholds, not global)
        if halo_min is None:
            med_core = float(np.median(sub[core > 0]))
            light_core = med_core >= med
            halo_dev = max(16.0, 0.35 * spread)
            if light_core:
                bled = ((dist <= halo_px) & (sub < med_core - halo_dev)).astype(np.uint8)
            else:
                bled = ((dist <= halo_px) & (sub > med_core + halo_dev)).astype(np.uint8)
        else:
            bled = ((dist <= halo_px) & (sub > halo_min)).astype(np.uint8)

        m = ((core | ring | bled) > 0).astype(np.uint8)
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
        m = cv2.dilate(
            m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)), 1)
        mask[T:B, L:R] = np.maximum(mask[T:B, L:R], m * 255)

    return mask


# ------------------------------------------------------------ region classifier


def _clean_probe(wm, sub_shape, pad=24):
    """Clean pixels in a ring hugging the mask (the fill's source material)."""
    wm01 = (wm > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2)
    near = cv2.dilate(wm01, k, 1) > 0
    safe = (wm01 == 0) & ~cv2.dilate(wm01, np.ones((5, 5), np.uint8), 1).astype(bool)
    probe = near & safe
    if probe.sum() < 200:
        probe = safe
    return probe


def dominant_angle(sub, clean, lo=-89.0, hi=90.0, coarse=4.0, fine=0.5):
    """Angle along which the image varies least, plus a coherence score."""
    h, w = sub.shape
    ok = clean.astype(np.uint8)
    inner = np.zeros((h, w), bool)
    m = max(6, int(min(h, w) * 0.08))
    inner[m:h - m, m:w - m] = True
    if inner.sum() < 200:
        inner[:] = True

    def score(angle):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        rs = cv2.warpAffine(sub, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        rk = cv2.warpAffine(ok, M, (w, h), flags=cv2.INTER_NEAREST,
                            borderValue=0) > 0
        sel = rk & inner
        if sel.sum() < 150:
            return None
        gx = cv2.Sobel(rs, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(rs, cv2.CV_32F, 0, 1, ksize=3)
        ex = float((gx[sel] ** 2).mean())
        ey = float((gy[sel] ** 2).mean())
        return ex / (ey + 1e-9)

    best = None
    for a in np.arange(lo, hi, coarse):
        s = score(a)
        if s is not None and (best is None or s < best[1]):
            best = (float(a), s)
    if best is None:
        return 0.0, 0.0
    for a in np.arange(best[0] - coarse, best[0] + coarse + 1e-9, fine):
        s = score(a)
        if s is not None and s < best[1]:
            best = (float(a), s)
    coherence = float(np.clip(1.0 - best[1], 0.0, 1.0))
    return best[0], coherence


def _along_direction_constancy(sub, clean, angle):
    """Is the COLOUR constant along `angle`? (stripes: yes, armor plates: no)

    Rotates the window so `angle` lies horizontal, then compares the median
    within-row std of clean pixels to the global std. A ratio well below 1
    means each row is nearly one colour -> safe to transplant along it.
    """
    h, w = sub.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
    rs = cv2.warpAffine(sub.astype(np.float32), M, (w, h),
                        borderMode=cv2.BORDER_REPLICATE)
    rk = cv2.warpAffine(clean.astype(np.uint8), M, (w, h),
                        flags=cv2.INTER_NEAREST, borderValue=0) > 0
    row_stds = []
    for y in range(h):
        vals = rs[y][rk[y]]
        if len(vals) >= 40:
            row_stds.append(float(np.std(vals)))
    if len(row_stds) < 5:
        return 1.0
    glob = float(np.std(sub[clean])) if clean.sum() > 100 else float(np.std(sub))
    if glob < 1e-6:
        return 1.0
    return float(np.median(row_stds)) / (glob + 1e-6)


def _lattice_period(sub, clean, max_period=64, min_period=4):
    """Dominant repetition period of the clean background (FFT autocorrelation).

    Returns (period_xy, strength). strength ~1: perfectly periodic (halftone);
    ~0: aperiodic. Only peaks inside [min_period, max_period] are trusted —
    larger ones are just slow gradients pretending to be patterns.
    """
    h, w = sub.shape
    vals = sub.astype(np.float32)
    if clean.sum() < 400:
        return None, 0.0
    mu = float(vals[clean].mean())
    v = np.where(clean, vals - mu, 0.0)
    f = np.fft.rfft2(v)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=v.shape)
    norm = ac[0, 0]
    if norm <= 1e-9:
        return None, 0.0
    ac = ac / norm
    # ignore the central blob (self-correlation at offsets < min_period)
    r = int(min_period) - 1
    if r >= 0:
        ac[:r + 1, :r + 1] = 0
        ac[:r + 1, -r:] = 0
        ac[-r:, :r + 1] = 0
        ac[-r:, -r:] = 0
    yy, xx = np.unravel_index(np.argmax(ac), ac.shape)
    peak = float(ac[yy, xx])
    # sub-pixel peak by 3-point parabola per axis (wrap-safe indices)
    def _sub(v0, v1, v2):
        denom = (v0 - 2 * v1 + v2)
        return 0.5 * (v0 - v2) / denom if abs(denom) > 1e-12 else 0.0
    fy = yy + _sub(ac[(yy - 1) % h, xx], peak, ac[(yy + 1) % h, xx])
    fx = xx + _sub(ac[yy, (xx - 1) % w], peak, ac[yy, (xx + 1) % w])
    dy = fy if fy <= h / 2 else fy - h
    dx = fx if fx <= w / 2 else fx - w
    period = float(np.hypot(dx, dy))
    if not (min_period <= period <= max_period) or peak < 0.35:
        return None, peak
    return (float(dx), float(dy)), peak


def classify_region(gray, mask, box, pad=24):
    """Decide how to fill one region.

    Returns (kind, angle, stats) with kind one of
    'flat', 'periodic', 'striped', 'blocky', 'fallback'.
    """
    H, W = gray.shape[:2]
    bx, by, bw, bh = box
    L, T = max(0, int(bx) - pad), max(0, int(by) - pad)
    R, B = min(W, int(bx + bw) + pad), min(H, int(by + bh) + pad)
    sub = gray[T:B, L:R].astype(np.float32)
    wm = mask[T:B, L:R]
    clean = wm == 0
    if clean.sum() < 200:
        return "fallback", 0.0, {}

    probe = _clean_probe(wm, sub.shape, pad=pad)
    vals = sub[probe]
    if len(vals) < 50:
        return "fallback", 0.0, {}
    spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))

    gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    energy = float(np.percentile(mag[probe], 75))

    if spread < 26 and energy < 14:
        return "flat", 0.0, dict(spread=spread, energy=energy)

    period, p_strength = _lattice_period(sub, clean)
    angle, coh = dominant_angle(sub, clean)
    constancy = _along_direction_constancy(sub, clean, angle)

    stats = dict(spread=spread, energy=energy, coherence=coh,
                 constancy=round(constancy, 2), lattice=round(p_strength, 2))
    # a strong 1D direction means stripes/fibres -> directional transplant is
    # the natural fill; period copying is reserved for true 2D lattices
    if period is not None and p_strength >= 0.5 and coh < 0.6:
        return "periodic", angle, stats
    if coh >= 0.45 and constancy <= 0.7:
        return "striped", angle, stats
    if energy >= 14 and constancy > 0.7:
        return "blocky", angle, stats
    if coh >= 0.35:
        return "striped", angle, stats
    return "fallback", angle, stats


# ------------------------------------------------------------------ fill pieces


def directional_exemplar(bgr, mask, angle_deg, max_walk=None, step=1.0):
    """Fill the mask by copying real pixels along `angle_deg` (both ways)."""
    src = bgr.astype(np.float32)
    H, W = mask.shape[:2]
    hole = mask > 0
    if not hole.any():
        return src
    valid = ~hole
    if max_walk is None:
        max_walk = int(np.hypot(H, W) * 0.6)

    th = np.radians(angle_deg)
    dx, dy = float(np.cos(th)), float(np.sin(th))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    picks = []
    for sgn in (+1.0, -1.0):
        found = np.zeros((H, W), bool)
        dist = np.full((H, W), np.inf, np.float32)
        sx = np.zeros((H, W), np.float32)
        sy = np.zeros((H, W), np.float32)
        k = step
        while k <= max_walk:
            px = xx + sgn * dx * k
            py = yy + sgn * dy * k
            inb = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
            vi = np.clip(np.rint(px), 0, W - 1).astype(np.int32)
            vj = np.clip(np.rint(py), 0, H - 1).astype(np.int32)
            hit = inb & valid[vj, vi] & hole & (~found)
            if hit.any():
                found |= hit
                dist[hit] = k
                sx[hit] = px[hit]
                sy[hit] = py[hit]
            if found[hole].all():
                break
            k += step
        vals = cv2.remap(src, np.where(found, sx, xx), np.where(found, sy, yy),
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
        picks.append((found, dist, vals))

    (f1, d1, v1), (f2, d2, v2) = picks
    w1 = np.where(f1, 1.0 / (d1 + 1e-3), 0.0)
    w2 = np.where(f2, 1.0 / (d2 + 1e-3), 0.0)
    tot = w1 + w2
    out = src.copy()
    reached = hole & (tot > 0)
    blend = (v1 * w1[:, :, None] + v2 * w2[:, :, None]) / np.maximum(tot, 1e-9)[:, :, None]
    out[reached] = blend[reached]

    missed = hole & ~reached
    if missed.any():
        iso = cv2.inpaint(bgr, missed.astype(np.uint8), 5, cv2.INPAINT_NS)
        out[missed] = iso.astype(np.float32)[missed]
    return out


def _shift(arr, oy, ox, fill=None):
    """Shift an array by (oy, ox) WITHOUT wraparound.

    shifted[p] == arr[p - (oy, ox)]; out-of-range sources read `fill`
    (or False/0). np.roll would smuggle the opposite edge of the window into
    the fill, planting seamless-looking foreign artifacts.
    """
    H, W = arr.shape[:2]
    if fill is None:
        out = np.zeros_like(arr)
    else:
        out = np.full_like(arr, fill)
    y0, y1 = max(0, oy), min(H, H + oy)
    x0, x1 = max(0, ox), min(W, W + ox)
    sy0, sy1 = max(0, -oy), min(H, H - oy)
    sx0, sx1 = max(0, -ox), min(W, W - ox)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = arr[sy0:sy1, sx0:sx1]
    return out


def periodic_fill(win, wm, period_xy, waves=6):
    """Fill a hole on a periodic background by walking to REAL clean pixels.

    Each hole pixel is filled from the first genuinely clean pixel found by
    walking along the lattice vector in both directions (inverse-distance
    blend of the two hits) — so copies always carry one period of real
    artwork, never the accumulated rounding drift of chained copies.

    Implemented as the same machinery as the directional transplant with
    the step set to the lattice period (after sub-pixel refinement, bilinear
    reads land exactly on the true lattice even mid-hole).
    """
    dx, dy = float(period_xy[0]), float(period_xy[1])
    step = float(np.hypot(dx, dy))
    if step < 2:
        return None
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return directional_exemplar(win, wm, angle, max_walk=None, step=step)


def class_propagate_fill(win, wm, n_classes=6):
    """Fill hard-edged artwork by nearest colour-class propagation.

    Clean pixels are quantised into a few dominant colours; every hole pixel
    takes the colour of the NEAREST clean pixel's class. Interrupted plate
    boundaries therefore extend into the hole roughly straight, which is
    exactly how blocky mecha/panel artwork looks — a transplant along the
    wrong direction smears them instead.
    """
    hole = wm > 0
    if not hole.any():
        return win.astype(np.float32)
    src = win.astype(np.float32)
    clean_px = src[~hole]
    if len(clean_px) < 100:
        return cv2.inpaint(win, (hole).astype(np.uint8), 5,
                           cv2.INPAINT_TELEA).astype(np.float32)
    Z = clean_px.reshape((-1, 1, 3)).astype(np.float32)
    K = int(min(n_classes, max(2, len(clean_px) // 40)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.5)
    try:
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3,
                                        cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return cv2.inpaint(win, hole.astype(np.uint8), 5,
                           cv2.INPAINT_TELEA).astype(np.float32)
    centers = centers.astype(np.float32)

    class_img = np.full(wm.shape[:2], -1, np.int32)
    class_img[~hole] = labels.ravel()
    # coordinates of the nearest clean pixel for every hole pixel:
    # distanceTransformWithLabels(DIST_LABEL_PIXEL) packs them as (x<<16)|y+1
    _, nearest = cv2.distanceTransformWithLabels(
        hole.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(hole)
    if len(xs) == 0:
        return src
    # label encodes source pixel: (x << 16) | (y + 1); 0 means none found
    lab = nearest[ys, xs]
    sx = (lab >> 16) - 1
    sy = (lab & 0xFFFF) - 1
    good = (sx >= 0) & (sy >= 0) & (sx < wm.shape[1]) & (sy < wm.shape[0])
    out = src.copy()
    chosen = np.full(len(xs), -1, np.int32)
    chosen[good] = class_img[sy[good], sx[good]]
    fallback_cls = int(np.bincount(labels.ravel()).argmax())
    chosen[chosen < 0] = fallback_cls
    out[ys, xs] = centers[chosen]
    return out


def _pull_push(values, known, levels=7):
    """Smoothly interpolate sparse `values` over the whole frame (pyramid)."""
    if values.ndim == 2:
        values = values[:, :, None]
    C = values.shape[2]
    w0 = known.astype(np.float32)
    vp, wp = [values * w0[:, :, None]], [w0]
    while len(vp) < levels and min(vp[-1].shape[:2]) >= 4:
        vp.append(cv2.pyrDown(vp[-1]).reshape(
            (vp[-1].shape[0] + 1) // 2, (vp[-1].shape[1] + 1) // 2, C))
        wp.append(cv2.pyrDown(wp[-1]))

    cur = vp[-1] / np.maximum(wp[-1], 1e-6)[:, :, None]
    for lv in range(len(vp) - 2, -1, -1):
        h, w = vp[lv].shape[:2]
        up = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[:, :, None]
        conf = np.clip(wp[lv], 0.0, 1.0)[:, :, None]
        fine = vp[lv] / np.maximum(wp[lv], 1e-6)[:, :, None]
        cur = fine * conf + up * (1.0 - conf)
    return cur


def harmonic_correct(orig, guide, mask, ring_px=7, levels=7, smooth=2.5):
    """Snap the guide's low-frequency brightness to the original boundary."""
    o = orig.astype(np.float32)
    g = guide.astype(np.float32)
    hole = mask > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1,) * 2)
    ring = (cv2.dilate(hole.astype(np.uint8), k, 1) > 0) & (~hole)
    if ring.sum() < 20:
        return np.where(hole[:, :, None], g, o)
    resid = np.zeros_like(o)
    resid[ring] = (o - g)[ring]
    corr = _pull_push(resid, ring, levels=levels)
    if smooth > 0:
        corr = cv2.GaussianBlur(corr, (0, 0), smooth)
    return np.where(hole[:, :, None], g + corr, o)


def _grain_stats(win_bgr, wm):
    """Noise sigma per channel, measured on the clean ring around the mask."""
    hole = wm > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = (cv2.dilate(hole.astype(np.uint8), k, 1) > 0) & (~hole)
    if ring.sum() < 300:
        ring = ~hole
    sigmas = []
    for c in range(3):
        ch = win_bgr[:, :, c].astype(np.float32)
        resid = ch - cv2.GaussianBlur(ch, (0, 0), 1.4)
        vals = resid[ring]
        if len(vals) < 100:
            sigmas.append(0.0)
            continue
        mad = np.median(np.abs(vals - np.median(vals)))
        sigmas.append(float(1.4826 * mad))
    return sigmas


def _add_grain(filled, wm, sigmas, rng=None):
    """Sprinkle statistically matched grain into the filled area.

    A perfectly smooth patch on a noisy page is what betrays an erase more
    than any color mismatch — this restores the page's own grain signature.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    if max(sigmas) < 1.2:
        return filled
    hole = wm > 0
    if not hole.any():
        return filled
    alpha = cv2.erode(hole.astype(np.float32), np.ones((5, 5), np.float32), 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    out = filled.copy()
    for c in range(3):
        if sigmas[c] < 1.0:
            continue
        n = rng.normal(0.0, sigmas[c], wm.shape).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 0.7) / 0.996 if False else n
        out[:, :, c] = out[:, :, c] + n * alpha
    return out


def _feather(orig, filled, mask, radius=0.8):
    m = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), radius)
    m = np.clip(m, 0, 1)[:, :, None]
    return orig.astype(np.float32) * (1 - m) + filled.astype(np.float32) * m


def _flat_fill(bgr, mask, box, pad=6):
    """Flood a flat bubble with its own interior colour."""
    H, W = mask.shape[:2]
    bx, by, bw, bh = box
    L, T = max(0, int(bx) - pad), max(0, int(by) - pad)
    R, B = min(W, int(bx + bw) + pad), min(H, int(by + bh) + pad)
    sub = bgr[T:B, L:R]
    wm = mask[T:B, L:R] > 0
    clean = ~wm
    out = bgr.astype(np.float32).copy()
    if clean.sum() < 20:
        return out
    med = np.median(sub[clean].reshape(-1, 3), axis=0)
    patch = out[T:B, L:R]
    patch[wm] = med
    out[T:B, L:R] = patch
    return out


def _inpaint_fill(win, wm):
    iso = cv2.inpaint(win, (wm > 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    iso = cv2.inpaint(iso, cv2.dilate((wm > 0).astype(np.uint8),
                                      np.ones((3, 3), np.uint8)),
                      5, cv2.INPAINT_NS)
    return iso.astype(np.float32)


# --------------------------------------------------------------------- scoring


def _window_quality(orig_win, out_win, wm):
    """grad/detail ratios of the fill vs its ring (within one work window)."""
    if (wm > 0).sum() < 100:
        return dict(grad=1.0, detail=1.0)
    g_out = cv2.cvtColor(np.clip(out_win, 0, 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
    hole = wm > 0
    inner = cv2.erode(hole.astype(np.uint8), np.ones((7, 7), np.uint8), 1) > 0
    ring = (cv2.dilate(hole.astype(np.uint8), np.ones((25, 25), np.uint8), 1) > 0) & (~hole)
    if inner.sum() < 100 or ring.sum() < 100:
        return dict(grad=1.0, detail=1.0)
    gx = cv2.Sobel(g_out, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_out, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    hf = g_out - cv2.GaussianBlur(g_out, (0, 0), 1.5)
    return dict(
        grad=float(mag[inner].mean()) / (float(mag[ring].mean()) + 1e-9),
        detail=float(hf[inner].std()) / (float(hf[ring].std()) + 1e-9),
    )


def _boundary_ratio(orig_win, filled_win, wm):
    """Did the fill create edges that ARTWORK would never have?

    Measures gradient energy in a band crossing the mask boundary (where a
    bad fill leaves seams) against the same band shifted into clean artwork
    (what the page itself looks like). Ratio ~1: seamless. >>1: the fill
    drew its own border — smears, ghost halos, fake-pattern discontinuities.
    This is what gradient/detail ratios inside the mask cannot see.
    """
    hole = (wm > 0).astype(np.uint8)
    if hole.sum() < 80:
        return 1.0
    band_in = cv2.erode(hole, np.ones((3, 3), np.uint8), 1).astype(bool) & hole.astype(bool)
    band_out = cv2.dilate(hole, np.ones((7, 7), np.uint8), 2).astype(bool) & ~hole.astype(bool)
    band = band_in | band_out
    g = cv2.cvtColor(np.clip(filled_win, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if band.sum() < 100:
        return 1.0
    j_fill = float(np.percentile(mag[band], 72))
    go = cv2.cvtColor(orig_win, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gxo = cv2.Sobel(go, cv2.CV_32F, 1, 0, ksize=3)
    gyo = cv2.Sobel(go, cv2.CV_32F, 0, 1, ksize=3)
    mago = np.sqrt(gxo * gxo + gyo * gyo)
    refs = []
    H, W = wm.shape[:2]
    sy = max(6, int(H * 0.33))
    sx = max(6, int(W * 0.33))
    clean = ~hole.astype(bool)
    for (dy, dx) in ((sy, 0), (-sy, 0), (0, sx), (0, -sx)):
        shifted = np.roll(band, (dy, dx), (0, 1)) & clean
        if shifted.sum() >= 150:
            refs.append(float(np.percentile(mago[shifted], 72)))
    if not refs:
        return 1.0
    j_art = float(np.median(refs))
    return j_fill / (j_art + 6.0)


def _score(q):
    lo = min(q["grad"], q["detail"])
    hi = max(q["grad"], q["detail"])
    return 0.65 * min(lo, 1.2) + 0.35 * min(hi, 1.2)


def _luma_consistency(orig_win, filled_win, wm):
    """Low-frequency fidelity: does the fill land the same BRIGHTNESS the
    artwork implies?

    The expected smooth field is interpolated (pull-push) from the clean
    ring; the fill's blurred version is compared against it. Dot lattices
    and fibers average out at blur scale and pass; a repeating-pattern copy
    of a stray cloud or a dragged colour slab does not. Ratio ~0.1 good,
    >0.17 means the patch would read as a smudge of the wrong tone.
    """
    hole = wm > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ring = (cv2.dilate(hole.astype(np.uint8), k, 1) > 0) & (~hole)
    inner = cv2.erode(hole.astype(np.uint8), np.ones((5, 5), np.uint8), 1) > 0
    if ring.sum() < 100 or inner.sum() < 60:
        return 0.0
    ests = []
    for c in range(3):
        ch = orig_win[:, :, c].astype(np.float32)
        vals = np.zeros_like(ch)
        vals[ring] = ch[ring]
        est = _pull_push(vals, ring, levels=6)[:, :, 0]
        ests.append(est)
    est = np.stack(ests, axis=2)
    blur = cv2.GaussianBlur(filled_win.astype(np.float32), (0, 0), 5)
    err = float(np.abs(blur - est)[inner].mean())
    spread = float((np.percentile(orig_win[ring], 90, axis=0)
                    - np.percentile(orig_win[ring], 10, axis=0)).mean())
    return err / (spread + 8.0)


def _spike_ratio(orig_win, filled_win, wm):
    """Are there edge spikes INSIDE the fill that the art itself never makes?

    Zone seams, hatch artifacts and cross-shape copies show up as p95
    gradient magnitudes far above the surroundings' p95 — while mean-based
    grad ratios look fine. Halftone/lines keep p95 inside ~= ring.
    """
    hole = wm > 0
    inner = cv2.erode(hole.astype(np.uint8), np.ones((7, 7), np.uint8), 1) > 0
    ring = (cv2.dilate(hole.astype(np.uint8), np.ones((25, 25), np.uint8), 1) > 0) & (~hole)
    if inner.sum() < 100 or ring.sum() < 100:
        return 1.0
    g = cv2.cvtColor(np.clip(filled_win, 0, 255).astype(np.uint8),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag[inner], 95)
                 / (np.percentile(mag[ring], 95) + 1.0))


_LUMA_REJECT = 0.17
_BOUNDARY_REJECT = 1.7
_SPIKE_REJECT = 2.4


def _combined_score(q, boundary_ratio, luma_err=0.0, spike_ratio=1.0):
    """One number per fill; -1 means the artifact detectors vetoed it."""
    if luma_err > _LUMA_REJECT or boundary_ratio > _BOUNDARY_REJECT \
            or spike_ratio > _SPIKE_REJECT:
        return -1.0
    s = _score(q)
    s -= 0.4 * max(0.0, max(q["grad"], q["detail"]) - 1.3)   # over-sharp suspicion
    return s / (1.0 + 0.5 * max(0.0, boundary_ratio - 1.3))


_GOOD_ENOUGH = 0.82


# ------------------------------------------------------------------ strategies


def _fill_with(win, wm, strategy, angle, period_xy, flat_box, rng):
    """Run one fill strategy; returns float32 window."""
    if strategy == "flat":
        mask_full = wm
        med = np.median(win[~ (mask_full > 0)].reshape(-1, 3), axis=0) \
            if (~ (mask_full > 0)).sum() > 20 else np.array([255, 255, 255], np.float32)
        out = win.astype(np.float32).copy()
        out[mask_full > 0] = med
        filled = out
    elif strategy == "periodic" and period_xy is not None:
        filled = periodic_fill(win, wm, period_xy)
        if filled is None:
            return None
        filled = harmonic_correct(win, np.clip(filled, 0, 255).astype(np.uint8)
                                  .astype(np.float32), wm)
    elif strategy == "striped":
        guide = directional_exemplar(win, wm, angle)
        guide = np.clip(guide, 0, 255).astype(np.uint8)
        filled = harmonic_correct(win, guide, wm)
    elif strategy == "blocky":
        filled = class_propagate_fill(win, wm)
        filled = harmonic_correct(win, np.clip(filled, 0, 255).astype(np.uint8)
                                  .astype(np.float32), wm)
    else:  # fallback inpaint
        filled = _inpaint_fill(win, wm)
    sigmas = _grain_stats(win, wm)
    filled = _add_grain(np.asarray(filled, np.float32), wm, sigmas, rng=rng)
    return _feather(win, filled, wm)


_STRATEGY_ORDER = {
    "flat": ["flat", "fallback"],
    "periodic": ["periodic", "striped", "blocky", "fallback"],
    "striped": ["striped", "blocky", "fallback"],
    "blocky": ["blocky", "striped", "fallback"],
    "fallback": ["fallback", "blocky"],
}


# ------------------------------------------------------------------ orchestration


def _leftover_glyph_components(orig_gray, wm, L, T):
    """Find text-like components in the ORIGINAL window that the mask missed.

    A component counts as leftover lettering when it is adjacent to the
    existing mask and spatially thin (glyph-stem-like), so artwork shapes a
    few pixels away are never grabbed. Returns a uint8 window mask.
    """
    sub = orig_gray[T:T + wm.shape[0], L:L + wm.shape[1]]
    near = cv2.dilate((wm > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    search = near & (wm == 0)
    if search.sum() < 30:
        return np.zeros_like(wm)
    med = float(np.median(sub))
    p10, p90 = np.percentile(sub, (10, 90))
    dev = max(12.0, 0.38 * float(p90 - p10))
    cand = ((sub > med + dev) | (sub < med - dev)).astype(np.uint8)
    cand &= search.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    add = np.zeros_like(wm)
    hh, ww = wm.shape
    for i in range(1, n):
        x_, y_, w_, h_, area = stats[i]
        if area < 14:
            continue
        comp = (lab == i)
        # must touch the mask-ish vicinity, be thin in at least one axis,
        # and not span the whole window (artwork, not a stray glyph part)
        if w_ > 0.9 * ww or h_ > 0.9 * hh:
            continue
        d = cv2.distanceTransform((1 - comp.astype(np.uint8)), cv2.DIST_L2, 3)
        thickness = float(d.max()) if comp.any() else 0
        if thickness > 0.45 * max(h_, w_):
            continue                    # blob, not stroke-like
        add[comp] = 255
    return add


def erase(bgr, mask, boxes=None, pad=24, work_pad=90, verbose=False,
          max_rounds=2):
    """Erase everything in `mask`, choosing — and VERIFYING — a strategy per box.

    For each box the classifier picks the most promising strategy, the fill
    is measured against the surrounding artwork, and weaker alternatives are
    tried until the patch is statistically indistinguishable (or we run out
    of options and keep the best). Then one leftover-glyph pass re-masks
    what escaped the first pass and re-erases.

    Pixels outside the mask are guaranteed byte-identical to the input.
    """
    H, W = mask.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = bgr.astype(np.float32).copy()
    if boxes is None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return (out, []) if verbose else out
        boxes = [(xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())]

    report = []
    rng = np.random.default_rng(11)

    for box in boxes:
        bx, by, bw, bh = box
        L, T = max(0, int(bx) - work_pad), max(0, int(by) - work_pad)
        R, B = min(W, int(bx + bw) + work_pad), min(H, int(by + bh) + work_pad)
        wm = mask[T:B, L:R]
        if (wm > 0).sum() == 0:
            continue

        kind, angle, st = classify_region(gray, mask, box, pad=pad)

        # periodicity is re-detected on the BIGGER erase window — the small
        # classify window regularly misses fine lattices (print grain, fiber
        # textures of 4-8 px) that the fill CAN copy exactly
        wsub = gray[T:B, L:R].astype(np.float32)
        period_xy, p_strength = _lattice_period(wsub, (wm == 0))
        order = list(_STRATEGY_ORDER.get(kind, ["fallback"]))
        if period_xy is not None and p_strength >= 0.5 and "periodic" not in order:
            # a genuine lattice copy stays available as an alternate, but the
            # classifier's own pick runs first — 1D stripes (speed lines,
            # fibres) belong to the directional transplant, only true 2D
            # grids (halftone) profit from period copying
            order.insert(1, "periodic")
        st["scores"] = []

        tried = []
        best = None
        _candidates_win = []
        for strategy in order[:3]:
            win = out[T:B, L:R].astype(np.uint8)
            filled = _fill_with(win, wm, strategy, angle, period_xy, box, rng)
            if filled is None:
                continue
            _candidates_win.append((strategy, filled))
            q = _window_quality(win, filled, wm)
            br = _boundary_ratio(win, filled, wm)
            lr = _luma_consistency(win, filled, wm)
            sp = _spike_ratio(win, filled, wm)
            tried.append((strategy, round(q["grad"], 2), round(q["detail"], 2),
                          round(br, 2), round(lr, 2), round(sp, 2)))
            sc = _combined_score(q, br, lr, sp)
            if best is None or sc > best[0]:
                best = (sc, filled, strategy, q)
            if sc >= _GOOD_ENOUGH:
                break
        if best[0] <= 0:
            # every strategy was vetoed: ship the least-bad patch rather than
            # nothing — a soft patch under new lettering beats raw lettering
            fallback_scores = [(_score(_window_quality(win, f, wm))
                                - _luma_consistency(win, f, wm), f, s, _window_quality(win, f, wm))
                               for (s, f) in _candidates_win]
            fallback_scores.sort(key=lambda t: -t[0])
            _fb = fallback_scores[0]
            best = (0.01, _fb[1], _fb[2], _fb[3])
        st["scores"] = tried

        _sc, filled, strategy, q = best
        out[T:B, L:R] = filled

        # ---- leftover-glyph pass: re-mask what escaped, re-erase once
        add = _leftover_glyph_components(gray, wm, L, T)
        leftover_px = int((add > 0).sum())
        if leftover_px > 0:
            add = cv2.dilate(add, np.ones((5, 5), np.uint8))
            big = np.maximum(wm, add)
            win = out[T:B, L:R].astype(np.uint8)
            best2 = None
            for strategy in order[:3]:
                filled2 = _fill_with(win, big, strategy, angle, period_xy, box, rng)
                if filled2 is None:
                    continue
                q2 = _window_quality(win, filled2, big)
                br2 = _boundary_ratio(win, filled2, big)
                lr2 = _luma_consistency(win, filled2, big)
                sp2 = _spike_ratio(win, filled2, big)
                sc2 = _combined_score(q2, br2, lr2, sp2)
                tried.append(("L+" + strategy, round(q2["grad"], 2), round(q2["detail"], 2),
                              round(br2, 2), round(lr2, 2), round(sp2, 2)))
                if best2 is None or sc2 > best2[0]:
                    best2 = (sc2, filled2, strategy, q2)
                if sc2 >= _GOOD_ENOUGH:
                    break
            _sc, filled, strategy, q = best2
            # commit only pixels the new mask added plus old mask area; keep
            # everything else exactly as the verified first pass left it
            combo = filled.copy()
            prev = out[T:B, L:R]
            only_new = (add > 0) & (wm == 0)
            combo[~ (only_new | (wm > 0))] = prev[~ (only_new | (wm > 0))]
            out[T:B, L:R] = combo
            st["leftover_px"] = leftover_px

        report.append(dict(box=list(map(int, box)), kind=kind, strategy=strategy,
                           angle=round(float(angle), 1),
                           grad=round(q["grad"], 2), detail=round(q["detail"], 2),
                           **st))
        if verbose:
            print(f"  box {box}: {kind}->{strategy} angle={angle:.1f} "
                  f"grad={q['grad']:.2f} detail={q['detail']:.2f} tried={tried}",
                  file=sys.stderr)

    # never modify anything the mask did not ask for
    out = np.where((mask == 0)[:, :, None], bgr.astype(np.float32), out)
    return (out, report) if verbose else out


def erase_auto(bgr, boxes, pad=20, stroke_px=7, work_pad=90, verbose=False):
    """Convenience wrapper: build the mask and erase in one call.

    boxes is a list of (x, y, w, h). Returns (cleaned_bgr_uint8, mask, report).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = build_mask(gray, boxes, pad=pad, stroke_px=stroke_px)
    out, report = erase(bgr, mask, boxes=boxes, work_pad=work_pad, verbose=True)
    return np.clip(out, 0, 255).astype(np.uint8), mask, report


# ------------------------------------------------------------------------ metrics


def fill_quality(orig_bgr, out_bgr, mask, boxes=None, pad=40):
    """Objective check: does the fill look statistically like its surroundings?"""
    H, W = mask.shape[:2]
    if boxes is None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        boxes = [(xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())]

    results = []
    for box in boxes:
        bx, by, bw, bh = box
        L, T = max(0, int(bx) - pad), max(0, int(by) - pad)
        R, B = min(W, int(bx + bw) + pad), min(H, int(by + bh) + pad)
        wm = mask[T:B, L:R] > 0
        if wm.sum() < 100:
            continue
        q = _window_quality(orig_bgr[T:B, L:R], out_bgr[T:B, L:R],
                            mask[T:B, L:R])
        g = cv2.cvtColor(out_bgr[T:B, L:R].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        Jxx = cv2.GaussianBlur(gx * gx, (0, 0), 6.0)
        Jyy = cv2.GaussianBlur(gy * gy, (0, 0), 6.0)
        Jxy = cv2.GaussianBlur(gx * gy, (0, 0), 6.0)
        coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / (Jxx + Jyy + 1e-9)
        hole = wm
        inner = cv2.erode(hole.astype(np.uint8), np.ones((7, 7), np.uint8), 1) > 0
        ring = (cv2.dilate(hole.astype(np.uint8), np.ones((25, 25), np.uint8), 1) > 0) & (~hole)
        if inner.sum() < 100 or ring.sum() < 100:
            continue
        results.append(dict(
            box=list(map(int, box)),
            grad=round(q["grad"], 2),
            coherence=round(float(coh[inner].mean()) / (float(coh[ring].mean()) + 1e-9), 2),
            detail=round(q["detail"], 2),
        ))
    return results


# ---------------------------------------------------------------------------- CLI


def parse_boxes(spec):
    boxes = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [int(round(float(v))) for v in part.split(",")]
        if len(nums) != 4:
            sys.exit(f"bad box {part!r}, want x,y,w,h")
        boxes.append(tuple(nums))
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--boxes", required=True,
                    help="text regions as 'x,y,w,h;x,y,w,h' (pixels)")
    ap.add_argument("--mask-out", help="also write the mask that was used")
    ap.add_argument("--pad", type=int, default=20, help="mask search padding")
    ap.add_argument("--stroke-px", type=int, default=7,
                    help="minimum outline thickness around glyphs")
    ap.add_argument("--report", action="store_true",
                    help="print per-region strategy and quality metrics as JSON")
    args = ap.parse_args()

    bgr = cv2.imread(args.src)
    if bgr is None:
        sys.exit(f"cannot read {args.src}")
    boxes = parse_boxes(args.boxes)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = build_mask(gray, boxes, pad=args.pad, stroke_px=args.stroke_px)
    if args.mask_out:
        cv2.imwrite(args.mask_out, mask)

    res = erase(bgr, mask, boxes=boxes, verbose=args.report)
    if args.report:
        out, report = res
    else:
        out, report = res, None
    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(args.dst, out)

    if args.report:
        print(json.dumps(dict(
            masked_px=int((mask > 0).sum()),
            regions=report,
            quality=fill_quality(bgr, out, mask, boxes),
        ), indent=2))
    else:
        print(f"{args.src} -> {args.dst}  ({int((mask>0).sum())} px erased)")


if __name__ == "__main__":
    main()
