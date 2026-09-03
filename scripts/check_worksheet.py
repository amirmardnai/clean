#!/usr/bin/env python3
"""
Stage 2b — QA the worksheet before you spend time translating.

Reports per-page region counts, suspicious boxes (tiny, huge, overlapping,
off-page), likely OCR garbage, and credit/watermark regions. Fixing boxes here
costs seconds; fixing them after a 100-page render costs a rerun.

Usage:
  python3 check_worksheet.py OUTDIR [--verbose]
"""
import argparse
import json
import os
import re
import sys


def is_garbage(text):
    """Heuristics for OCR noise that is not real dialogue."""
    t = text.strip()
    if len(t) <= 2:
        return True
    letters = sum(c.isalpha() for c in t)
    if letters == 0:
        return True
    if letters / max(1, len(t.replace(" ", ""))) < 0.45:
        return True
    if re.fullmatch(r"[\W\d_]+", t):
        return True
    return False


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = min(ax + aw, bx + bw) - max(ax, bx)
    iy = min(ay + ah, by + bh) - max(ay, by)
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / min(aw * ah, bw * bh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--verbose", action="store_true",
                    help="list every dialogue region, not just problems")
    args = ap.parse_args()

    wpath = os.path.join(args.outdir, "worksheet.json")
    if not os.path.exists(wpath):
        sys.exit("worksheet.json missing — run ocr_pages.py first")
    with open(wpath, encoding="utf-8") as f:
        ws = json.load(f)

    n_dialogue = n_credit = n_translated = 0
    problems = []
    empty_pages = []

    for page in ws["pages"]:
        pw, ph = page.get("width", 0), page.get("height", 0)
        items = page.get("items", [])
        dial = [i for i in items if i.get("kind") == "dialogue"]
        n_dialogue += len(dial)
        n_credit += sum(1 for i in items if i.get("kind") == "credit")
        n_translated += sum(1 for i in items if i.get("fa", "").strip())

        if page.get("ocr_failed"):
            problems.append((page["page"], "-", "OCR FAILED for this page"))
        elif not items:
            empty_pages.append(page["page"])

        for it in dial:
            x, y, w, h = it["box"]
            area = (w * h) / max(1, pw * ph)
            if w < 12 or h < 10:
                problems.append((page["page"], it["id"], f"box tiny ({w}x{h})"))
            if area > 0.30:
                problems.append((page["page"], it["id"],
                                 f"box covers {area:.0%} of page — likely bad merge"))
            if x < 0 or y < 0 or x + w > pw or y + h > ph:
                problems.append((page["page"], it["id"], "box extends off-page"))
            if is_garbage(it.get("en", "")):
                problems.append((page["page"], it["id"],
                                 f"likely OCR noise: {it['en'][:30]!r}"))
            if it.get("conf", 1) < 0.5:
                problems.append((page["page"], it["id"],
                                 f"low confidence ({it['conf']})"))

        for i in range(len(dial)):
            for j in range(i + 1, len(dial)):
                ov = overlaps(dial[i]["box"], dial[j]["box"])
                if ov > 0.35:
                    problems.append((page["page"],
                                     f"{dial[i]['id']}/{dial[j]['id']}",
                                     f"boxes overlap {ov:.0%} — masks will collide"))

        if args.verbose:
            print(f"--- page {page['page']} ({pw}x{ph})")
            for it in dial:
                mark = "OK " if it.get("fa", "").strip() else "TODO"
                print(f"  {mark} {it['id']} {it['box']} {it['en'][:60]!r}")

    print(f"pages:            {ws['page_count']}")
    print(f"dialogue regions: {n_dialogue}")
    print(f"credit regions:   {n_credit} (skipped at render time)")
    print(f"translated:       {n_translated}/{n_dialogue}")
    if empty_pages:
        preview = ", ".join(map(str, empty_pages[:15]))
        print(f"no text found on {len(empty_pages)} page(s): {preview}"
              + (" ..." if len(empty_pages) > 15 else ""))
        print("  (normal for splash/art pages; if it is most of the chapter, "
              "re-extract at higher --dpi)")

    if problems:
        print(f"\n{len(problems)} issue(s):")
        for pg, iid, msg in problems[:60]:
            print(f"  p{pg} {iid}: {msg}")
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more")
    else:
        print("\nno issues detected")


if __name__ == "__main__":
    main()
