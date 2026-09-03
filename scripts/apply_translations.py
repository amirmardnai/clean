#!/usr/bin/env python3
"""
Stage 3 helper — patch translations into worksheet.json by item id.

Lets you translate in safe batches instead of rewriting the whole worksheet.
A crash mid-chapter then costs you one batch, not the entire pass.

Input is a JSON file (or stdin) in either shape:

  {"p1-i1": "متن فارسی", "p1-i2": "متن دیگر"}

  {"items": [{"id": "p1-i1", "fa": "متن فارسی"}, ...]}

Usage:
  python3 apply_translations.py OUTDIR batch01.json
  cat batch01.json | python3 apply_translations.py OUTDIR -
  python3 apply_translations.py OUTDIR batch01.json --dry-run
"""
import argparse
import json
import os
import shutil
import sys


def load_patch(path):
    raw = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    data = json.loads(raw)
    if isinstance(data, dict) and "items" in data:
        return {it["id"]: it.get("fa", "") for it in data["items"] if "id" in it}
    if isinstance(data, list):
        return {it["id"]: it.get("fa", "") for it in data if "id" in it}
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, str)}
    sys.exit("unrecognised patch format")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("patch", help="JSON file of id->translation, or - for stdin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace translations that are already filled in")
    args = ap.parse_args()

    wpath = os.path.join(args.outdir, "worksheet.json")
    if not os.path.exists(wpath):
        sys.exit("worksheet.json missing — run ocr_pages.py first")

    patch = load_patch(args.patch)
    if not patch:
        sys.exit("patch contained no translations")

    with open(wpath, encoding="utf-8") as f:
        ws = json.load(f)

    index = {it["id"]: it for p in ws["pages"] for it in p.get("items", [])}

    applied = skipped_filled = 0
    unknown = []
    for item_id, fa in patch.items():
        target = index.get(item_id)
        if target is None:
            unknown.append(item_id)
            continue
        if target.get("fa", "").strip() and not args.overwrite:
            skipped_filled += 1
            continue
        target["fa"] = fa
        applied += 1

    if unknown:
        print(f"! {len(unknown)} unknown id(s): {', '.join(unknown[:8])}"
              + (" ..." if len(unknown) > 8 else ""), file=sys.stderr)

    if args.dry_run:
        print(f"dry run: would apply {applied}, "
              f"skip {skipped_filled} already-filled")
        return

    shutil.copy(wpath, wpath + ".bak")
    with open(wpath, "w", encoding="utf-8") as f:
        json.dump(ws, f, ensure_ascii=False, indent=2)

    remaining = sum(
        1 for it in index.values()
        if it.get("kind") == "dialogue" and not it.get("fa", "").strip()
    )
    print(f"applied {applied}, skipped {skipped_filled} already-filled")
    print(f"{remaining} dialogue region(s) still untranslated")
    print(f"backup -> {wpath}.bak")


if __name__ == "__main__":
    main()
