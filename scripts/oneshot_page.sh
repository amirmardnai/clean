#!/bin/sh
# End-to-end demo of Stage 4b on one page: OCR -> erase -> typeset.
#
# This is the shortest path for "clean the English off this page and put my
# translation in its place". It shows how to turn apple-vision's normalised
# OCR boxes into the pixel boxes the two scripts expect.
#
#   sh scripts/oneshot_page.sh PAGE.png OUT.png "line one" "line two" ...
#
# The translated lines are matched to the OCR lines in reading order, so pass
# them in the same order apple-vision reports (top to bottom).
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$1"; DST="$2"; shift 2 || true

if [ -z "$SRC" ] || [ -z "$DST" ] || [ "$#" -eq 0 ]; then
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

echo "==> OCR"
apple-vision ocr "$SRC" --level accurate > /tmp/ocr.$$.json
python3 - "$SRC" /tmp/ocr.$$.json "$@" > /tmp/args.$$.sh <<'PY'
import json
import sys

from PIL import Image

src, ocr_path, *lines = sys.argv[1:]
W, H = Image.open(src).size
data = json.load(open(ocr_path))["data"]
blocks = data.get("blocks", [])

# apple-vision returns [x, y, w, h] normalised with the ORIGIN AT BOTTOM-LEFT.
# Convert to top-left pixel boxes and sort into reading order.
boxes = []
for b in blocks:
    x, y, w, h = b["bbox"]
    boxes.append((int(round(x * W)), int(round((1 - y - h) * H)),
                  int(round(w * W)), int(round(h * H)), b["text"]))
boxes.sort(key=lambda t: (t[1], t[0]))

if len(lines) != len(boxes):
    print(f"echo '    ! {len(boxes)} OCR lines but {len(lines)} translations given'",
          file=sys.stderr)
    for bx, by, bw, bh, text in boxes:
        print(f"echo '      {bx},{by},{bw},{bh}  {text}'", file=sys.stderr)
    raise SystemExit(1)

erase = ";".join(f"{bx},{by},{bw},{bh}" for bx, by, bw, bh, _ in boxes)
print(f"BOXES='{erase}'")
for (bx, by, bw, bh, eng), fa in zip(boxes, lines):
    print(f"echo '    {eng}  ->  {fa}'")
    safe = fa.replace("'", "'\"'\"'")
    print(f"set -- \"$@\" --line '{bx},{by},{bw},{bh}:{safe}'")
PY

set --
. /tmp/args.$$.sh

echo "==> erase"
python3 "$HERE/clean_lettering.py" "$SRC" "/tmp/cleaned.$$.png" \
    --boxes "$BOXES" --report | python3 -c "
import json, sys
r = json.load(sys.stdin)
for reg, q in zip(r['regions'], r['quality']):
    print(f\"    {reg['box']}  {reg['kind']:9s} grad={q['grad']} detail={q['detail']}\")"

echo "==> typeset"
python3 "$HERE/typeset_lettering.py" "/tmp/cleaned.$$.png" "$DST" "$@"

echo "==> verify (OCR should find no English; Persian may read as garbage)"
apple-vision ocr "$DST" --level accurate -q 2>/dev/null | grep '"text"' | head -3

rm -f /tmp/ocr.$$.json /tmp/args.$$.sh /tmp/cleaned.$$.png
echo "done -> $DST"
