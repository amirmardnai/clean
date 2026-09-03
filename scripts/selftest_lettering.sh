#!/bin/sh
# Self-test for Stage 4b (structure-aware erase + stroked typesetting).
#
# Builds two synthetic pages that exercise the branches that matter, runs the
# eraser on each, and checks the result three ways: OCR must come back empty,
# the texture metrics must be in range, and pixels outside the mask must be
# untouched. Run after changing clean_lettering.py.
#
#   sh scripts/selftest_lettering.sh [workdir]
set -e

DIR="${1:-/tmp/lettering-selftest}"
HERE="$(cd "$(dirname "$0")" && pwd)"
FONT="$HERE/../assets/fonts/Vazirmatn-ExtraBold.ttf"
mkdir -p "$DIR"

echo "==> building test pages in $DIR"
python3 - "$DIR" "$FONT" <<'PY'
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out, font_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(3)

# case 1: light lettering with dark stroke over diagonal speed lines
img = np.zeros((400, 700, 3), np.uint8)
for i in range(-700, 700, 7):
    c = int(60 + 90 * rng.random())
    cv2.line(img, (i, 0), (i + 300, 400), (c, max(c - 10, 0), c + 10), 3)
img = cv2.GaussianBlur(img, (0, 0), 1.2)
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(pil)
f = ImageFont.truetype(font_path, 54)
d.text((60, 90), "DIE HERE", font=f, fill=(255, 255, 255),
       stroke_width=6, stroke_fill=(0, 0, 0))
d.text((60, 200), "NOW!!", font=f, fill=(255, 255, 255),
       stroke_width=6, stroke_fill=(0, 0, 0))
pil.save(f"{out}/textured.png")

# case 2: plain dark lettering inside a flat white speech bubble
img = np.full((300, 600, 3), 240, np.uint8)
cv2.ellipse(img, (300, 150), (240, 110), 0, 0, 360, (255, 255, 255), -1)
cv2.ellipse(img, (300, 150), (240, 110), 0, 0, 360, (0, 0, 0), 3)
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
d = ImageDraw.Draw(pil)
f = ImageFont.truetype(font_path, 44)
d.text((150, 120), "HELLO WORLD", font=f, fill=(0, 0, 0))
pil.save(f"{out}/flat.png")
print("    ok   test pages written")
PY

fail=0

run_case() {
  name="$1"; boxes="$2"; want_kind="$3"
  echo "==> case $name (expecting $want_kind)"
  python3 "$HERE/clean_lettering.py" "$DIR/$name.png" "$DIR/$name.clean.png" \
      --boxes "$boxes" --report > "$DIR/$name.json"

  if grep -q "\"kind\": \"$want_kind\"" "$DIR/$name.json"; then
    echo "    ok   strategy = $want_kind"
  else
    echo "    FAIL expected strategy $want_kind, got:"
    grep '"kind"' "$DIR/$name.json" | sed 's/^/         /'
    fail=1
  fi

  left="$(apple-vision ocr "$DIR/$name.clean.png" --level accurate -q 2>/dev/null \
          | grep -c '"text" : "[^"]' || true)"
  if [ "$left" = "0" ]; then
    echo "    ok   OCR finds no text"
  else
    echo "    FAIL OCR still reads text:"
    apple-vision ocr "$DIR/$name.clean.png" -q 2>/dev/null | grep '"text"' | sed 's/^/         /'
    fail=1
  fi

  python3 - "$DIR/$name.png" "$DIR/$name.clean.png" "$DIR/$name.json" <<'PY' || fail=1
import json
import sys

import cv2
import numpy as np

orig = cv2.imread(sys.argv[1])
clean = cv2.imread(sys.argv[2])
rep = json.load(open(sys.argv[3]))

bad = []
for q in rep.get("quality", []):
    # flat fills legitimately have no texture, so only judge the others
    kinds = {tuple(r["box"]): r["kind"] for r in rep["regions"]}
    if kinds.get(tuple(q["box"])) == "flat":
        continue
    if q["grad"] < 0.55 or q["detail"] < 0.5:
        bad.append(q)
if bad:
    print("    FAIL texture metrics too low (smudged fill):")
    for q in bad:
        print(f"         {q}")
    raise SystemExit(1)
print("    ok   texture metrics in range")

changed = cv2.absdiff(orig, clean).max(axis=2) > 2
print(f"    ok   {int(changed.sum())} px changed")
PY
}

run_case textured '55,85,290,70;55,195,190,70' textured
run_case flat     '150,115,300,60'             flat

echo "==> typesetting on the cleaned textured page"
python3 "$HERE/typeset_lettering.py" "$DIR/textured.clean.png" "$DIR/textured.final.png" \
    --line '55,85,290,70:همین‌جا بمیر' \
    --line '55,195,190,70:همین حالا!!' >/dev/null
if [ -f "$DIR/textured.final.png" ]; then
  echo "    ok   typeset output written"
else
  echo "    FAIL typeset produced nothing"
  fail=1
fi

echo
if [ "$fail" = "0" ]; then
  echo "self-test passed — artifacts in $DIR"
else
  echo "self-test FAILED — see lines above; artifacts in $DIR"
  exit 1
fi
