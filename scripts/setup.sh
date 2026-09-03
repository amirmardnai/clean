#!/bin/sh
# One-shot setup for the manhwa-translate skill.
# Idempotent — safe to re-run. Verifies every dependency the pipeline needs.
set -e

echo "==> installing system packages"
apk add --quiet poppler-utils 7zip py3-numpy py3-pillow py3-opencv 2>/dev/null || \
  echo "    (apk step reported issues — continuing to verify)"

echo "==> installing python packages"
# apk's py3-opencv registers itself as version "python-4.10.0", which pip's
# version parser rejects — it crashes while scanning installed packages even
# though nothing is wrong. Failures here are non-fatal; the verify step below
# is what decides whether setup succeeded.
pip install --quiet --break-system-packages \
  arabic-reshaper python-bidi img2pdf Pillow >/dev/null 2>&1 \
  || echo "    (pip reported issues — verifying imports directly)"

echo "==> verifying"
fail=0

for cmd in pdftoppm pdfimages 7z img2pdf apple-vision; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "    ok   $cmd"
  else
    echo "    MISS $cmd"
    fail=1
  fi
done

python3 - <<'PY' || fail=1
mods = ["PIL", "arabic_reshaper", "bidi.algorithm", "numpy", "cv2"]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"    ok   python:{m}")
    except ImportError:
        print(f"    MISS python:{m}")
        bad.append(m)
raise SystemExit(1 if bad else 0)
PY

FONT_DIR="$(dirname "$0")/../assets/fonts"
for f in Vazirmatn-Regular.ttf Vazirmatn-Bold.ttf Vazirmatn-ExtraBold.ttf; do
  if [ -f "$FONT_DIR/$f" ]; then
    echo "    ok   font:$f"
  else
    echo "    MISS font:$f"
    fail=1
  fi
done

echo "==> RTL shaping self-test"
python3 - <<'PY' || fail=1
import sys
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    out = get_display(arabic_reshaper.reshape("سلام دنیا"))
    assert out and out != "سلام دنیا", "shaping produced no change"
    print("    ok   shaping + bidi working")
except Exception as e:
    print(f"    FAIL shaping: {e}")
    sys.exit(1)
PY

echo
if [ "$fail" = "0" ]; then
  echo "setup complete — all dependencies present"
else
  echo "setup INCOMPLETE — see MISS/FAIL lines above"
  exit 1
fi
