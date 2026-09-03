---
name: manhwa-translate
description: Translate the English text in a manhwa/manga/comic (PDF, CBZ, CBR, or image folder) into Persian or another language, then typeset the translation back into the speech bubbles and export a finished PDF/CBZ. Also handles single pages and lettering that sits on artwork rather than in bubbles — shouts, narration and SFX over speed lines or texture — using a structure-aware eraser that rebuilds the background instead of leaving a grey patch. Use when the user asks to translate a comic, localize scanlation pages, replace bubble text, clean or redraw lettering, or build a Persian version of a manhwa chapter.
---

# Manhwa / Comic Translation Pipeline

Turns an English comic into a translated, re-lettered comic. The pipeline
separates **mechanical work** (scripted: extraction, OCR, masking, typesetting,
packaging) from **judgment work** (the model's: actually translating, and
deciding what is dialogue vs. noise).

Stage 3 (translation) is deliberately a human/model step — it is the only stage
that needs language judgment, and it is where quality is won or lost.

## Pipeline at a glance

```
source.pdf/cbz/cbr/dir
      │
      ▼  Stage 1  extract_pages.py       → pages/*.png + manifest.json
      ▼  Stage 2  ocr_pages.py           → worksheet.json  (English + boxes)
      ▼  Stage 2b check_worksheet.py     → QA report        [recommended]
      ▼  Stage 3  ***you translate***    → fill every "fa" field
      ▼           apply_translations.py  → patch in batches  [optional]
      ▼  Stage 4  render_pages.py        → rendered/*.png    (masked + typeset)
      ▼  Stage 4b clean_lettering.py     → text on artwork, not in bubbles
      ▼           typeset_lettering.py     (structure-aware erase + stroked text)
      ▼  Stage 5  assemble_output.py     → Title.pdf / Title.cbz
```

Every stage reads and writes plain JSON, so any stage can be rerun without
redoing the ones before it.

For a single page, Stages 4b and 5 are all you need: get the boxes from
`apple-vision ocr`, erase, typeset. There is no need to build a job directory.

## Before you start

**Rights check.** Re-lettering a comic reproduces a copyrighted work in another
language. Only run this on material the user has the right to translate: their
own work, public-domain titles, officially licensed work they are contracted on,
or personal-use translation of a copy they own. If the user is asking for a
redistributable translation of a commercial title they do not hold rights to,
say so and stop. Do not use the pipeline to strip out or replace another
scanlation team's credits so the output looks original — that is what the
`kind: "credit"` tag exists to protect.

## One-time setup

```sh
sh scripts/setup.sh
```

Installs and verifies everything, and is safe to re-run. Equivalent to:

```sh
apk add poppler-utils 7zip py3-numpy py3-opencv   # pdftoppm, pdfimages, 7z, cv2
pip install --break-system-packages arabic-reshaper python-bidi img2pdf Pillow
```

`py3-opencv` is only needed for Stage 4b (the structure-aware eraser). Install
it from apk, never from pip — there is no musllinux aarch64 wheel, so pip tries
to build OpenCV from source and will not finish.

`assets/fonts/` ships Vazirmatn Regular/Bold/ExtraBold (SIL OFL). Bold is the
default — comic bubbles need weight to stay legible at small sizes. For non-RTL
target languages, point `--font` at any TTF; the shaping step is harmless for
Latin text.

## Pipeline

Work in a scratch directory (`/tmp/<job>`), not in `/var/minis/shared`. Only the
final deliverable goes to a shared location.

### Stage 1 — extract pages

```sh
python3 scripts/extract_pages.py "SOURCE" /tmp/job --dpi 200
```

Handles `.pdf`, `.cbz`/`.zip`, `.cbr`/`.cb7`/`.7z`, and directories of images.
Writes `pages/page-0001.png…` in exact reading order plus `manifest.json`.
Order is preserved via natural sort — `page10` lands after `page9`, not after
`page1`.

- `--dpi 200` is the sweet spot: OCR reads reliably, files stay manageable.
  Drop to 150 for very long chapters, raise to 300 only if lettering is tiny.
- `--first N --last N` limits the range. **Always test on 3–5 pages before
  committing to a 100+ page run.**

Long manhwa pages are extremely tall (e.g. 1667×4167 at 150 dpi). That is
normal and the rest of the pipeline handles it.

### Stage 2 — OCR into a worksheet

```sh
python3 scripts/ocr_pages.py /tmp/job
```

Uses `apple-vision ocr --level accurate` (Apple Vision framework). It
substantially outperforms tesseract on comic lettering — do not swap it out.

Vision returns bboxes as `[x, y, w, h]` normalized with the **origin at
bottom-left**. The script converts these to top-left pixel rects; downstream
code assumes top-left. Do not "fix" this twice.

Adjacent OCR lines are merged into bubble-level regions when they overlap
horizontally and sit within `--merge-gap` (default 2% of page height) of each
other. Output `worksheet.json`:

```json
{"pages":[{"page":1,"file":"page-0001.png","width":1667,"height":4167,
  "items":[{"id":"p1-i1","box":[676,1198,689,316],"en":"LEAVE THE SOCIAL WORLD TO YOUR BROTHER AND SISTER.","fa":"","kind":"dialogue","conf":1.0,"line_count":4}]}]}
```

Regions matching watermark/credit patterns (site URLs, "translator", "join our
discord", …) are tagged `kind: "credit"` and skipped at render time.

Tuning knobs:
- `--min-conf 0.35` — raise to cut junk, lower if real dialogue is dropping.
- `--merge-gap` — raise if one bubble splits into several items, lower if two
  bubbles collapse into one.
- `--lang`, `--level fast|accurate`.

### Stage 2b — QA the worksheet before translating

```sh
python3 scripts/check_worksheet.py /tmp/job
```

Flags tiny/huge/off-page boxes, overlapping masks that will collide, likely OCR
noise, and low-confidence regions. Fixing boxes here takes seconds; discovering
them after a 100-page render costs a full rerun. `--verbose` lists every
dialogue region with its text and TODO status.

### Stage 3 — translate (this is your job)

Read `worksheet.json` and fill every `fa` field. Rules:

1. **Never leave a `dialogue` item empty** unless it is genuine noise (stray
   sound-effect fragments, page numbers, OCR garbage like `"3S005"`). Empty `fa`
   means the original English stays visible.
2. **Translate in page order, reading the whole page's items together.**
   Bubbles are a conversation; translating them in isolation produces
   disconnected dialogue and wrong pronouns.
3. **Fix OCR damage from context.** All-caps comic lettering produces
   predictable errors (`WERSITE`→`WEBSITE`, `NORE`→`MORE`, `I`/`l`/`1`
   confusion). Translate what the line clearly means, not the corrupted string.
4. **Match register.** Manhwa dialogue is spoken, not literary. Use natural
   conversational Persian. Court/historical settings (common in this genre) need
   appropriate honorifics — keep forms of address like «اعلی‌حضرت»، «سرورم»،
   «بانوی من» consistent across the whole chapter.
5. **Keep a glossary.** Copy `assets/glossary.template.json` to
   `glossary.json` next to the worksheet, fill in names, titles, places and
   in-world terms, and reuse it for every chapter of the series. Inconsistent
   character names across a chapter is the most common quality failure.
6. **Respect the bubble.** Short English bubbles need short Persian. Persian
   runs ~15–25% longer than English; if a translation is much longer than the
   original, tighten it rather than relying on autosizing to shrink it to
   unreadable.
7. **Leave `kind: "credit"` items untranslated** unless the user explicitly
   asks otherwise.

For long chapters, translate in batches (20–30 pages) and patch them in with
`apply_translations.py` so a failure never loses the whole pass:

```sh
# batch01.json:  {"p1-i1": "متن فارسی", "p1-i2": "متن دیگر"}
python3 scripts/apply_translations.py /tmp/job batch01.json
```

It backs up the worksheet, refuses to clobber already-filled fields unless
`--overwrite` is given, warns about unknown ids, and reports how many dialogue
regions remain. `--dry-run` previews without writing.

### Stage 4 — mask and typeset

```sh
python3 scripts/render_pages.py /tmp/job
```

Per region: samples the dominant colour in a 4px ring just outside the box,
floods the box with it (clean erase inside flat bubbles), then shapes the
Persian with `arabic_reshaper` + the bidi algorithm and draws it centred,
auto-sized via binary search over font size, word-wrapped to the box. Text
colour flips to white automatically on dark backgrounds.

Options:
- `--font assets/fonts/Vazirmatn-ExtraBold.ttf` — heavier lettering.
- `--min-size 11 --max-size 90` — size search bounds.
- `--grow 1.15` — how far a box may expand when text refuses to fit. Set `1.0`
  to forbid expansion (safer on dense pages where boxes nearly touch).
- `--pad 6` — inner margin.
- `--debug-boxes` — outlines every replaced region in red. **Use this on your
  test pages**; it makes bad boxes obvious instantly.
- `--include-credits` — also overwrite credit regions (see the rights note).

Output: `rendered/page-0001.png…`

### Stage 4b — text that is NOT inside a bubble

`render_pages.py` erases by flooding the box with the surrounding colour. That
is exactly right inside a flat speech bubble and exactly wrong for lettering
that sits directly on artwork — shouts, narration, SFX, anything over speed
lines, gradients or texture. Flooding those leaves a grey rectangle that is
more obvious than the original text was.

For those regions use the two dedicated scripts instead. They work per page,
per region, so run them only on the pages that need it.

**Erase:**

```sh
python3 scripts/clean_lettering.py page.png cleaned.png \
    --boxes '37,31,341,67;39,109,291,75' --report
```

The eraser (v2, self-verifying) masks glyph cores plus their outline — the
outline width is estimated from the image itself (largest kernel that fits
inside two flat zones), weak-but-touching glyph fragments are adopted
iteratively, and after every fill one leftover pass re-finds ghost text the
first mask missed, so nothing is left behind or eaten into the art. It then
classifies each region, TRIES the best fill, quantitatively compares the
patch with the surrounding artwork (texture energy, border seams, brightness
drift, interior edge spikes, luminance regularity), and retries with an
alternative fill whenever the first one would read as a smudge — the output
never knowingly ships a bad patch:

| kind | when | how it fills |
|---|---|---|
| `flat` | box interior is nearly uniform | flood with the sampled interior colour (+ measured grain) |
| `periodic` | background repeats on a 2D lattice (halftone dots) | walk along the lattice vector to real clean pixels |
| `striped` | constant along one direction (speed lines, fibres, gradients) | transplant real pixels along that direction, seam-correct after |
| `blocky` | hard-edged shapes (armour plates, panel borders) | nearest colour-class propagation |
| `fallback` | busy, directionless artwork | two-pass isotropic inpaint |

Every fill finishes with grain re-synthesis measured from the clean ring —
a perfectly smooth patch betrays itself even when its macrostructure is
right. Pixels outside the mask are guaranteed unchanged.

`--report` prints the classified kind, every attempted fill with its quality
scores, and the winning metrics — see Quality control below. Even in the
worst case (every strategy vetoed, e.g. genuinely irrecoverable artwork) the
least-bad patch is shipped rather than a guaranteed smudge — retry the box
by hand only when even that is not good enough.


**Lettering colour follows the original, not a default.** `typeset_lettering.py`
defaults to black fill with a white stroke, which suits dark lettering on light
art. Pages that letter in white with a black outline need
`--fill ffffff --stroke 000000`, and non-italic lettering needs `--slant 0`.
Check the source page before typesetting; getting this backwards is the most
visible mistake available. Do not compensate for a thin-looking outline by
raising `--stroke-ratio` much past the default — beyond roughly 0.13 the stroke
starts closing up the counters of Persian letters.

**Typeset:**

```sh
python3 scripts/typeset_lettering.py cleaned.png out.png \
    --line '37,31,341,67:گور بابای' \
    --line '39,109,291,75:جنگتون!'
```

One `--line` per original text line, using that line's own box, so translated
lines land where their source lines were instead of being re-wrapped as a
block. Draws black fill with a white stroke by default (`--fill` / `--stroke`,
`--stroke none` to disable), applies an italic shear (`--slant 0.16`, `0` for
upright), and renders at 4x before downsampling because stroked glyphs are
visibly ragged otherwise.

Both scripts are importable if you want to drive them from Python:
`build_mask`, `erase`, `fill_quality` from `clean_lettering`.

**One page, one command.** For a single image, `oneshot_page.sh` chains
OCR → erase → typeset and handles the box conversion for you (apple-vision
reports normalised boxes with the origin at bottom-left, which is easy to get
wrong by hand):

```sh
sh scripts/oneshot_page.sh page.png out.png "خط اول" "خط دوم"
```

Translated lines are matched to OCR lines in reading order, top to bottom. Run
it with a wrong number of lines and it prints the boxes it found with their
English text, so you can see the order before committing.

`oneshot_page.sh` uses the typesetting defaults. When the page letters in white
on black, or upright rather than italic, call the two scripts separately so you
can pass `--fill` / `--stroke` / `--slant`.

**Self-test.** After changing `clean_lettering.py`, run:

```sh
sh scripts/selftest_lettering.sh
```

It builds two synthetic pages (light lettering over speed lines; dark lettering
in a flat bubble), checks each takes the expected strategy, that OCR finds
nothing afterwards, and that the texture metrics stay in range.

### Stage 5 — package

```sh
python3 scripts/assemble_output.py /tmp/job --title "Title Ch 1 - Persian" \
    --pdf --cbz --dest /var/minis/shared/manhwa-fa
```

`--quality 85` (default) re-encodes to JPEG for a much smaller PDF; pass
`--quality 0` for lossless PNG pages. CBZ always keeps the rendered PNGs.

## Quality control

Run this loop before the full chapter, and again on a sample afterwards:

1. Render 3–5 representative pages with `--debug-boxes`.
2. Open them: `minis-open /tmp/job/rendered/page-0003.png`
3. Check: text inside the bubble, no clipping, no original English peeking out
   at box edges, no bubble outline erased, no text overlapping artwork.
4. Verify pixels actually changed rather than trusting the log:

```sh
python3 -c "
from PIL import Image, ImageChops
a=Image.open('/tmp/job/pages/page-0001.png').convert('RGB')
b=Image.open('/tmp/job/rendered/page-0001.png').convert('RGB')
print('changed px:', sum(1 for p in ImageChops.difference(a,b).convert('L').getdata() if p>40))"
```

### Judging an erase objectively

Vision models are unreliable at grading erase quality in isolation — asked
twice, the same patch gets called "clean" and "obvious artifacts". Two things
work better.

**Metrics.** `clean_lettering.py --report` compares the filled pixels against a
ring around them on three axes: `grad` (edge energy), `coherence` (how
directional the structure is) and `detail` (high-frequency content). Ratios
near 1.0 mean the patch is statistically indistinguishable from its
surroundings. Well under 1.0 means a smudge — for reference, on speed-line
artwork plain `cv2.inpaint` scores `grad` around 0.45 while the directional
transplant scores 0.7–1.0.

**A/B, not absolute.** If you do ask a vision model, paste the two candidates
side by side with a white separator and ask which is better and why. Comparative
judgements are consistent; absolute grades are not.

Also re-run OCR on the cleaned page: it should come back empty over the erased
regions. Note that a stray low-confidence hit elsewhere on the page is usually
artwork the eraser never touched — check the box coordinates before chasing it.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Persian shows as disconnected letters or reversed | shaping bypassed | must go through `shape()` in `render_pages.py`; never draw raw Persian |
| Tofu boxes / blank glyphs | font lacks Arabic script | use the bundled Vazirmatn, not DejaVu |
| Erase block bleeds over artwork or bubble outline | OCR box too generous | lower `--grow` to 1.0, reduce `--pad`, or shrink the `box` in the worksheet |
| Grey rectangle where text used to be | flat flood used on textured background | this region needs Stage 4b `clean_lettering.py`, not `render_pages.py` |
| Streaking / smeared colour bands after a `textured` fill | background is blocky (armour plates, panel borders, hard-edged shapes) rather than striped — the transplant drags one shape's colour across another | acceptable when the new lettering covers it; otherwise erase with plain `cv2.inpaint` for that box, or redraw by hand. The `coherence` score does not catch this: blocky art scores high because the *direction* is consistent even though the *colour* along it is not. |
| Ghost outline still visible after erasing | white stroke not fully masked | raise `--stroke-px` (try 9–11) |
| Erased patch has right texture but wrong brightness | seam correction had too little clean boundary to measure | widen the box slightly so the boundary ring sits on real background |
| Original English still visible at box edges | box too tight | widen the `box` values for that item |
| One bubble split into 2–3 items | leading exceeds merge gap | raise `--merge-gap` (try 0.03–0.04) |
| Two bubbles merged into one item | boxes too close | lower `--merge-gap`, or split the item by hand |
| Text microscopic | translation too long for bubble | shorten the Persian; do not lower `--min-size` |
| Pages out of order | source used non-padded names | `extract_pages.py` natural-sorts; verify `manifest.json` order |
| OCR returns almost nothing | rendered too small, or text is stylized SFX | re-extract at higher `--dpi`; heavily stylized sound effects often need manual entry |

## Notes for whoever runs this

- Every stage is resumable and reads/writes plain JSON. If translation dies
  mid-chapter, rerun Stage 4 — it only touches items with a non-empty `fa`.
- Stage 2's OCR call is the slow part on long chapters. Run it once, keep
  `worksheet.json`, iterate on translation and rendering freely.
- Sound effects baked into artwork cannot be cleanly replaced by box-masking.
  Either leave them, or add a worksheet item by hand with a tight box.
- The scripts never modify the source file.
