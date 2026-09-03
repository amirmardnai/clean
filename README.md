# manhwa-translate

A pipeline that turns an English comic (manhwa / manga / webtoon)
into a translated, re-lettered comic — PDF or CBZ out the other end.

There are **two entry points**:

### A. One-shot auto translator — `translate_manhwa_pro.py`

Fully automatic: EasyOCR (tiled, safe for 50,000-px webtoon strips) → LLM
translation → cleaning + re-lettering → PDF + CBZ + chapter ZIP. This is
what `run_translate_all.bat` drives.

Since v4 the re-lettering stage routes **every detected region** either
through the classic speech-bubble path (flat interior → inpaint → centred
plain text) or the **structure-aware artwork path**: SFX, titles, shouts,
system-panel and caption text sitting on artwork are erased with
`scripts/clean_lettering.py` (self-verifying — it never ships a smudged
patch without trying an alternative strategy) and redrawn with
`typeset_artwork_block` in the *sampled original colours and stroke*.

```
python translate_manhwa_pro.py <folder-or-pdf> -o OUTDIR [--work DIR]
       [--langs en,ko] [--cpu] [--slant 0.12] [--no-artwork] [--debug-boxes]
```

- every page resumes automatically (finished pages are skipped)
- bubbles keep the old behaviour that already worked
- artwork text now keeps the page's texture instead of a white/grey box
- env overrides: `MANHWA_API_KEY`, `MANHWA_LLM_URL`, `MANHWA_LLM_MODEL`

### B. Five-stage manual/worksheet pipeline — `scripts/`

The design principle: **scripts do the mechanical work, the model does the
translating.** Only Stage 3 needs language judgment. Everything else is
deterministic and resumable.

```
source.pdf/cbz/cbr/dir
      │
      ▼  Stage 1  extract_pages.py       → pages/*.png + manifest.json
      ▼  Stage 2  ocr_pages.py           → worksheet.json  (English + boxes)
      ▼  Stage 2b check_worksheet.py     → QA report        [recommended]
      ▼  Stage 3  ***you translate***    → fill every "fa" field
      ▼           apply_translations.py  → patch in batches  [optional]
      ▼  Stage 4  render_pages.py        → rendered/*.png    (masked + typeset)
      ▼  Stage 5  assemble_output.py     → Title.pdf / Title.cbz
```

## Quick start

```sh
sh scripts/setup.sh                                    # once

python3 scripts/extract_pages.py "chapter.pdf" /tmp/job --dpi 200 --first 1 --last 5
python3 scripts/ocr_pages.py /tmp/job
python3 scripts/check_worksheet.py /tmp/job

#  ... translate: fill the "fa" fields in /tmp/job/worksheet.json ...

python3 scripts/render_pages.py /tmp/job --debug-boxes   # QA the boxes first
python3 scripts/assemble_output.py /tmp/job --title "Chapter 1 - Persian" --pdf --cbz
```

Test on 3–5 pages before committing to a full chapter. Stage 2 is the slow part;
once `worksheet.json` exists you can iterate on translation and rendering for
free.

## Contents

| Path | Purpose |
|---|---|
| `SKILL.md` | Full instructions — read this first |
| `scripts/setup.sh` | Installs and verifies every dependency |
| `scripts/extract_pages.py` | Stage 1 — source → ordered page images |
| `scripts/ocr_pages.py` | Stage 2 — Apple Vision OCR → translation worksheet |
| `scripts/check_worksheet.py` | Stage 2b — flags bad boxes and OCR noise |
| `scripts/apply_translations.py` | Stage 3 — patch translations in by id, in batches |
| `scripts/render_pages.py` | Stage 4 — mask bubbles, shape + typeset RTL text |
| `scripts/assemble_output.py` | Stage 5 — package as PDF / CBZ |
| `assets/fonts/Vazirmatn-*.ttf` | Persian font, 3 weights (SIL OFL, see `OFL.txt`) |
| `assets/glossary.template.json` | Naming consistency scaffold |
| `assets/worksheet.example.json` | Reference for the worksheet schema |

## What makes or breaks quality

1. **Persian text must be shaped.** Raw Persian drawn with PIL comes out as
   disconnected, reversed letters. `render_pages.py` runs everything through
   `arabic_reshaper` + the bidi algorithm. Never bypass it.
2. **Translate per page, not per bubble.** Bubbles are a conversation.
   Isolated translation produces wrong pronouns and disconnected dialogue.
3. **Keep a glossary.** Character names and court honorifics drifting mid-chapter
   is the most common failure. Copy `assets/glossary.template.json` and use it.
4. **Persian runs 15–25% longer than English.** Tighten the translation instead
   of letting autosizing shrink it to unreadable.
5. **Fix OCR damage from context.** All-caps comic lettering yields predictable
   errors (`WERSITE`→`WEBSITE`, `NORE`→`MORE`). Translate the meaning.

## Rights

Re-lettering a comic reproduces a copyrighted work in another language. Use this
only on material you have the right to translate — your own work, public-domain
titles, officially licensed work, or personal-use translation of a copy you own.

Scanlation credits and watermarks are auto-tagged `kind: "credit"` and left
untouched by default. Do not use `--include-credits` to strip another team's
attribution so the output looks original.

## Requirements

Alpine/iSH on iOS, or any Linux with: `poppler-utils`, `7zip`, `img2pdf`,
Python 3 with `Pillow`, `arabic-reshaper`, `python-bidi`, and `apple-vision`
for OCR. `setup.sh` installs and verifies all of it.

On non-Apple platforms substitute a different OCR engine in Stage 2 — keep the
`worksheet.json` output shape identical and the rest of the pipeline is unchanged.
