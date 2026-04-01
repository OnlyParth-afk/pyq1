"""
splitter.py — P2E Manual Page Splitter v4
══════════════════════════════════════════
• Renders ONE page at a time on demand — no upfront processing
• No page classifier — you decide which pages to skip
• 4-line workflow:
    ① BLUE   = top cut     (remove Marrow chrome)
    ② RED    = end of stem → question.png
    ③ ORANGE = end of options (options discarded, typed in sidebar)
    ④ PURPLE = end of explanation → explanation.png
• Sidebar: A/B/C/D text fields, double-click = correct answer
• MCQ ID + Topic fields
• Session fields (+) with presets

Usage:
    py splitter.py
    py splitter.py --pdf "D:\\path\\to\\file.pdf"  --port 5050
"""
from __future__ import annotations
import argparse, base64, io, json, os, pickle, sys, threading, time, webbrowser
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────
STATE: dict = {
    "pdf_path"   : "",
    "total"      : 0,
    "current"    : 0,
    "q_counter"  : 0,
    "img_dir"    : "",
    "splits_path": "",
    "splits"     : {},
    "dpi"        : 150,
    "page_cache" : {},
}

# ── Option extractor (handles all Marrow formats) ────────────────
# Lines that should never appear inside option text
_OPTION_STOP_RE = None
def _get_option_stop_re():
    global _OPTION_STOP_RE
    if _OPTION_STOP_RE is None:
        import re
        _OPTION_STOP_RE = re.compile(
            r"(\d+%\s+of\s+the\s+people"    # stats line
            r"|% of the people"
            r"|got this right"
            r"|MCQ ID"
            r"|MCQ\s*id"
            r"|Reference"
            r"|Ananthanarayan"
            r"|Paniker"
            r"|Jawetz"
            r"|Harrison"
            r"|Robbins"
            r"|Online\s+Resource"
            r"|http[s]?://"
            r"|Page\s+no"
            r"|Marrow\s+QBank"
            r"|NEET\s+PG"
            r"|Image\s+Attribution"
            r")",
            re.I
        )
    return _OPTION_STOP_RE


def _extract_options(text_block: str) -> dict:
    """
    Robustly extracts A/B/C/D from any Marrow option format:
      A. text   A) text   (A) text   A text
    Handles multi-line options, strips confidence [12%] / (12%).
    Stops before stats lines, references, or other non-option content.
    """
    import re
    stop_re = _get_option_stop_re()

    # Cut the text block at the first stop line
    # so the D option regex doesn't swallow the stats line
    clean_lines = []
    for line in text_block.splitlines():
        if stop_re.search(line):
            break           # stop here — everything below is not options
        clean_lines.append(line)
    text_block = "\n".join(clean_lines)

    pattern = r"(?im)^\s*\(?([A-D])\)?[\.\-]?\s+(.*?)(?=(?:^\s*\(?[A-D]\)?[\.\-]?\s+)|\Z)"
    matches  = re.findall(pattern, text_block, re.DOTALL)
    opts = {}
    for letter, text in matches:
        text = re.sub(r"\s+", " ", text.strip())
        # Strip confidence percentage — anywhere in text e.g. "Rickettsia [84%]" or "(21%)"
        text = re.sub(r"\s*[\[(]\d+%[\])]", "", text).strip()
        text = re.sub(r"\s*\(\d+%\)",         "", text).strip()
        # Strip correct/wrong icons inserted by Marrow UI (✅ U+2705, ❌ U+274C and variants)
        text = re.sub(r"[\u2705\u274c\u2713\u2717\u2611\u2612\u2714\u2718]", "", text).strip()
        # Strip trailing icon noise: any non-ASCII character at end of text
        text = re.sub(r"[^\x00-\x7F]+$", "", text).strip()
        # Strip topic tags
        text = re.sub(r"[©⊙●◉].+$", "", text, flags=re.DOTALL).strip()
        # Strip any remaining stop-pattern fragments
        if stop_re.search(text):
            text = stop_re.split(text)[0].strip()
        # Reject if still looks like UI noise or too short
        if text and len(text) >= 2 and text.lower() not in _UI_NOISE:
            opts[letter.upper()] = text
    return opts


# ── Tesseract path (Windows) ───────────────────────────────────────
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR	esseract.exe"
except ImportError:
    pass

# ── Auto-cut model (optional) ──────────────────────────────────────
_CUT_MODEL = None

def _extract_page_features(pil_img, bins=80):
    """
    4-channel feature vector (320 floats) — trained on 50 labeled Marrow pages.
    Channels: full brightness | left-strip brightness | left-strip R | left-strip G
    Each resampled to `bins` rows so page height doesn't matter.
    This is what the RF model in train_cutter.pkl (v2) expects.
    """
    import numpy as np
    arr = np.array(pil_img.convert('RGB'), dtype=np.float32) / 255.0
    H, W = arr.shape[:2]
    lW   = max(1, int(W * 0.22))
    idx  = np.linspace(0, H-1, bins).astype(int)
    return np.concatenate([
        arr.mean(axis=(1,2))[idx],
        arr[:, :lW, :].mean(axis=(1,2))[idx],
        arr[:, :lW, 0].mean(axis=1)[idx],
        arr[:, :lW, 1].mean(axis=1)[idx],
    ])

def _load_cut_model():
    global _CUT_MODEL
    candidates = [
        Path(__file__).parent / "train_cutter.pkl",
        Path.cwd() / "train_cutter.pkl",
        Path(r"D:\Study meteial\NEET PG QBANK\p2e\train_cutter.pkl"),
    ]
    model_path = next((c for c in candidates if c.exists()), None)
    if model_path is None:
        print("[splitter] No train_cutter.pkl found — manual line placement only")
        return
    try:
        import joblib
        _CUT_MODEL = joblib.load(model_path)
        print(f"[splitter] Model loaded: {_CUT_MODEL['n_train']} training pages, "
              f"features={_CUT_MODEL.get('fv','v1')}")
    except Exception as e:
        print(f"[splitter] Could not load model: {e}")





def _render_page_pil(page_idx: int, dpi: int = 150):
    """Render page to PIL Image, using cache when available."""
    import pypdfium2 as pdfium, base64, io
    from PIL import Image

    cached = STATE.get("page_cache", {}).get(page_idx)
    if cached:
        img_data = base64.b64decode(cached["img_b64"])
        pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        if dpi != 150:
            cw, ch = pil.size
            pil = pil.resize((int(cw * dpi/150), int(ch * dpi/150)), Image.LANCZOS)
    else:
        doc   = pdfium.PdfDocument(STATE["pdf_path"])
        bmp   = doc[page_idx].render(scale=dpi/72.0, rotation=0)
        pil   = bmp.to_pil().convert("RGB")
        doc.close()
    return pil


def _ocr_page(page_idx: int, dpi: int = 150):
    """
    Render page and run Tesseract image_to_data.
    Returns (pil, data_dict) where data_dict has keys:
      text, top, left, width, height, conf
    Returns (pil, None) if Tesseract not available.
    """
    try:
        import pytesseract
        pil  = _render_page_pil(page_idx, dpi=dpi)
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT,
                                          config="--psm 6")
        return pil, data
    except Exception as e:
        print(f"[ocr] _ocr_page failed: {e}")
        pil = _render_page_pil(page_idx, dpi=dpi)
        return pil, None


def _ocr_strip(pil, x0_frac: float = 0.0, x1_frac: float = 1.0,
               y0_frac: float = 0.0, y1_frac: float = 1.0,
               psm: int = 6) -> list[dict]:
    """
    OCR a fractional crop of a PIL image.
    Returns list of word dicts: [{text, top, left, bottom, conf}, ...]
    Only includes words with conf > 25.
    """
    try:
        import pytesseract
        W, H  = pil.size
        crop  = pil.crop((int(x0_frac*W), int(y0_frac*H),
                           int(x1_frac*W), int(y1_frac*H)))
        data  = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT,
                                           config=f"--psm {psm}")
        words = []
        for i, t in enumerate(data["text"]):
            if str(t).strip() and int(data["conf"][i]) > 25:
                words.append({
                    "text"  : str(t).strip(),
                    "left"  : data["left"][i],
                    "top"   : data["top"][i],
                    "bottom": data["top"][i] + data["height"][i],
                    "conf"  : int(data["conf"][i]),
                })
        return words
    except Exception as e:
        print(f"[ocr] _ocr_strip failed: {e}")
        return []


def _color_scan(page_idx: int) -> dict | None:
    """
    Detect line positions using Marrow's visual color markers.

    WHAT THE HUMAN EYE SEES:
      b1 = first content row (question text starts here)
      r1 = where the colored option circles begin (A/B/C/D dots)
      r2 = where the green progress bar ends ("62% got this right")
      b2 = where the explanation text ends

    COLOR SIGNATURES (confirmed from real pages):
      Stats bar  : wide green stripe, G-R>15, G-B>40, G>100, R>80
                   spans >8% of page width
      Option dots: small green OR red circles, left 20% of page
                   green = correct answer, red = wrong

    These markers are rendered as vector graphics — they exist even on
    full clinical image pages, outside the photo area. This is why
    color beats OCR for Marrow.
    """
    import numpy as np

    pil  = _render_page_pil(page_idx, dpi=150)
    arr  = np.array(pil, dtype=np.int32)
    H, W = arr.shape[:2]
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # ── Stats bar detection (r2) ───────────────────────────────────
    # Marrow green progress bar: yellowish-green, wide
    stats_green = (
        (G - R > 15) &
        (G - B > 40) &
        (G > 100)    &
        (R > 80)
    )
    green_per_row = stats_green.sum(axis=1)

    # Must span >8% of width to be the bar (not a small icon)
    BAR_THRESH = W * 0.08
    bar_rows = np.where(green_per_row > BAR_THRESH)[0]

    r2 = None
    if len(bar_rows) > 0:
        # Group into continuous bands
        bar_bands = []
        s = bar_rows[0]; p = bar_rows[0]
        for y in bar_rows[1:]:
            if y - p > 15:
                bar_bands.append((s, p))
                s = y
            p = y
        bar_bands.append((s, p))

        # The stats bar is typically in bottom 60% of the options area
        # (not at very top which would be header icons)
        valid_bars = [(s, e) for s, e in bar_bands if s / H > 0.02]
        if valid_bars:
            # Use the LOWEST stats bar on the page (some pages have pearl boxes with green)
            # Actually use the one just before the explanation section
            # Heuristic: rightmost/widest bar = stats bar
            widest = max(valid_bars,
                        key=lambda b: green_per_row[b[0]:b[1]+1].max())
            r2 = widest[1] / H + 0.004  # bottom of bar + margin
            print(f"[COLOR] r2: stats bar y={widest[0]}-{widest[1]} "
                  f"(frac {widest[0]/H:.4f}-{widest[1]/H:.4f}) → r2={r2:.4f}")

    # ── Option circle detection (r1) ───────────────────────────────
    # Option circles: green (correct) or red/orange (wrong), left margin only
    left_W = int(W * 0.20)  # only look in left 20%

    opt_green = (
        (G[:, :left_W] - R[:, :left_W] > 30) &
        (G[:, :left_W] > 100)
    )
    opt_red = (
        (R[:, :left_W] - G[:, :left_W] > 40) &
        (R[:, :left_W] > 120)
    )
    opt_colored = (opt_green | opt_red).sum(axis=1)  # per-row colored px count

    # Option circle rows: >3px colored in left strip
    circle_rows = np.where(opt_colored > 3)[0]

    r1 = None
    if len(circle_rows) > 0:
        # Group into bands (each band = one option circle)
        circle_bands = []
        s = circle_rows[0]; p = circle_rows[0]
        for y in circle_rows[1:]:
            if y - p > 20:
                circle_bands.append((s, p))
                s = y
            p = y
        circle_bands.append((s, p))

        # Filter: option circles are ABOVE r2 (if found) and after header
        HEADER_H = H * 0.025  # skip top 2.5% (Marrow logo)
        if r2 is not None:
            r2_px = r2 * H
            valid_circles = [(s, e) for s, e in circle_bands
                           if s > HEADER_H and s < r2_px]
        else:
            valid_circles = [(s, e) for s, e in circle_bands if s > HEADER_H]

        # Find a cluster of 3-4 bands within 15% of page height = A/B/C/D
        best_cluster_top = None
        for i, (s0, e0) in enumerate(valid_circles):
            window = H * 0.15
            cluster = [(s, e) for s, e in valid_circles
                      if s >= s0 and s <= s0 + window]
            if len(cluster) >= 3:  # found at least A/B/C or A/B/C/D
                best_cluster_top = cluster[0][0]
                break

        if best_cluster_top is not None:
            r1 = max(0.005, best_cluster_top / H - 0.008)
            print(f"[COLOR] r1: option cluster starts at y={best_cluster_top} "
                  f"(frac {best_cluster_top/H:.4f}) → r1={r1:.4f}")
        elif valid_circles:
            # Fallback: just use first circle
            r1 = max(0.005, valid_circles[0][0] / H - 0.008)
            print(f"[COLOR] r1 fallback: first circle y={valid_circles[0][0]} → r1={r1:.4f}")

    # ── b1: first content row (brightness) ────────────────────────
    brightness = arr.mean(axis=(1, 2)) / 255.0
    WHITE = 0.95
    # Skip header (top 2.5%), find first non-white row
    header_end = int(H * 0.025)
    non_white   = np.where(brightness[header_end:] < WHITE)[0]
    b1 = ((non_white[0] + header_end) / H - 0.005) if len(non_white) else 0.03
    b1 = max(0.005, b1)

    # ── b2: last content row (brightness) ─────────────────────────
    # Scan from BOTTOM UP. Cannot use top-down because reference section
    # has small book-cover thumbnails on wide white rows — their row-mean
    # brightness is > 0.95 (white), causing top-down scan to stop at the
    # gap just below the topic line (ORANGE=84%, PURPLE=85% = 1% crop).
    # Bottom-up with a lower threshold (0.88) catches faint text + thumbnails.
    r2_px      = int(r2 * H) if r2 else int(H * 0.4)
    DARK_THRESH = 0.88          # lower than WHITE to catch faint reference rows
    BOTTOM_MARGIN = int(H * 0.02)
    b2 = None
    for y in range(H - BOTTOM_MARGIN, r2_px, -1):
        if brightness[y] < DARK_THRESH:
            b2 = min(y / H + 0.010, 0.985)
            break
    if b2 is None:
        b2 = 0.92               # fallback: nothing found below r2

    # ── Fallbacks ─────────────────────────────────────────────────
    if r1 is None:
        print(f"[COLOR] WARNING: r1 not found — no option circles detected")
        r1 = 0.30  # will be corrected by Layer 2 (OCR/pdfplumber)
    if r2 is None:
        print(f"[COLOR] WARNING: r2 not found — no stats bar detected")
        r2 = r1 + 0.20

    # ── Clamp ─────────────────────────────────────────────────────
    b1 = max(0.005, min(b1, 0.12))
    r1 = max(b1 + 0.005, min(r1, 0.82))
    r2 = max(r1 + 0.015, min(r2, 0.91))
    b2 = max(r2 + 0.015, min(b2, 0.985))

    print(f"[COLOR] p{page_idx}: b1={b1:.4f} r1={r1:.4f} r2={r2:.4f} b2={b2:.4f}")
    return {"b1": b1, "r1": r1, "r2": r2, "b2": b2}


def _pdfplumber_anchors(page_idx: int) -> dict:
    """
    Extract anchor positions from PDF text layer.
    Returns dict with any of: r1, r2, b1, b2 — only keys that were found.
    """
    import re
    found = {}
    try:
        import pdfplumber
        with pdfplumber.open(STATE["pdf_path"]) as pdf:
            pg = pdf.pages[page_idx]
            PW, PH = float(pg.width), float(pg.height)
            words = pg.extract_words(x_tolerance=3, y_tolerance=3)

        def yn(y): return float(y) / PH
        def xn(x): return float(x) / PW

        pl = [{"text": w["text"].strip(), "top": yn(w["top"]),
               "bot": yn(w["bottom"]), "x0": xn(w["x0"])}
              for w in words if w["text"].strip()]

        if not pl:
            return found

        mcq_re  = re.compile(r"M[A-Z]\d{3,6}")
        hash_re = re.compile(r"^#\w{2,}")
        opt_re  = re.compile(r"^[A-D][.)]$")
        ref_kw  = ("reference", "ananthanarayan", "paniker", "harrison",
                   "robbins", "http", "www", "edition", "doi", "pubmed")

        # b1
        content = [w for w in pl if w["top"] > 0.025]
        if content:
            found["b1"] = max(0.005, content[0]["top"] - 0.006)

        # r2: MCQ ID line
        mcq_words = [w for w in pl if mcq_re.search(w["text"])]
        if mcq_words:
            lt = mcq_words[0]["top"]
            line = [w for w in pl if abs(w["top"] - lt) < 0.015]
            found["r2"] = max(w["bot"] for w in line) + 0.005

        # r2 fallback: hashtag
        if "r2" not in found:
            hw = [w for w in pl if hash_re.match(w["text"])]
            if hw:
                lt = min(w["top"] for w in hw)
                line = [w for w in pl if abs(w["top"] - lt) < 0.015]
                found["r2"] = max(w["bot"] for w in line) + 0.005

        # r1: A. at left margin
        r2_lim = found.get("r2", 1.0)
        opt_words = [w for w in pl if opt_re.match(w["text"])
                     and w["x0"] < 0.15 and w["top"] < r2_lim - 0.01]
        if opt_words:
            first = min(opt_words, key=lambda w: w["top"])
            found["r1"] = max(0.005, first["top"] - 0.006)

        # b2
        if "r2" in found:
            exp = [w for w in pl if w["top"] > found["r2"]
                   and not any(k in w["text"].lower() for k in ref_kw)
                   and not hash_re.match(w["text"])]
            if exp:
                found["b2"] = min(max(w["bot"] for w in exp) + 0.005, 0.985)

    except Exception as e:
        print(f"[PL] pdfplumber failed: {e}")
    return found


# ── Page type classifier ──────────────────────────────────────────────
_NON_MCQ_SIGNALS = [
    # Cover page signals
    "microbiology pyq", "marrow pyq", "neet pg pyq", "surgery pyq",
    "pharmacology pyq", "anatomy pyq", "physiology pyq", "pathology pyq",
    "forensic pyq", "biochemistry pyq", "ophthalmology pyq", "ent pyq",
    "pediatrics pyq", "psychiatry pyq", "obstetrics pyq", "gynecology pyq",
    "medicine pyq", "radiology pyq", "dermatology pyq", "orthopedics pyq",
    # Catalog page signals
    "data log", "total questions", "neet 2022", "neet 2021", "neet 2020",
    "neet 2019", "neet 2018", "neet 2017", "aiims nov", "aiims may",
    "inicet", "fmge",
    # Schema page signals
    "schema", "previous year question papers", "solve now", "bookmarks",
    "topics will be covered",
]

def classify_page(page_idx: int) -> str:
    """
    Classify a page as 'mcq' or 'skip'.
    'skip' = cover, catalog, schema, or other non-MCQ page.

    Detection logic:
    1. Text layer: look for MCQ anchors (A. at left margin + MCQ ID)
       → if found = mcq
    2. Text layer: look for non-MCQ signals (cover/catalog/schema keywords)
       → if found = skip
    3. Color scan: look for option circles (green/red dots left side)
       → if found = mcq
    4. Default = mcq (don't skip if uncertain)
    """
    import re

    # ── Try text layer first (fast) ───────────────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(STATE["pdf_path"]) as pdf:
            page = pdf.pages[page_idx]
            text = (page.extract_text() or "").lower()
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            PH = float(page.height)
            PW = float(page.width)

        # Strong MCQ signal: A. at left margin
        opt_re = re.compile(r"^[A-D][.)]$")
        has_option_letter = any(
            opt_re.match(w["text"].strip())
            and float(w["x0"]) / PW < 0.15
            for w in words
        )
        # Strong MCQ signal: MCQ ID pattern
        has_mcq_id = bool(re.search(r"M[A-Z]\d{3,6}", text))

        if has_option_letter or has_mcq_id:
            return "mcq"

        # Non-MCQ signals
        for signal in _NON_MCQ_SIGNALS:
            if signal in text:
                print(f"[classify] p{page_idx}: SKIP — matched '{signal}'")
                return "skip"

        # If text exists but no MCQ signals at all → likely schema/catalog
        word_count = len([w for w in words if w["text"].strip()])
        if word_count > 5 and not has_option_letter and not has_mcq_id:
            # Check if it looks like a topic list (many short lines, no option letters)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)
            if avg_line_len < 40 and len(lines) > 6:
                print(f"[classify] p{page_idx}: SKIP — looks like topic list "
                      f"({len(lines)} short lines, avg {avg_line_len:.0f} chars)")
                return "skip"

    except Exception as e:
        print(f"[classify] p{page_idx}: text layer failed: {e}")

    # ── Color scan fallback ───────────────────────────────────────
    try:
        import numpy as np
        pil  = _render_page_pil(page_idx, dpi=72)  # low res, just for color
        arr  = np.array(pil, dtype=np.int32)
        H, W = arr.shape[:2]
        R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        # Option circles: colored dots in left 20%
        left_W = int(W * 0.20)
        opt_green = ((G[:, :left_W] - R[:, :left_W] > 30) & (G[:, :left_W] > 100))
        opt_red   = ((R[:, :left_W] - G[:, :left_W] > 40) & (R[:, :left_W] > 120))
        colored_rows = ((opt_green | opt_red).sum(axis=1) > 2).sum()

        if colored_rows >= 3:
            return "mcq"

        # Stats bar (wide green): present on MCQ pages
        stats_green = ((G - R > 15) & (G - B > 40) & (G > 100) & (R > 80))
        has_stats_bar = (stats_green.sum(axis=1) > W * 0.08).any()
        if has_stats_bar:
            return "mcq"

    except Exception as e:
        print(f"[classify] p{page_idx}: color scan failed: {e}")

    # Default: don't skip (safe)
    print(f"[classify] p{page_idx}: defaulting to mcq (no skip signals found)")
    return "mcq"


def auto_skip_to_first_mcq():
    """
    On startup, advance STATE['current'] past any non-MCQ pages
    (cover, catalog, schema) to land on the first real MCQ page.
    Only runs if current == 0 (fresh start, not resuming).
    """
    if STATE["current"] != 0:
        return
    total = STATE["total"]
    MAX_PREAMBLE = min(20, total)  # never skip more than 20 pages

    skipped = []
    for i in range(MAX_PREAMBLE):
        ptype = classify_page(i)
        if ptype == "skip":
            skipped.append(i)
            STATE["current"] = i + 1
        else:
            break  # first MCQ found

    if skipped:
        print(f"[splitter] Auto-skipped {len(skipped)} preamble pages "
              f"(cover/catalog/schema): {skipped}")
        print(f"[splitter] Starting at page {STATE['current'] + 1}")


def _predict_lines(page_idx: int) -> dict | None:
    """
    Line detection that works like the human eye.

    The human eye finds sections by COLOR, not text:
      - Option circles (green/red dots) = where options start → r1
      - Green progress bar = where options end → r2
      - First non-white row = question starts → b1
      - Last non-white row = explanation ends → b2

    Pipeline:
      1. Color scan (fast, works on all page types including image pages)
      2. pdfplumber refinement (exact text coordinates to sharpen any anchor)
      3. Clamp and return

    Color scan is primary. pdfplumber overrides if it finds a more exact match.
    This approach never fails on Marrow pages because the colored UI elements
    (circles, progress bar) are always present regardless of page content.
    """
    # Step 1: Color scan — primary anchors
    result = _color_scan(page_idx)
    if result is None:
        result = {"b1": 0.025, "r1": 0.30, "r2": 0.55, "b2": 0.92}

    # Step 2: pdfplumber refinement — override with exact text coords when available
    pl = _pdfplumber_anchors(page_idx)
    for key, val in pl.items():
        # Guard: don't let pdfplumber push r2 past 0.75 — if MCQ ID is at
        # the very bottom of the page (image-heavy question), a high r2
        # collapses the explanation zone to near-zero.
        if key == "r2" and val > 0.75:
            print(f"[PL] r2={val:.4f} > 0.75 — skipping (image-heavy page, keeping color r2)")
            continue
        if key in result:
            old = result[key]
            result[key] = val
            print(f"[PL] refined {key}: {old:.4f} → {val:.4f} (pdfplumber exact)")
        else:
            result[key] = val

      # ── Layer 3: RF model fills gaps color/pdfplumber missed ─────────
    if _CUT_MODEL is not None:
        try:
            import numpy as np
            pil_lo = _render_page_pil(page_idx, dpi=72)
            feats  = _extract_page_features(pil_lo).reshape(1, -1)
            pred   = _CUT_MODEL['model'].predict(feats)[0]
            p_b1, p_r1, p_r2, p_b2 = [float(x) for x in pred]
            # Apply model where color scan is still at its raw fallback value
            if abs(result.get('r1', 0.30) - 0.30) < 0.015:
                result['r1'] = p_r1
                print(f"[RF] r1 → {p_r1:.4f}")
            if abs(result.get('r2', 0.55) - 0.55) < 0.015:
                result['r2'] = p_r2
                print(f"[RF] r2 → {p_r2:.4f}")
            if result.get('b2', 0.92) > 0.91:
                result['b2'] = p_b2
                print(f"[RF] b2 → {p_b2:.4f}")
        except Exception as e:
            print(f"[RF] predict failed: {e}")

    # Final clamp
    b1 = max(0.005, min(result.get("b1", 0.025), 0.12))
    r1 = max(b1 + 0.005, min(result.get("r1", 0.30), 0.82))
    r2 = max(r1 + 0.015, min(result.get("r2", r1+0.20), 0.91))
    b2 = max(r2 + 0.015, min(result.get("b2", 0.92), 0.985))

    print(f"[FINAL] p{page_idx}: b1={b1:.4f} r1={r1:.4f} r2={r2:.4f} b2={b2:.4f}")
    return {"b1": b1, "r1": r1, "r2": r2, "b2": b2}



def render_page(page_idx: int) -> dict:
    """Render page and cache it. Returns {img_b64, width, height}."""
    if page_idx in STATE["page_cache"]:
        return STATE["page_cache"][page_idx]

    import pypdfium2 as pdfium
    pdf   = pdfium.PdfDocument(STATE["pdf_path"])
    scale = STATE["dpi"] / 72
    bmp   = pdf[page_idx].render(scale=scale, rotation=0)
    pil   = bmp.to_pil()
    pdf.close()

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = {"img_b64": b64, "width": pil.width, "height": pil.height}
    STATE["page_cache"][page_idx] = result
    return result


SAVE_DPI = 450  # 3x sharpness for saved PNGs (preview stays 150)

def render_page_hires(page_idx: int) -> dict:
    """Render at SAVE_DPI for cropping."""
    import pypdfium2 as pdfium
    pdf   = pdfium.PdfDocument(STATE["pdf_path"])
    scale = SAVE_DPI / 72
    bmp   = pdf[page_idx].render(scale=scale, rotation=0)
    pil   = bmp.to_pil()
    pdf.close()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"img_b64": b64, "width": pil.width, "height": pil.height}


# UI strings that pdfplumber picks up from PDF overlay buttons — must be excluded
_UI_NOISE = {
    "next", "back", "skip", "save", "save & next", "append next page",
    "skip page", "undo last line", "go", "resume", "jump", "load",
    "marrow", "marrow qbank", "qbank", "neet pg", "neet", "pg",
}

def _clean_pdf_text(raw_text: str) -> list[str]:
    """
    Split raw pdfplumber text into clean lines, removing:
      - Navigation UI strings (NEXT, BACK, SAVE, etc.)
      - Stats line ("X% of the people got this right")
      - Empty lines
      - Lines that are purely symbols/numbers with no letter content
    Returns list of cleaned lines — safe to parse for options/MCQ/topic.
    """
    import re
    stats_re = re.compile(r"\d+%\s+of\s+the\s+people", re.I)
    pure_sym  = re.compile(r"^[\d\s%©⊙●◉|\-–—•.,:;!?()\[\]{}]+$")

    lines = []
    for raw in raw_text.splitlines():
        l = raw.strip()
        if not l:
            continue
        # Drop UI noise (case-insensitive exact match)
        if l.lower() in _UI_NOISE:
            continue
        # Drop stats line
        if stats_re.search(l):
            continue
        # Drop lines that are purely punctuation/symbols/numbers
        if pure_sym.match(l):
            continue
        lines.append(l)
    return lines


def extract_page_text(page_idx: int) -> dict:
    """Parse options / MCQ ID / topic from PDF text layer."""
    import re
    try:
        import pdfplumber
        with pdfplumber.open(STATE["pdf_path"]) as pdf:
            text = pdf.pages[page_idx].extract_text() or ""
    except Exception:
        return {}

    lines = _clean_pdf_text(text)

    # Topic tag — FIRST: find ⊙/©/● tagged lines (they are the topic)
    # Must do this BEFORE option parsing so we can exclude these lines
    topic_lines = set()
    topic = ""
    for i, l in enumerate(lines):
        if re.search(r'[©⊙●◉⊙]', l):
            clean = re.sub(r'^[©⊙●◉⊙,\s]+', '', l).strip()
            # Remove confidence percentage if present
            clean = re.sub(r'\s*\[\d+%\].*$', '', clean).strip()
            if clean and len(clean) > 4 and not re.match(r'^[A-D][.)]\s', clean):
                topic = clean
                topic_lines.add(i)

    # Options A/B/C/D — use robust extractor
    # Exclude topic-tagged lines from the text first
    clean_lines = [l for i, l in enumerate(lines) if i not in topic_lines]
    options = _extract_options("\n".join(clean_lines))

    # MCQ ID — e.g. MF7446, MC3178
    mcq_id = ""
    mid = re.search(r'(M[A-Z]\d{3,6})', text)
    if mid:
        mcq_id = mid.group(1)

    # If any options missing from PDF text layer → try OCR
    missing = [l for l in "ABCD" if l not in options]
    if missing:
        try:
            import re as re2
            pil, ocr = _ocr_page(page_idx)
            if ocr and pil:
                W2, H2 = pil.size
                ocr_texts   = ocr["text"]
                ocr_tops    = ocr["top"]
                ocr_lefts   = ocr["left"]
                ocr_heights = ocr["height"]
                ocr_confs   = ocr["conf"]

                # Build word list with positions
                ocr_words = [
                    {"text": t.strip(), "top": ocr_tops[i],
                     "left": ocr_lefts[i],
                     "bottom": ocr_tops[i] + ocr_heights[i],
                     "conf": int(ocr_confs[i])}
                    for i, t in enumerate(ocr_texts)
                    if str(t).strip() and int(ocr_confs[i]) > 25
                ]

                # Find image end (largest vertical gap)
                all_tops2 = sorted(set(w["top"] for w in ocr_words))
                image_end_y2 = 0
                if len(all_tops2) > 3:
                    gaps2 = [(all_tops2[i+1] - all_tops2[i], all_tops2[i+1])
                             for i in range(len(all_tops2)-1)]
                    bg = max(gaps2, key=lambda g: g[0])
                    if bg[0] > H2 * 0.06:
                        image_end_y2 = bg[1]

                # Only look for options BELOW the image
                text_words = [w for w in ocr_words if w["top"] >= image_end_y2]

                # Group text_words into lines by proximity
                line_groups = {}
                for w in text_words:
                    bucket = w["top"] // 20   # group within 20px
                    line_groups.setdefault(bucket, []).append(w)

                # Reconstruct text lines sorted by y then x
                # Reconstruct full text from OCR lines and use robust extractor
                ocr_line_texts = []
                for bucket in sorted(line_groups.keys()):
                    ws = sorted(line_groups[bucket], key=lambda w: w["left"])
                    ocr_line_texts.append(" ".join(w["text"] for w in ws).strip())
                ocr_block = "\n".join(ocr_line_texts)
                ocr_opts = _extract_options(ocr_block)
                for ltr in missing:
                    if ltr in ocr_opts:
                        options[ltr] = ocr_opts[ltr]
                missing = [l for l in "ABCD" if l not in options]
        except Exception as e:
            print(f"[splitter] OCR options fallback failed: {e}")

    # ── OCR fallback for MCQ ID on image pages ──────────────────────
    # If no MCQ ID found in text layer, OCR a narrow strip in the center
    # of the page — MCQ ID ("MCQ ID: MFxxxx") always appears there.
    if not mcq_id:
        try:
            import numpy as np
            pil = _render_page_pil(page_idx, dpi=200)
            W2, H2 = pil.size

            # MCQ ID strip: center 60% width, search entire page in bands
            # Use color to locate the orange divider line (r2) first
            arr = np.array(pil, dtype=np.int32)
            R2, G2, B2 = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            # Orange line: R high, G medium, B low
            orange = (R2 > 180) & (G2 > 80) & (G2 < 180) & (B2 < 80)
            orange_rows = np.where(orange.sum(axis=1) > W2 * 0.3)[0]

            if len(orange_rows):
                # MCQ ID is just ABOVE the orange line, within ~8% of page height
                orange_top = orange_rows[0]
                strip_y0 = max(0, orange_top - int(H2 * 0.08))
                strip_y1 = min(H2, orange_top + int(H2 * 0.02))
            else:
                # No orange line found — search middle 60% of page
                strip_y0 = int(H2 * 0.20)
                strip_y1 = int(H2 * 0.80)

            ocr_words = _ocr_strip(pil,
                                   x0_frac=0.20, x1_frac=0.85,
                                   y0_frac=strip_y0/H2,
                                   y1_frac=strip_y1/H2,
                                   psm=6)
            full_text = " ".join(w["text"] for w in ocr_words)
            mid2 = re.search(r'M[A-Z]\d{3,6}', full_text)
            if mid2:
                mcq_id = mid2.group(0)
                print(f"[extract] MCQ ID from OCR strip: {mcq_id}")
        except Exception as e:
            print(f"[extract] OCR MCQ ID fallback failed: {e}")

    return {"options": options, "mcq_id": mcq_id, "topic": topic}


# ── Crop and save PNGs ───────────────────────────────────────────
def save_crops(page_idx: int, q_num: int,
               b1: float, r1: float, r2: float, b2: float,
               appended_b64s: list):
    import numpy as np, cv2

    # Render at 3x DPI for sharp PNGs
    pg   = render_page_hires(page_idx)
    w, h = pg["width"], pg["height"]

    def decode(b64, target_w=None):
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if target_w and img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, int(img.shape[0] * target_w / img.shape[1])))
        return img

    main   = decode(pg["img_b64"])
    b1_px  = max(2,        int(h * b1))
    r1_px  = max(b1_px+2,  int(h * r1))
    r2_px  = max(r1_px+2,  int(h * r2))
    b2_px  = max(r2_px+2,  min(int(h * b2), h))

    key      = f"q{q_num:03d}"
    q_crop   = main[b1_px:r1_px]
    exp_parts = [main[r2_px:b2_px]]
    for b64 in appended_b64s:
        exp_parts.append(decode(b64, w))
    exp_crop = np.vstack(exp_parts)

    def _save(arr, path):
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if ok:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.write(buf.tobytes())

    _save(q_crop,   os.path.join(STATE["img_dir"], f"{key}_question.png"))
    _save(exp_crop, os.path.join(STATE["img_dir"], f"{key}_explanation.png"))


# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index(): return HTML

@app.route("/api/state")
def api_state():
    cur   = STATE["current"]
    total = STATE["total"]
    if total == 0:
        return jsonify({"error": "No PDF loaded"}), 500
    if cur >= total:
        return jsonify({"done": True, "total": total,
                        "q_saved": STATE["q_counter"]})
    try:
        pg = render_page(cur)
    except Exception as e:
        return jsonify({"error": f"Render failed p{cur}: {e}"}), 500

    # Next page b64 for append (render quietly)
    next_b64 = None
    if cur + 1 < total:
        try:
            next_b64 = render_page(cur + 1)["img_b64"]
        except Exception:
            pass

    saved = STATE["splits"].get(str(cur))
    prediction = None if saved else _predict_lines(cur)
    return jsonify({
        "done"      : False,
        "page_idx"  : cur,
        "page_num"  : cur + 1,        # 1-based for display
        "total"     : total,
        "q_counter" : STATE["q_counter"],
        "img_b64"   : pg["img_b64"],
        "width"     : pg["width"],
        "height"    : pg["height"],
        "saved"     : saved,
        "next_b64"  : next_b64,
        "stem"      : Path(STATE["pdf_path"]).stem,
        "prediction": prediction,     # {b1,r1,r2,b2} or null
        "page_type" : classify_page(cur),  # "mcq" or "skip"
    })

@app.route("/api/extract/<int:page_idx>")
def api_extract(page_idx):
    return jsonify(extract_page_text(page_idx))

@app.route("/api/save", methods=["POST"])
def api_save():
    d       = request.json
    cur     = STATE["current"]
    q_num   = STATE["q_counter"] + 1
    STATE["q_counter"] = q_num

    save_crops(cur, q_num, d["b1"], d["r1"], d["r2"], d["b2"],
               d.get("appended_b64s", []))

    STATE["splits"][str(cur)] = {
        "page_idx"   : cur,
        "q_num"      : q_num,
        "b1": d["b1"], "r1": d["r1"], "r2": d["r2"], "b2": d["b2"],
        "options"    : d.get("options", {}),
        "answer"     : d.get("answer", ""),
        "mcq_id"     : d.get("mcq_id", ""),
        "topic"      : d.get("topic", ""),
        "session_meta": d.get("session_meta", {}),
        "appended"   : len(d.get("appended_b64s", [])),
    }
    _write_splits()
    STATE["current"] += 1
    return jsonify({"ok": True, "q_num": q_num})

@app.route("/api/unsaved_pages")
def api_unsaved_pages():
    """Return first unsaved page index and count of unsaved pages."""
    total  = STATE["total"]
    saved  = set(int(k) for k in STATE["splits"].keys())
    unsaved = [i for i in range(total) if i not in saved]
    first  = unsaved[0] if unsaved else None
    return jsonify({"first_unsaved": first, "count": len(unsaved), "total": total})

@app.route("/api/skip", methods=["POST"])
def api_skip():
    STATE["current"] += 1
    return jsonify({"ok": True})

@app.route("/api/back", methods=["POST"])
def api_back():
    if STATE["current"] > 0:
        STATE["current"] -= 1
        # If we went back over a saved page, restore q_counter
        # (find highest q_num in saves up to current)
        cur = STATE["current"]
        saved_before = [v["q_num"] for k, v in STATE["splits"].items()
                        if int(k) < cur and "q_num" in v]
        STATE["q_counter"] = max(saved_before) if saved_before else 0
    return jsonify({"ok": True})

@app.route("/api/jump", methods=["POST"])
def api_jump():
    idx = int(request.json.get("page", 1)) - 1  # 1-based input
    idx = max(0, min(idx, STATE["total"] - 1))
    STATE["current"] = idx
    saved_before = [v["q_num"] for k, v in STATE["splits"].items()
                    if int(k) < idx and "q_num" in v]
    STATE["q_counter"] = max(saved_before) if saved_before else 0
    return jsonify({"ok": True})


# ── Auto-process background job state ────────────────────────────────
_AUTO = {
    "running"  : False,
    "done"     : False,
    "total"    : 0,
    "processed": 0,
    "skipped"  : 0,
    "current_page": 0,
    "errors"   : [],
    "message"  : "",
}

def _auto_worker(todo, dry_run):
    """Background thread: process pages one by one, update _AUTO dict."""
    import time
    _AUTO["running"]   = True
    _AUTO["done"]      = False
    _AUTO["total"]     = len(todo)
    _AUTO["processed"] = 0
    _AUTO["skipped"]   = 0
    _AUTO["errors"]    = []
    _AUTO["message"]   = f"Processing {len(todo)} pages…"

    for page_idx in todo:
        if not _AUTO["running"]:   # allow cancel
            _AUTO["message"] = "Cancelled"
            break
        _AUTO["current_page"] = page_idx + 1
        try:
            pred = _predict_lines(page_idx)
            if pred is None:
                _AUTO["skipped"] += 1
                continue

            b1, r1, r2, b2 = pred["b1"], pred["r1"], pred["r2"], pred["b2"]
            text_data = extract_page_text(page_idx)
            options   = text_data.get("options", {})
            mcq_id    = text_data.get("mcq_id", "")
            topic     = text_data.get("topic", "")

            if dry_run:
                _AUTO["processed"] += 1
                continue

            q_num = STATE["q_counter"] + 1
            STATE["q_counter"] = q_num
            save_crops(page_idx, q_num, b1, r1, r2, b2, appended_b64s=[])
            STATE["splits"][str(page_idx)] = {
                "page_idx": page_idx, "q_num": q_num,
                "b1": b1, "r1": r1, "r2": r2, "b2": b2,
                "options": options, "answer": "",
                "mcq_id": mcq_id, "topic": topic,
                "session_meta": {}, "appended": 0, "auto": True,
            }
            STATE["current"] = page_idx + 1
            _write_splits()
            _AUTO["processed"] += 1
            print(f"[auto] p{page_idx+1}: Q{q_num} id={mcq_id} opts={list(options.keys())}")

        except Exception as e:
            print(f"[auto] p{page_idx+1}: ERROR — {e}")
            _AUTO["errors"].append({"page": page_idx+1, "error": str(e)})
            _AUTO["skipped"] += 1

    _AUTO["running"] = False
    _AUTO["done"]    = True
    _AUTO["message"] = (f"Done — {_AUTO['processed']} saved, "
                        f"{_AUTO['skipped']} skipped")
    print(f"[auto] {_AUTO['message']}")


@app.route("/api/auto_process", methods=["POST"])
def api_auto_process():
    """
    Start background auto-processing. Returns immediately.
    Poll /api/auto_status for progress.
    POST body: { "dry_run": false, "cancel": false }
    """
    import threading
    data = request.json or {}

    # Cancel request
    if data.get("cancel"):
        _AUTO["running"] = False
        return jsonify({"ok": True, "message": "Cancel requested"})

    # Already running
    if _AUTO["running"]:
        return jsonify({"ok": False, "message": "Already running"}), 409

    dry_run    = data.get("dry_run", False)
    total      = STATE["total"]
    saved_keys = set(STATE["splits"].keys())

    todo = []
    for i in range(total):
        if str(i) in saved_keys:
            continue
        try:
            ptype = classify_page(i)
        except Exception:
            ptype = "mcq"
        if ptype == "skip":
            continue
        todo.append(i)

    if not todo:
        return jsonify({"ok": True, "processed": 0, "skipped": 0,
                        "done": True,
                        "message": "Nothing to process — all pages already done"})

    t = threading.Thread(target=_auto_worker, args=(todo, dry_run), daemon=True)
    t.start()
    return jsonify({"ok": True, "queued": len(todo),
                    "message": f"Started — {len(todo)} pages queued"})


@app.route("/api/auto_status")
def api_auto_status():
    """Poll this every second to get auto-process progress."""
    total_pdf = STATE["total"]
    saved     = len(STATE["splits"])
    done_pgs  = _AUTO["processed"]
    todo_tot  = _AUTO["total"] if _AUTO["total"] > 0 else 1
    pct       = round(done_pgs / todo_tot * 100) if _AUTO["total"] else 0
    return jsonify({
        "running"     : _AUTO["running"],
        "done"        : _AUTO["done"],
        "processed"   : _AUTO["processed"],
        "skipped"     : _AUTO["skipped"],
        "total_todo"  : _AUTO["total"],
        "total_pdf"   : total_pdf,
        "saved"       : saved,
        "current_page": _AUTO["current_page"],
        "pct"         : pct,
        "errors"      : _AUTO["errors"][-5:],  # last 5 errors
        "message"     : _AUTO["message"],
    })


def _write_splits():
    with open(STATE["splits_path"], "w", encoding="utf-8") as f:
        json.dump(STATE["splits"], f, indent=2)


# ── HTML ─────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P2E Splitter</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--panel:#0d1117;--border:#1c2333;
  --b1:#38bdf8;--r1:#f87171;--r2:#fb923c;--b2:#a78bfa;
  --green:#4ade80;--amber:#fbbf24;--text:#e2e8f0;--muted:#4b5563;
}
body{background:var(--bg);color:var(--text);
     font-family:'SF Mono','Cascadia Code','Fira Code',monospace;
     height:100vh;display:flex;flex-direction:column;overflow:hidden;font-size:13px}

/* header */
header{background:var(--panel);border-bottom:2px solid var(--border);
       padding:8px 14px;display:flex;align-items:center;gap:10px;flex-shrink:0;user-select:none}
.logo{color:var(--green);font-weight:800;letter-spacing:3px;font-size:11px;white-space:nowrap}
.q-badge{font-size:11px;font-weight:800;color:#000;background:var(--green);
         padding:2px 8px;border-radius:4px;letter-spacing:1px;min-width:38px;text-align:center}
.prog-outer{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden;min-width:40px}
.prog-inner{height:100%;background:var(--green);border-radius:2px;transition:width .3s}
.counter{font-size:11px;color:var(--muted);white-space:nowrap}
.stem-badge{font-size:10px;color:var(--amber);background:#1a1200;
            padding:2px 8px;border-radius:4px;border:1px solid #3a2800;white-space:nowrap;
            max-width:260px;overflow:hidden;text-overflow:ellipsis}

/* layout */
.body-wrap{flex:1;display:flex;overflow:hidden;min-height:0}

/* scroll */
.scroll{flex:1;overflow-y:auto;overflow-x:hidden;background:#030508;
        display:flex;flex-direction:column;align-items:center;padding:20px 0}

/* page */
.page-wrap{position:relative;display:inline-block;cursor:crosshair;
           max-width:700px;width:95%}
.page-wrap img{display:block;width:100%;height:auto;pointer-events:none;
               user-select:none;box-shadow:0 4px 40px #000c;border-radius:2px}
#appended-wrap{width:95%;max-width:700px}
.appended-img{display:block;width:100%;height:auto;pointer-events:none;
              user-select:none;border-top:3px dashed var(--amber)}

/* lines */
.line{position:absolute;left:0;right:0;height:3px;pointer-events:none;z-index:10}
.lb1{background:var(--b1);box-shadow:0 0 8px var(--b1)}
.lr1{background:var(--r1);box-shadow:0 0 8px var(--r1)}
.lr2{background:var(--r2);box-shadow:0 0 8px var(--r2)}
.lb2{background:var(--b2);box-shadow:0 0 8px var(--b2)}
.line-lbl{position:absolute;right:6px;top:-17px;font-size:10px;
          padding:1px 7px;border-radius:3px;font-weight:700;letter-spacing:1px;white-space:nowrap}
.lb1 .line-lbl{color:var(--b1);background:#020b14}
.lr1 .line-lbl{color:var(--r1);background:#140404}
.lr2 .line-lbl{color:var(--r2);background:#140800}
.lb2 .line-lbl{color:var(--b2);background:#0c0814}

/* tints */
.tint{position:absolute;left:0;right:0;pointer-events:none;z-index:5}
.tq  {background:rgba(56,189,248,.08)}
.topt{background:rgba(251,146,60,.06)}
.texp{background:rgba(167,139,250,.08)}

/* region labels */
.rlabel{position:absolute;left:10px;font-size:10px;font-weight:800;letter-spacing:2px;
        padding:2px 10px;border-radius:20px;pointer-events:none;z-index:12;white-space:nowrap}
.rq  {background:#052e16cc;color:var(--green);border:1px solid #4ade8033}
.ropt{background:#1a0e00cc;color:var(--r2);   border:1px solid #fb923c33}
.rexp{background:#160e2ecc;color:var(--b2);   border:1px solid #a78bfa33}

.hover-guide{position:absolute;left:0;right:0;height:1px;background:#ffffff15;
             pointer-events:none;z-index:8;display:none}

/* hint */
.hint-overlay{position:absolute;inset:0;display:flex;align-items:center;
              justify-content:center;pointer-events:none;z-index:15}
.hint-box{background:#0d111799;border:1px dashed var(--border);color:var(--muted);
          font-size:11px;letter-spacing:1px;padding:10px 20px;border-radius:6px;
          text-align:center;line-height:2.3}

/* loading spinner */
.spinner{display:none;position:absolute;inset:0;align-items:center;
         justify-content:center;z-index:20;pointer-events:none}
.spinner.show{display:flex}
.spin-ring{width:36px;height:36px;border:3px solid var(--border);
           border-top-color:var(--green);border-radius:50%;
           animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* sidebar */
.sidebar{width:255px;flex-shrink:0;background:var(--panel);
         border-left:2px solid var(--border);
         display:flex;flex-direction:column;padding:12px;gap:0;overflow-y:auto}
.sb-section{border-top:1px solid var(--border);padding-top:10px;margin-top:10px}
.sb-section:first-child{border-top:none;padding-top:0;margin-top:0}
.sb-h{font-size:10px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;
      margin-bottom:8px;display:flex;align-items:center;justify-content:space-between}

/* line legend */
.leg-row{display:flex;align-items:center;gap:8px;font-size:11px;margin-bottom:5px}
.leg-dot{width:14px;height:3px;border-radius:2px;flex-shrink:0}
.leg-active{font-weight:800 !important}

/* info rows */
.ir{display:flex;justify-content:space-between;align-items:center;
    padding:4px 0;border-bottom:1px solid var(--border);font-size:11px}
.ir .l{color:var(--muted)} .ir .v{font-weight:700}

/* jump */
.jump-row{display:flex;gap:5px;margin-top:8px}
.jump-row input{flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);
                padding:4px 6px;border-radius:4px;font-family:inherit;font-size:11px;min-width:0}
.jump-lbl{font-size:9px;color:var(--muted);margin-top:3px;letter-spacing:0.5px}

/* session fields */
.meta-add-btn{background:transparent;border:1px solid var(--border);color:var(--green);
              width:20px;height:20px;border-radius:4px;cursor:pointer;font-size:15px;
              font-weight:700;line-height:1;padding:0;letter-spacing:0;width:auto;
              display:flex;align-items:center;justify-content:center;padding:0 6px}
.meta-add-btn:hover{background:var(--green);color:#000;border-color:var(--green)}
.mfr{display:flex;align-items:center;gap:5px;margin-bottom:5px}
.mfk{font-size:10px;color:var(--muted);white-space:nowrap;min-width:46px;
     text-transform:uppercase;letter-spacing:0.8px}
.mfv{flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);
     padding:4px 6px;border-radius:4px;font-family:inherit;font-size:11px;
     outline:none;min-width:0}
.mfv:focus{border-color:var(--b1)}
.mfd{background:transparent;border:none;color:var(--muted);cursor:pointer;font-size:14px;padding:0 2px}
.mfd:hover{color:#ef4444}

/* popup */
.popup-wrap{position:relative}
.add-popup{display:none;position:absolute;right:0;top:24px;z-index:50;
           background:var(--panel);border:1px solid var(--border);
           border-radius:6px;padding:7px;min-width:155px;box-shadow:0 4px 20px #000c}
.add-popup.open{display:block}
.pb{display:block;width:100%;text-align:left;padding:5px 7px;background:transparent;
    border:none;color:var(--text);font-family:inherit;font-size:11px;cursor:pointer;
    border-radius:3px;letter-spacing:0.3px}
.pb:hover{background:var(--border)}
.pdiv{height:1px;background:var(--border);margin:4px 0}
.prow{display:flex;gap:4px;margin-top:4px}
.prow input{flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);
            padding:4px 5px;border-radius:4px;font-family:inherit;font-size:10px;min-width:0}
.prow button{flex:0 0 auto;padding:4px 7px;font-size:10px;width:auto}

/* options */
.opt-row{display:flex;align-items:flex-start;gap:6px;margin-bottom:6px}
.opt-letter{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:11px;font-weight:800;flex-shrink:0;
            cursor:pointer;border:2px solid var(--border);color:var(--muted);
            transition:all .15s;user-select:none;margin-top:2px}
.opt-letter:hover{border-color:var(--green);color:var(--green)}
.opt-letter.correct{background:var(--green);color:#000;border-color:var(--green);
                    box-shadow:0 0 8px var(--green)}
.opt-letter.wrong{background:#7f1d1d;color:#fca5a5;border-color:#ef4444;
                  box-shadow:0 0 6px #ef444466}
.opt-input{flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);
           padding:4px 7px;border-radius:4px;font-family:inherit;font-size:11px;
           resize:vertical;min-height:24px;line-height:1.4;outline:none}
.opt-input:focus{border-color:var(--b1)}
.opt-input.correct-input{border-color:var(--green) !important;background:#052e1633}
.opt-input.wrong-input{border-color:#ef4444 !important;background:#1a000033}

/* meta fields */
.meta-row{display:flex;align-items:flex-start;gap:5px;margin-bottom:6px}
.meta-badge{width:28px;height:22px;border-radius:3px;display:flex;align-items:center;
            justify-content:center;font-size:9px;font-weight:800;letter-spacing:1px;
            flex-shrink:0;cursor:default}
.badge-id{color:var(--amber);border:1px solid var(--amber);background:transparent}
.badge-tag{color:var(--b2);   border:1px solid var(--b2);  background:transparent}
.meta-input{flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);
            padding:4px 7px;border-radius:4px;font-family:inherit;font-size:11px;
            outline:none;min-width:0}
.meta-input:focus{border-color:var(--b1)}
textarea.meta-input{resize:vertical;min-height:28px;line-height:1.4}

/* kb */
.kb{font-size:10px;color:var(--muted);line-height:2}
.kb b{color:var(--green)}

/* buttons */
.btns{margin-top:auto;padding-top:10px;display:flex;flex-direction:column;gap:6px}
button{padding:7px 10px;border:2px solid var(--border);border-radius:5px;
       background:transparent;color:var(--text);font-family:inherit;font-size:11px;
       font-weight:700;cursor:pointer;letter-spacing:1px;transition:all .15s;width:100%}
button:hover:not(:disabled){filter:brightness(1.2)}
button.primary{background:var(--green);color:#000;border-color:var(--green)}
button.amber{border-color:var(--amber);color:var(--amber)}
button.danger{border-color:#ef4444;color:#ef4444}
button:disabled{opacity:.25;cursor:not-allowed}

/* done overlay */
#done-overlay{display:none;position:fixed;inset:0;background:var(--bg);z-index:200;
              align-items:center;justify-content:center;flex-direction:column;gap:14px}
#done-overlay h2{font-size:28px;color:var(--green)}
#done-overlay p{color:var(--muted);font-size:12px}

/* toast */
.toast{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);
       background:var(--panel);border:2px solid var(--green);color:var(--green);
       padding:5px 16px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:1px;
       opacity:0;transition:opacity .25s;pointer-events:none;z-index:999;white-space:nowrap}
.toast.show{opacity:1}
</style>
</head>
<body>

<header>
  <div class="logo">P2E SPLITTER</div>
  <div class="q-badge" id="q-badge">Q—</div>
  <div class="prog-outer"><div class="prog-inner" id="prog-bar" style="width:0%"></div></div>
  <div class="counter" id="counter">—</div>
  <div class="stem-badge" id="stem-badge">—</div>
</header>

<div class="body-wrap">

  <!-- scroll + canvas -->
  <div class="scroll" id="scroll">
    <div class="page-wrap" id="page-wrap">
      <img id="page-img" src="" alt="">

      <!-- loading spinner -->
      <div class="spinner" id="spinner"><div class="spin-ring"></div></div>

      <!-- tints -->
      <div class="tint tq"   id="tq"   style="display:none"></div>
      <div class="tint topt" id="topt" style="display:none"></div>
      <div class="tint texp" id="texp" style="display:none"></div>

      <!-- 4 lines -->
      <div class="line lb1" id="lb1" style="display:none"><div class="line-lbl" id="lbl-b1">① BLUE 0%</div></div>
      <div class="line lr1" id="lr1" style="display:none"><div class="line-lbl" id="lbl-r1">② RED 0%</div></div>
      <div class="line lr2" id="lr2" style="display:none"><div class="line-lbl" id="lbl-r2">③ ORANGE 0%</div></div>
      <div class="line lb2" id="lb2" style="display:none"><div class="line-lbl" id="lbl-b2">④ PURPLE 0%</div></div>

      <!-- region labels -->
      <div class="rlabel rq"   id="rq"   style="display:none">QUESTION</div>
      <div class="rlabel ropt" id="ropt" style="display:none">OPTIONS</div>
      <div class="rlabel rexp" id="rexp" style="display:none">EXPLANATION</div>

      <div class="hover-guide" id="hover-guide"></div>

      <div class="hint-overlay" id="hint-overlay">
        <div class="hint-box">
          Click 4 lines in order<br>
          <span style="color:var(--b1)">① BLUE</span> top cut &nbsp;
          <span style="color:var(--r1)">② RED</span> end of stem<br>
          <span style="color:var(--r2)">③ ORANGE</span> end of options &nbsp;
          <span style="color:var(--b2)">④ PURPLE</span> end of explanation
        </div>
      </div>
    </div>
    <div id="appended-wrap"></div>
  </div>

  <!-- sidebar -->
  <div class="sidebar">

    <!-- Line status -->
    <div class="sb-section">
      <div class="sb-h">Lines</div>
      <div class="leg-row" id="leg0"><div class="leg-dot" style="background:var(--b1)"></div><span>① BLUE — top cut</span></div>
      <div class="leg-row" id="leg1"><div class="leg-dot" style="background:var(--r1)"></div><span>② RED — end of stem</span></div>
      <div class="leg-row" id="leg2"><div class="leg-dot" style="background:var(--r2)"></div><span>③ ORANGE — end of options</span></div>
      <div class="leg-row" id="leg3"><div class="leg-dot" style="background:var(--b2)"></div><span>④ PURPLE — end of explanation</span></div>
    </div>

    <!-- Page info -->
    <div class="sb-section">
      <div class="ir"><span class="l">PDF page</span><span class="v" id="i-pg">—</span></div>
      <div class="ir"><span class="l">Q saved</span><span class="v" id="i-qn">—</span></div>
      <div class="ir"><span class="l">Next line</span><span class="v" id="i-next">—</span></div>
    </div>

    <!-- Jump -->
    <div class="sb-section">
      <div class="jump-row">
        <input id="j-page" type="number" min="1" placeholder="Page #">
        <button onclick="jumpToPage()">GO</button>
      </div>
      <div class="jump-lbl">jump to PDF page number</div>
    </div>

    <!-- Session fields -->
    <div class="sb-section">
      <div class="sb-h">Session Fields</div>
      <div id="sf-toggles" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px"></div>
      <div id="sf-inputs"></div>
    </div>

    <!-- MCQ ID + Topic -->
    <div class="sb-section">
      <div class="sb-h">
        MCQ ID &amp; Topic
        <button class="meta-add-btn" id="btn-autofill" onclick="autoFill()" title="Auto-fill from PDF text layer">⚡ Auto</button>
      </div>
      <div class="meta-row">
        <div class="meta-badge badge-id">ID</div>
        <input  class="meta-input" id="meta-id"    placeholder="MC3178">
      </div>
      <div class="meta-row">
        <div class="meta-badge badge-tag">TAG</div>
        <textarea class="meta-input" id="meta-tag" rows="2"
                  placeholder="T Cells - Properties…"></textarea>
      </div>
    </div>

    <!-- Options -->
    <div class="sb-section">
      <div class="sb-h">Options <span style="color:var(--muted);font-size:9px;font-weight:400">click letter = correct · others = wrong</span></div>
      <div id="opts-list"></div>
    </div>

    <!-- Keyboard shortcuts -->
    <div class="sb-section kb">
      <b>Click ×4</b> set lines<br>
      <b>U</b> undo last line<br>
      <b>A</b> append next page<br>
      <b>Enter</b> save &amp; next<br>
      <b>S</b> skip (no save) &nbsp; <b>B</b> back
    </div>

    <!-- Buttons -->
    <div class="btns">
      <button class="primary" id="btn-save"   onclick="doSave()"   disabled>✓ SAVE &amp; NEXT</button>
      <button class="amber"   id="btn-append" onclick="doAppend()">+ APPEND NEXT PAGE</button>
      <button                 id="btn-skip"   onclick="doSkip()">→ SKIP PAGE</button>
      <button id="btn-auto" onclick="doAutoProcess()"
              style="background:#7c3aed;color:#fff;font-weight:700;
                     border:none;padding:8px 18px;border-radius:6px;cursor:pointer;">
        ⚡ AUTO-PROCESS ALL
      </button>
      <div id="non-mcq-banner" style="display:none;margin-top:8px;padding:8px 12px;
           background:#f59e0b;color:#1a1a1a;border-radius:6px;font-size:12px;font-weight:700;
           letter-spacing:0.05em;text-align:center;">
        ⚠ NON-MCQ PAGE DETECTED (cover / catalog / schema) — auto-skipping…
      </div>
      <button                 id="btn-back"   onclick="doBack()">← BACK</button>
      <button class="danger"  id="btn-undo"   onclick="doUndo()">✕ UNDO LAST LINE</button>
    </div>

  </div>
</div>

<!-- done overlay (outside body-wrap) -->
<div id="done-overlay">
  <h2>✓ ALL DONE</h2>
  <p id="done-msg"></p>
  <div style="display:flex;gap:12px;margin-top:20px;justify-content:center">
    <button onclick="restartFromFirst()"
      style="padding:10px 22px;background:var(--green);color:#000;border:2px solid #000;
             border-radius:8px;font-weight:900;cursor:pointer;font-size:13px">
      ↩ Resume Unsaved Pages
    </button>
    <button onclick="jumpToPage()"
      style="padding:10px 22px;background:var(--b1);color:#fff;border:2px solid #000;
             border-radius:8px;font-weight:900;cursor:pointer;font-size:13px">
      ⌨ Jump to Page
    </button>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ── State ─────────────────────────────────────────────────────────
let cur          = null;
let lines        = [null,null,null,null];  // b1,r1,r2,b2
let answer       = "";
let appendedB64s = [];
let sessionFields= [];
let loading      = false;

const LINE_COLORS = ["var(--b1)","var(--r1)","var(--r2)","var(--b2)"];
const LINE_NAMES  = ["BLUE top cut","RED end stem","ORANGE end opts","PURPLE end exp"];
const LINE_IDS    = ["lb1","lr1","lr2","lb2"];
const LBLS        = ["lbl-b1","lbl-r1","lbl-r2","lbl-b2"];
const OPTS        = ["A","B","C","D"];

// ── Load current page ─────────────────────────────────────────────
async function loadState() {
  showSpinner(true);
  let data;
  try {
    const r = await fetch("/api/state");
    data    = await r.json();
  } catch(e) {
    showSpinner(false);
    toast("Network error — retrying…");
    setTimeout(loadState, 1500);
    return;
  }
  showSpinner(false);

  if (data.error) {
    document.getElementById("hint-overlay").querySelector(".hint-box").innerHTML =
      '<span style="color:#f87171">⚠ ' + data.error + '</span>';
    return;
  }

  if (data.done) {
    STATE_total = data.total;
    document.getElementById("done-overlay").style.display = "flex";
    document.getElementById("done-msg").textContent =
      data.q_saved + " questions saved from " + data.total + " PDF pages.";
    return;
  }

  cur          = data;
  appendedB64s = [];
  document.getElementById("appended-wrap").innerHTML = "";
  document.getElementById("btn-append").disabled     = false;

  // Reset lines unless restoring saved
  if (data.saved) {
    lines  = [data.saved.b1, data.saved.r1, data.saved.r2, data.saved.b2];
    answer = data.saved.answer || "";
    OPTS.forEach(l => {
      const el = document.getElementById("opt-"+l);
      if (el) el.value = (data.saved.options||{})[l] || "";
    });
    const mid  = document.getElementById("meta-id");
    const mtag = document.getElementById("meta-tag");
    if (mid)  mid.value  = data.saved.mcq_id || "";
    if (mtag) mtag.value = data.saved.topic  || "";
  } else {
    // Use model prediction if available, else blank
    const p = data.prediction;
    if (p) {
      lines = [p.b1, p.r1, p.r2, p.b2];
      toast("🤖 Lines auto-predicted — adjust if needed", 3000);
    } else {
      lines = [null,null,null,null];
    }
    answer = "";
    OPTS.forEach(l => { const el = document.getElementById("oi-"+l); if(el) el.value=""; });
    document.getElementById("meta-id").value  = "";
    document.getElementById("meta-tag").value = "";
  }

  // ── Silent auto-fill: runs on every page load ──────────────────
  // MCQ ID is unique per page — always overwrite (or clear if not found)
  // Options + topic — only fill if currently empty
  fetch("/api/extract/" + cur.page_idx)
    .then(r => r.json())
    .then(d => {
      const midEl  = document.getElementById("meta-id");
      const mtagEl = document.getElementById("meta-tag");

      // MCQ ID: set if found, CLEAR if not found (never leave stale previous page ID)
      if (midEl) midEl.value = d.mcq_id || "";

      if (d.topic  && mtagEl && !mtagEl.value) mtagEl.value = d.topic;
      if (d.options) {
        OPTS.forEach(l => {
          const el = document.getElementById("oi-" + l);
          if (el && !el.value && d.options[l]) el.value = d.options[l];
        });
      }

      // Show image-page hint if no text layer found
      const hasData = d.mcq_id || d.topic || Object.keys(d.options||{}).length > 0;
      if (!hasData) {
        toast("📷 Image page — MCQ ID needs OCR or manual entry", 4000);
      }
    })
    .catch(() => {});

  // Auto-skip non-MCQ pages (cover / catalog / schema)
  const banner = document.getElementById("non-mcq-banner");
  if (data.page_type === "skip") {
    if (banner) banner.style.display = "block";
    setTimeout(() => doSkip(), 800);  // brief flash so user sees it, then skip
    return;
  } else {
    if (banner) banner.style.display = "none";
  }

  // Load image
  const img  = document.getElementById("page-img");
  img.onload = () => { showSpinner(false); renderLines(); renderOpts(); };
  showSpinner(true);
  img.src = "data:image/jpeg;base64," + data.img_b64;

  // Header
  const qSaved = data.q_counter;
  document.getElementById("q-badge").textContent   = "Q" + (qSaved + 1);
  document.getElementById("prog-bar").style.width  = Math.round((data.page_num-1)/data.total*100)+"%";
  document.getElementById("counter").textContent   = "pg " + data.page_num + " / " + data.total;
  document.getElementById("stem-badge").textContent = data.stem + "  ·  pg " + data.page_num;

  // Sidebar info
  document.getElementById("i-pg").textContent = data.page_num + " / " + data.total;
  document.getElementById("i-qn").textContent = qSaved;

  updateNextLabel();
  updateLegend();
  updateButtons();
}

function showSpinner(on) {
  document.getElementById("spinner").classList.toggle("show", on);
}

function updateNextLabel() {
  const idx = lines.findIndex(v => v === null);
  const el  = document.getElementById("i-next");
  if (idx === -1) {
    el.innerHTML = '<span style="color:var(--green)">✓ ready</span>';
  } else {
    el.innerHTML = '<span style="color:' + LINE_COLORS[idx] + '">' + LINE_NAMES[idx] + '</span>';
  }
}

function updateLegend() {
  lines.forEach((v, i) => {
    const el   = document.getElementById("leg"+i);
    const span = el.querySelector("span");
    if (v !== null) {
      span.style.opacity    = "1";
      span.style.fontWeight = "400";
      span.style.color      = "";
      el.querySelector(".leg-dot").style.opacity = "1";
      // add checkmark
      span.textContent = ["① BLUE — top cut","② RED — end of stem",
                          "③ ORANGE — end of options","④ PURPLE — end of explanation"][i] + " ✓";
    } else if (i === lines.findIndex(x => x === null)) {
      span.style.opacity    = "1";
      span.style.fontWeight = "800";
      span.style.color      = LINE_COLORS[i];
      span.textContent = ["① BLUE — top cut","② RED — end of stem",
                          "③ ORANGE — end of options","④ PURPLE — end of explanation"][i];
    } else {
      span.style.opacity    = "0.35";
      span.style.fontWeight = "400";
      span.style.color      = "";
      span.textContent = ["① BLUE — top cut","② RED — end of stem",
                          "③ ORANGE — end of options","④ PURPLE — end of explanation"][i];
    }
  });
}

function updateButtons() {
  document.getElementById("btn-save").disabled = lines.some(v => v === null);
}

// ── Clicks ────────────────────────────────────────────────────────
document.getElementById("page-wrap").addEventListener("click", e => {
  if (!cur || loading) return;
  const frac = getFrac(e);
  const idx  = lines.findIndex(v => v === null);

  if (idx === -1) {
    // All set — move nearest line
    let best=0, bestD=Infinity;
    lines.forEach((v,i) => { const d=Math.abs(v-frac); if(d<bestD){bestD=d;best=i;} });
    lines[best] = frac;
    // Re-sort to preserve order
    const filled = lines.filter(v=>v!==null).sort((a,b)=>a-b);
    for(let i=0;i<4;i++) lines[i] = filled[i] ?? null;
  } else {
    if (idx > 0 && lines[idx-1] !== null && frac <= lines[idx-1]+0.01) {
      toast("Must be below previous line"); return;
    }
    lines[idx] = frac;
  }

  renderLines(); updateLegend(); updateNextLabel(); updateButtons();
});

document.getElementById("page-wrap").addEventListener("mousemove", e => {
  const g = document.getElementById("hover-guide");
  const r = document.getElementById("page-img").getBoundingClientRect();
  const y = e.clientY - r.top;
  if (y<0||y>r.height){g.style.display="none";return;}
  g.style.display="block"; g.style.top=y+"px";
});
document.getElementById("page-wrap").addEventListener("mouseleave",()=>{
  document.getElementById("hover-guide").style.display="none";
});

// ── Render ────────────────────────────────────────────────────────
function renderLines() {
  const img = document.getElementById("page-img");
  const H   = img.offsetHeight;
  if (!H) return;

  const show = (id, lblId, frac, label) => {
    const el = document.getElementById(id);
    if (frac===null){el.style.display="none";return;}
    el.style.display="block";
    el.style.top=(frac*H)+"px";
    document.getElementById(lblId).textContent = label+" "+Math.round(frac*100)+"%";
  };
  show("lb1","lbl-b1",lines[0],"① BLUE");
  show("lr1","lbl-r1",lines[1],"② RED");
  show("lr2","lbl-r2",lines[2],"③ ORANGE");
  show("lb2","lbl-b2",lines[3],"④ PURPLE");

  document.getElementById("hint-overlay").style.display = lines[0]===null?"flex":"none";

  // tints
  const pairs = [
    ["tq",  lines[0], lines[1], "rq"],
    ["topt",lines[1], lines[2], "ropt"],
    ["texp",lines[2], lines[3], "rexp"],
  ];
  pairs.forEach(([tid, top, bot, rid]) => {
    const te = document.getElementById(tid);
    const re = document.getElementById(rid);
    if (top!==null && bot!==null) {
      const y1=top*H, y2=bot*H;
      te.style.cssText="display:block;top:"+y1+"px;height:"+(y2-y1)+"px";
      re.style.cssText="display:block;top:"+(y1+(y2-y1)*.35)+"px";
    } else {
      te.style.display="none"; re.style.display="none";
    }
  });
}

// ── Options ───────────────────────────────────────────────────────
function renderOpts() {
  const c = document.getElementById("opts-list");
  c.innerHTML = "";
  OPTS.forEach(l => {
    const row = document.createElement("div");
    row.className = "opt-row";

    const btn = document.createElement("div");
    btn.id = "ob-"+l; btn.textContent = l;
    btn.title = "Click = mark as correct answer";
    btn.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      answer = (answer===l) ? "" : l;
      refreshOptStyles();
      toast(answer ? "✓ Answer: "+answer : "Answer cleared");
    });

    const ta = document.createElement("textarea");
    ta.id = "oi-"+l; ta.rows=2; ta.placeholder="Option "+l+"…"; ta.spellcheck=false;
    const saved = cur?.saved?.options?.[l] || "";
    ta.value = saved;

    row.appendChild(btn); row.appendChild(ta);
    c.appendChild(row);
  });
  refreshOptStyles();
}

function refreshOptStyles() {
  OPTS.forEach(x => {
    const btn = document.getElementById("ob-"+x);
    const inp = document.getElementById("oi-"+x);
    if (!btn) return;
    if (!answer) {
      btn.className = "opt-letter";
      inp.className = "opt-input";
    } else if (x === answer) {
      btn.className = "opt-letter correct";
      inp.className = "opt-input correct-input";
    } else {
      btn.className = "opt-letter wrong";
      inp.className = "opt-input wrong-input";
    }
  });
}

// ── Session fields ────────────────────────────────────────────────
// Preset definitions
const SF_PRESETS = [
  {key:'Subject', val:''},
  {key:'PYQ',     val:''},
  {key:'Year',    val:''},
  {key:'Exam',    val:'NEET PG'},
  {key:'Source',  val:'Marrow'},
];
// Which presets are active (toggled on)
let sfActive = {};   // key → true/false

function initSF() {
  const tgl = document.getElementById('sf-toggles');
  tgl.innerHTML = '';
  SF_PRESETS.forEach(p => {
    const btn = document.createElement('button');
    btn.id          = 'sft-' + p.key;
    btn.textContent = p.key;
    btn.style.cssText = 'padding:3px 8px;font-size:10px;border-radius:12px;cursor:pointer;' +
      'border:1px solid var(--border);background:transparent;color:var(--muted);' +
      'font-family:inherit;letter-spacing:0.5px;transition:all .15s;width:auto;font-weight:600';
    btn.addEventListener('click', () => toggleSF(p.key, p.val));
    tgl.appendChild(btn);
  });
  // Custom add row
  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:4px;width:100%;margin-top:2px';
  row.innerHTML = '<input id="sf-ckey" placeholder="Custom field…" maxlength="20" ' +
    'style="flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);' +
    'padding:3px 6px;border-radius:4px;font-family:inherit;font-size:10px;min-width:0">' +
    '<button id="sf-cadd" style="flex:0 0 auto;padding:3px 8px;font-size:10px;width:auto;' +
    'border:1px solid var(--green);background:transparent;color:var(--green);' +
    'border-radius:4px;cursor:pointer;font-family:inherit;font-weight:700">+</button>';
  tgl.appendChild(row);
  document.getElementById('sf-cadd').addEventListener('click', () => {
    const k = (document.getElementById('sf-ckey').value || '').trim();
    if (!k) return;
    if (!SF_PRESETS.find(p => p.key === k)) SF_PRESETS.push({key:k, val:''});
    document.getElementById('sf-ckey').value = '';
    if (!sfActive[k]) toggleSF(k, '');
    initSF();  // re-render toggles with new preset
  });
}

function toggleSF(key, defaultVal) {
  sfActive[key] = !sfActive[key];
  // Update toggle button style
  const btn = document.getElementById('sft-' + key);
  if (btn) {
    btn.style.background = sfActive[key] ? 'var(--green)' : 'transparent';
    btn.style.color      = sfActive[key] ? '#000' : 'var(--muted)';
    btn.style.borderColor= sfActive[key] ? 'var(--green)' : 'var(--border)';
  }
  renderSFInputs(key, defaultVal);
}

function renderSFInputs(changedKey, defaultVal) {
  const c = document.getElementById('sf-inputs');
  // Add or remove input for changedKey
  if (sfActive[changedKey]) {
    if (document.getElementById('sfi-' + changedKey)) return; // already exists
    const row = document.createElement('div');
    row.id = 'sfrow-' + changedKey;
    row.style.cssText = 'display:flex;align-items:center;gap:5px;margin-bottom:5px';
    const lbl = document.createElement('span');
    lbl.style.cssText = 'font-size:10px;color:var(--muted);min-width:46px;text-transform:uppercase;letter-spacing:0.8px';
    lbl.textContent = changedKey;
    const inp = document.createElement('input');
    inp.id          = 'sfi-' + changedKey;
    inp.value       = defaultVal || '';
    inp.placeholder = changedKey;
    inp.style.cssText = 'flex:1;background:#0a0e1a;border:1px solid var(--border);color:var(--text);' +
      'padding:4px 6px;border-radius:4px;font-family:inherit;font-size:11px;min-width:0;outline:none';
    inp.addEventListener('focus', function(){ this.style.borderColor='var(--b1)'; });
    inp.addEventListener('blur',  function(){ this.style.borderColor='var(--border)'; });
    row.appendChild(lbl);
    row.appendChild(inp);
    c.appendChild(row);
  } else {
    const row = document.getElementById('sfrow-' + changedKey);
    if (row) row.remove();
  }
}

function getSessionMeta() {
  const o = {};
  Object.keys(sfActive).forEach(key => {
    if (!sfActive[key]) return;
    const inp = document.getElementById('sfi-' + key);
    if (inp && inp.value.trim()) o[key] = inp.value.trim();
  });
  return o;
}

// ── Auto-fill from PDF text layer ─────────────────────────────────
async function autoFill() {
  if (!cur) return;
  const btn = document.getElementById("btn-autofill");
  btn.textContent = "…";
  btn.disabled    = true;
  try {
    const r = await fetch("/api/extract/" + cur.page_idx);
    const d = await r.json();

    // Options
    let filled = 0;
    OPTS.forEach(l => {
      if (d.options && d.options[l]) {
        const el = document.getElementById("oi-"+l);
        if (el && !el.value) { el.value = d.options[l]; filled++; }
      }
    });

    // MCQ ID
    if (d.mcq_id) {
      const el = document.getElementById("meta-id");
      if (el && !el.value) el.value = d.mcq_id;
    }

    // Topic
    if (d.topic) {
      const el = document.getElementById("meta-tag");
      if (el && !el.value) el.value = d.topic;
    }

    const n = filled + (d.mcq_id?1:0) + (d.topic?1:0);
    toast(n > 0 ? "⚡ Auto-filled " + n + " fields" : "No text layer on this page");
  } catch(e) {
    toast("Extract failed");
  }
  btn.textContent = "⚡ Auto";
  btn.disabled    = false;
}

// ── Append ────────────────────────────────────────────────────────
function doAppend() {
  if (!cur?.next_b64){toast("No next page");return;}
  appendedB64s.push(cur.next_b64);
  const wrap=document.getElementById("appended-wrap");
  const img=document.createElement("img");
  img.src="data:image/jpeg;base64,"+cur.next_b64;
  img.className="appended-img";
  wrap.appendChild(img);
  document.getElementById("btn-append").disabled=true;
  toast("+ Page appended");
}

// ── Save ──────────────────────────────────────────────────────────
async function doSave() {
  if (lines.some(v=>v===null)) return;
  const options={};
  OPTS.forEach(l=>{const v=(document.getElementById("oi-"+l)?.value||"").trim();if(v)options[l]=v;});
  const mcq_id=(document.getElementById("meta-id")?.value||"").trim();
  const topic =(document.getElementById("meta-tag")?.value||"").trim();
  document.getElementById("btn-save").disabled=true;
  const r = await fetch("/api/save",{
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({
      b1:lines[0],r1:lines[1],r2:lines[2],b2:lines[3],
      options, answer, mcq_id, topic,
      session_meta: getSessionMeta(),
      appended_b64s: appendedB64s,
    }),
  });
  const d = await r.json();
  toast("✓ Q"+d.q_num+" saved");
  loadState();
}

async function doSkip(){ await fetch("/api/skip",{method:"POST"}); toast("→ Skipped"); loadState(); }

async function restartFromFirst(){
  // Find first page that has no saved split
  const r = await fetch("/api/unsaved_pages");
  const d = await r.json();
  if(!d.first_unsaved && d.first_unsaved !== 0){
    alert("All pages have been split!"); return;
  }
  await fetch("/api/jump",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({page: d.first_unsaved + 1})});
  document.getElementById("done-overlay").style.display = "none";
  loadState();
  toast("Resuming from page " + (d.first_unsaved+1) + " — " + d.count + " unsaved pages remaining");
}

async function jumpToPage(){
  const p = prompt("Jump to page number (1–" + (STATE_total||282) + "):");
  if(!p) return;
  await fetch("/api/jump",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({page: parseInt(p)})});
  document.getElementById("done-overlay").style.display = "none";
  loadState();
}

let STATE_total = 0;
async function doBack(){ await fetch("/api/back",{method:"POST"}); loadState(); }

// ── Auto-process all remaining pages ─────────────────────────────
async function doAutoProcess() {
  const confirmed = confirm(
    "⚡ AUTO-PROCESS ALL REMAINING PAGES\n\n" +
    "This will automatically:\n" +
    "• Skip cover / catalog / schema pages\n" +
    "• Detect cut lines using color + text anchors + model\n" +
    "• Extract MCQ ID and options from text layer\n" +
    "• Save all as Q1, Q2, Q3…\n\n" +
    "Already-saved pages are skipped.\n" +
    "You can review and fix any page afterward.\n\n" +
    "Continue?"
  );
  if (!confirmed) return;

  const ov  = document.getElementById("auto-overlay");
  const bar = document.getElementById("auto-bar");
  const st  = document.getElementById("auto-status");
  const log = document.getElementById("auto-log");
  ov.style.display = "flex";
  bar.style.width  = "0%";
  st.textContent   = "Starting…";
  log.textContent  = "";

  // Start background job (returns immediately)
  let startResp;
  try {
    startResp = await fetch("/api/auto_process", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({dry_run: false})
    });
  } catch(e) {
    st.textContent = "Failed to start: " + e;
    return;
  }
  const startData = await startResp.json();
  if (!startData.ok) {
    st.textContent = startData.message || "Failed to start";
    setTimeout(() => { ov.style.display="none"; }, 3000);
    return;
  }
  if (startData.done) {
    // Nothing to process
    st.textContent = startData.message;
    bar.style.width = "100%";
    setTimeout(() => { ov.style.display="none"; loadState(); }, 2000);
    return;
  }

  st.textContent = `Queued ${startData.queued} pages — processing…`;

  // Poll function
  async function pollStatus() {
    try {
      const r = await fetch("/api/auto_status");
      const d = await r.json();

      const pct = d.pct || 0;
      bar.style.width = pct + "%";
      st.textContent  = `${d.processed} / ${d.total_todo} saved  (${pct}%)  —  page ${d.current_page} / ${d.total_pdf}`;

      if (d.errors && d.errors.length) {
        log.textContent = "Recent errors:\n" +
          d.errors.map(e => `  p${e.page}: ${e.error}`).join("\n");
      }

      if (d.done && !d.running) {
        bar.style.width = "100%";
        st.textContent  = d.message || "Done!";
        log.textContent = (d.errors && d.errors.length)
          ? "Errors:\n" + d.errors.map(e => `  p${e.page}: ${e.error}`).join("\n")
          : `✓ ${d.processed} questions saved, ${d.skipped} skipped`;
        setTimeout(() => { ov.style.display="none"; loadState(); }, 3000);
        return;  // stop polling
      }
    } catch(e) { /* server busy, retry */ }
    setTimeout(pollStatus, 1000);  // schedule next poll
  }

  // Start polling immediately
  setTimeout(pollStatus, 300);
}

function doUndo(){
  for(let i=3;i>=0;i--){ if(lines[i]!==null){lines[i]=null;break;} }
  renderLines(); updateLegend(); updateNextLabel(); updateButtons();
}

async function jumpToPage(){
  const v=parseInt(document.getElementById("j-page").value);
  if(!v)return;
  await fetch("/api/jump",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({page:v})});
  loadState();
}

// keyboard
document.addEventListener("keydown", e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA")return;
  if(e.key==="Enter") doSave();
  if(e.key==="s"||e.key==="S") doSkip();
  if(e.key==="b"||e.key==="B") doBack();
  if(e.key==="u"||e.key==="U") doUndo();
  if(e.key==="a"||e.key==="A") doAppend();
});

function getFrac(e){
  const r=document.getElementById("page-img").getBoundingClientRect();
  return Math.max(0.01,Math.min(0.99,(e.clientY-r.top)/r.height));
}
function toast(msg){
  const t=document.getElementById("toast");
  t.textContent=msg; t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"),2200);
}

// start
initSF();
loadState();
</script>

<!-- Auto-process progress overlay -->
<div id="auto-overlay" style="display:none;position:fixed;inset:0;
     background:rgba(0,0,0,0.82);z-index:9999;
     display:none;flex-direction:column;align-items:center;justify-content:center;gap:16px;">
  <div style="color:#7c3aed;font-size:48px;">⚡</div>
  <div style="color:#fff;font-size:22px;font-weight:700;">Auto-Processing All Pages…</div>
  <div style="width:380px;height:16px;background:#333;border-radius:8px;overflow:hidden;">
    <div id="auto-bar" style="height:100%;background:#7c3aed;width:0%;transition:width 0.4s;"></div>
  </div>
  <div id="auto-status" style="color:#ccc;font-size:14px;">Initialising…</div>
  <div id="auto-log"
       style="max-height:200px;overflow-y:auto;width:520px;
              background:#111;border-radius:8px;padding:12px;
              font-family:monospace;font-size:11px;color:#6ee7b7;"></div>
  <button id="btn-auto-cancel"
          style="margin-top:4px;padding:6px 20px;background:#374151;
                 color:#fff;border:none;border-radius:6px;cursor:pointer;
                 font-size:13px;">
    ✕ Cancel
  </button>
</div>

</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="P2E Splitter v4")
    ap.add_argument("--pdf",  default=None)
    ap.add_argument("--out",  default=None)
    ap.add_argument("--dpi",  type=int, default=150)
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()

    # PDF picker
    if not args.pdf:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.wm_attributes("-topmost", True)
            args.pdf = filedialog.askopenfilename(
                title      = "Select Marrow PDF",
                filetypes  = [("PDF files","*.pdf"),("All files","*.*")],
                initialdir = r"D:\Study meteial\NEET PG QBANK\pyq",
            )
            root.destroy()
        except Exception:
            pass

    if not args.pdf:
        raw = input("Paste PDF path: ").strip()
        args.pdf = raw.strip('"').strip("'")

    if not args.pdf or not os.path.exists(args.pdf):
        print(f"ERROR: file not found — {args.pdf!r}")
        sys.exit(1)

    # Count pages (fast — no rendering)
    import pypdfium2 as pdfium
    pdf   = pdfium.PdfDocument(args.pdf)
    total = len(pdf)
    pdf.close()
    print(f"\nPDF: {args.pdf}")
    print(f"Pages: {total}  (rendering on demand at {args.dpi} DPI)")

    stem    = Path(args.pdf).stem
    out_dir = args.out or str(Path(args.pdf).parent.parent / "p2e" / "output")
    img_dir = os.path.join(out_dir, stem, "images")
    os.makedirs(img_dir, exist_ok=True)

    splits_path = os.path.join(out_dir, stem, "splits.json")
    existing    = {}
    if os.path.exists(splits_path):
        with open(splits_path, encoding="utf-8") as f:
            existing = json.load(f)
        # Find highest q_num saved
        q_max = max((v.get("q_num",0) for v in existing.values() if isinstance(v, dict)), default=0)
        # Resume from first unsaved page
        saved_pages = {int(k) for k, v in existing.items() if isinstance(v, dict)}
        start_page  = next((i for i in range(total) if i not in saved_pages), total)
        print(f"Resuming — {len(existing)} pages already done, q_counter={q_max}, starting at page {start_page+1}")
    else:
        q_max      = 0
        start_page = 0

    STATE.update({
        "pdf_path"   : args.pdf,
        "total"      : total,
        "current"    : start_page,
        "q_counter"  : q_max,
        "img_dir"    : img_dir,
        "splits_path": splits_path,
        "splits"     : existing,
        "dpi"        : args.dpi,
    })

    print(f"\nOpen → http://127.0.0.1:{args.port}")
    print("Each page renders on first load — no wait upfront\n")

    _load_cut_model()   # loads train_cutter.pkl if present

    def _open():
        time.sleep(1.0)
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    threading.Thread(target=_open, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
