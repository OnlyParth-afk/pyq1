"""
pdfplumber_noise_removal.py
Provides: run_pipeline, is_schema_page, is_green, PageResult,
          build_rich_lines, detect_answer, remove_noise
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import pdfplumber

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class RichLine:
    top       : float
    plain     : str
    html      : str
    font_sizes: List[float] = field(default_factory=list)
    is_bold   : bool        = False
    is_noise  : bool        = False


@dataclass
class PageResult:
    lines  : List[Dict[str, Any]]
    answer : Optional[str]
    plain  : str
    html   : str


# ═══════════════════════════════════════���══════════════════════════
# NOISE PATTERNS
# ══════════════════════════════════════════════════════════════════

RE_NOISE = re.compile(
    r"""
    ©\s*[Mm]arrow           |
    marrow\.in              |
    \bSOLVE\b               |
    \bNEXT\b                |
    \bBOOKMARK\b            |
    \bCOMPLETE\b            |
    \bPREMIUM\b             |
    \b[a-f0-9]{20,}\b       |
    \[\s*\d{1,3}%\s*\]      |
    Page\s+\d+\s+of\s+\d+  |
    ^\s*\d+\s*$
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

# Standard answer line: Ans: B / Answer: C / Key: A
RE_ANS_STD = re.compile(
    r"""
    (?:ans(?:wer)?|correct\s*ans(?:wer)?|key)
    \s*[:\-]?\s*
    \(?([A-Da-d])\)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Green tick pattern in Marrow PDFs:
# ✓ A. / ✔ A. / ✅ A. / >> A. / * A.
RE_ANS_TICK = re.compile(
    r"""
    (?:✓|✔|✅|☑|🗸|>>|=>|\*\*)
    \s*
    ([A-Da-d])
    [\.\):\s]
    """,
    re.VERBOSE,
)

# Percentage confidence: A. 17D [65%] — highest % = correct
RE_ANS_PCT = re.compile(
    r"""
    ([A-Da-d])[\.\)]\s*.+?\[(\d{1,3})%\]
    """,
    re.VERBOSE,
)

# Schema / chapter header
RE_SCHEMA = re.compile(
    r"""
    (?:chapter|unit|section|topic|schema|subject\s*:)
    \s*[:\-]?\s*.{2,}
    """,
    re.VERBOSE | re.IGNORECASE,
)

BOLD_SIZE_THRESH = 13.0


# ══════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════

def is_green(rgb: Tuple[float, float, float]) -> bool:
    """
    Check if an RGB colour (0-255) is in the green range.
    Used to detect Marrow green highlight / correct-answer boxes.
    """
    r, g, b = rgb
    if g < 100:
        return False
    if r > g * 0.8:
        return False
    if b > g * 0.8:
        return False
    return True


def _colour_of_word(word: dict) -> Optional[Tuple[float,float,float]]:
    """
    Try to read non-stroking (fill) colour from a pdfplumber word dict.
    Returns (R,G,B) 0-255 or None.
    """
    # pdfplumber exposes colour via chars
    chars = word.get("chars", [])
    if not chars:
        return None
    c = chars[0]
    ncs = c.get("non_stroking_color")
    if ncs is None:
        return None
    try:
        if len(ncs) == 3:
            return tuple(v * 255 for v in ncs)
        if len(ncs) == 1:           # greyscale
            v = ncs[0] * 255
            return (v, v, v)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════
# NOISE REMOVAL — text level
# ══════════════════════════════════════════════════════════════════

def remove_noise(text: str) -> str:
    """Remove watermarks and noise tokens from plain text."""
    if not text:
        return ""
    lines   = text.splitlines()
    cleaned = []
    for line in lines:
        if RE_NOISE.search(line):
            continue
        line = re.sub(r"©\s*[Mm]arrow[^\n]*", "", line)
        line = re.sub(r"\b[a-f0-9]{20,}\b",   "", line, flags=re.IGNORECASE)
        line = re.sub(r"\[\s*\d{1,3}%\s*\]",  "", line)
        if line.strip():
            cleaned.append(line)
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]+", " ",    result)
    return result.strip()


# ══════════════════════════════════════════════════════════════════
# ANSWER DETECTOR
# ══════════════════════════════════════════════════════════════════

def detect_answer(text: str) -> Optional[str]:
    """
    Detect correct answer letter from page text.

    Priority:
    1. Standard   — Ans: B  /  Answer: C
    2. Tick mark  — ✓ A.  /  ✔ B.
    3. Percentage — highest % among A/B/C/D options (≥40%)
    """
    if not text:
        return None

    # 1. Standard answer line
    m = RE_ANS_STD.search(text)
    if m:
        return m.group(1).upper()

    # 2. Tick / checkmark
    m = RE_ANS_TICK.search(text)
    if m:
        return m.group(1).upper()

    # 3. Percentage fallback
    pct_matches = RE_ANS_PCT.findall(text)
    if pct_matches:
        best = max(pct_matches, key=lambda x: int(x[1]))
        if int(best[1]) >= 40:
            return best[0].upper()

    return None


def detect_answer_by_colour(page) -> Optional[str]:
    """
    Detect the correct answer by looking for green-coloured text
    on the page (Marrow marks correct option in green).

    Returns A/B/C/D or None.
    """
    try:
        words = page.extract_words(
            extra_attrs=["non_stroking_color", "chars"],
            keep_blank_chars=False,
        )
    except Exception:
        return None

    opt_re = re.compile(r'^([A-Da-d])[\.\):]?$')

    for word in words:
        colour = _colour_of_word(word)
        if colour and is_green(colour):
            # Check if this word is an option letter
            txt = word.get("text", "").strip()
            m   = opt_re.match(txt)
            if m:
                return m.group(1).upper()
            # Or if the next character after letter is a dot/bracket
            m2 = re.match(r'^([A-Da-d])[\.\):\s]', txt)
            if m2:
                return m2.group(1).upper()

    return None


# ══════════════════════════════════════════════════════════════════
# SCHEMA PAGE DETECTOR
# ══════════════════════════════════════════════════════════════════

def is_schema_page(page) -> bool:
    """
    Return True if page looks like a chapter / schema header
    rather than an MCQ page.
    """
    try:
        text = page.extract_text() or ""
    except Exception:
        return False

    text = text.strip()
    if not text or len(text) < 20:
        return False

    # Short page with schema keywords = schema page
    if len(text) < 350 and RE_SCHEMA.search(text):
        return True

    # Very short page = likely cover / header
    if len(text) < 80:
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# RICH LINE BUILDER
# ══════════════════════════════════════════════════════════════════

def build_rich_lines(page) -> List[Dict[str, Any]]:
    """
    Extract lines from a pdfplumber page with font metadata.

    Returns list of dicts:
      { top, plain, html, font_sizes, is_bold, is_noise }
    """
    lines: List[Dict[str, Any]] = []

    try:
        words = page.extract_words(
            extra_attrs=["fontname", "size"],
            keep_blank_chars=False,
        )
    except Exception:
        words = []

    if not words:
        # Fallback: plain text only
        try:
            text = page.extract_text() or ""
            text = remove_noise(text)
            for i, line in enumerate(text.splitlines()):
                line = line.strip()
                if line:
                    lines.append({
                        "top"       : float(i * 14),
                        "plain"     : line,
                        "html"      : _text_to_html(line),
                        "font_sizes": [12.0],
                        "is_bold"   : False,
                        "is_noise"  : bool(RE_NOISE.search(line)),
                    })
        except Exception:
            pass
        return lines

    # Group words into lines by top coordinate (±3pt tolerance)
    line_map: Dict[int, List[dict]] = {}
    for w in words:
        top_key = int(round(float(w.get("top", 0)) / 3)) * 3
        if top_key not in line_map:
            line_map[top_key] = []
        line_map[top_key].append(w)

    for top_key in sorted(line_map.keys()):
        words_in_line = sorted(
            line_map[top_key],
            key=lambda w: float(w.get("x0", 0)),
        )
        plain_parts : List[str]   = []
        html_parts  : List[str]   = []
        font_sizes  : List[float] = []
        is_bold                   = False

        for w in words_in_line:
            text  = w.get("text", "").strip()
            if not text:
                continue

            size  = float(w.get("size", 12) or 12)
            fname = str(w.get("fontname", "") or "")
            bold  = (
                "bold"  in fname.lower()
                or "black" in fname.lower()
                or size >= BOLD_SIZE_THRESH
            )

            plain_parts.append(text)
            font_sizes.append(size)

            if bold:
                is_bold = True
                html_parts.append(f"<b>{_escape(text)}</b>")
            else:
                html_parts.append(_escape(text))

        plain = " ".join(plain_parts).strip()
        html  = " ".join(html_parts).strip()

        if not plain:
            continue

        # Skip pure-noise lines
        if RE_NOISE.search(plain):
            continue

        lines.append({
            "top"       : float(top_key),
            "plain"     : plain,
            "html"      : html,
            "font_sizes": font_sizes,
            "is_bold"   : is_bold,
            "is_noise"  : False,
        })

    return lines


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    page        : Any,
    image_bboxes: List[Tuple] = None,
) -> Optional[PageResult]:
    """
    Full noise-removal pipeline for one pdfplumber page.

    Parameters
    ----------
    page         : pdfplumber page object
    image_bboxes : list of (x0, top, x1, bottom, card_type) tuples
                   Lines overlapping these are suppressed.

    Returns
    -------
    PageResult with .lines .answer .plain .html
    Returns None if page should be skipped (schema/cover).
    """
    if image_bboxes is None:
        image_bboxes = []

    # Skip schema / cover pages
    if is_schema_page(page):
        return None

    # Build rich lines
    rich_lines = build_rich_lines(page)

    # Filter lines that fall inside image bounding boxes
    def _in_image(top: float) -> bool:
        for bbox in image_bboxes:
            itop = bbox[1]
            ibot = bbox[3]
            if itop - 5 <= top <= ibot + 5:
                return True
        return False

    filtered = [ln for ln in rich_lines if not _in_image(ln["top"])]

    # Detect answer — try text first, then colour
    all_plain = "\n".join(ln["plain"] for ln in rich_lines)
    answer    = detect_answer(all_plain)
    if not answer:
        answer = detect_answer_by_colour(page)

    plain = "\n".join(ln["plain"] for ln in filtered)
    html  = "".join(
        f'<p>{ln["html"]}</p>'
        for ln in filtered
        if ln["html"].strip()
    )

    return PageResult(
        lines  = filtered,
        answer = answer,
        plain  = plain,
        html   = html,
    )


# ══════════════════════════════════════════════════════════════════
# HTML HELPERS
# ══════════════════════════════════════════════════════════════════

def _escape(text: str) -> str:
    import html
    return html.escape(text)


def _text_to_html(text: str) -> str:
    t = _escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\b_(.+?)_\b",   r"<i>\1</i>", t)
    return t