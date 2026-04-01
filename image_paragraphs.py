"""
image_paragraphs.py — v5
NEW FIXES vs v4:
  FIX-1  _is_blank()        — skip solid-colour background rectangles
  FIX-2  _is_logo()         — skip org logos (CDC, WHO, Harrison etc.)
  FIX-3  _is_ui_chrome()    — skip scrollbars, teal strips, UI artifacts
  FIX-4  _trim_borders()    — auto-crop coloured borders from clinical photos
  FIX-5  _is_table() improved — catches more table types
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pdfplumber

from pdfplumber_noise_removal import (
    run_pipeline,
    is_schema_page,
    is_green,
    PageResult,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

RENDER_DPI        = 300
IMG_MIN_W_PT      = 60
IMG_MIN_H_PT      = 40
IMG_MAX_W_FRAC    = 0.95
IMG_MAX_H_FRAC    = 0.95
LINE_GAP_JOIN_PT  = 18
LINE_GAP_BREAK_PT = 40

UPSCALE_MIN_PX    = 400
UPSCALE_MAX_SCALE = 3.0

# OpenCV card detection
WHITE_THRESH    = 251
MORPH_CLOSE_PX  = 8
MIN_AREA_FRAC   = 0.025
MIN_CARD_DIM_PX = 60
MAX_WIDTH_FRAC  = 0.99
MIN_HEIGHT_PX   = 100
MAX_ASPECT      = 6.0
SAFE_TOP_FRAC   = 0.10
SAFE_BOT_FRAC   = 0.92
OVERLAP_THRESH  = 0.70
PEARL_GAP_PX    = 10

MIN_REAL_W_PX   = 120
MIN_REAL_H_PX   = 80

TEAL_HSV_LO = np.array([160,  25, 140])
TEAL_HSV_HI = np.array([185, 160, 255])

PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

# Table colour ranges
BEIGE_HSV_LO  = np.array([ 18,  8, 210])
BEIGE_HSV_HI  = np.array([ 45, 70, 255])
TABLE_BLUE_LO = np.array([ 85, 20, 160])
TABLE_BLUE_HI = np.array([130, 80, 240])

# Pie widget
PIE_GREEN_LO  = np.array([ 60, 80,  80])
PIE_GREEN_HI  = np.array([ 90,255, 255])

# Border colours to trim (yellow, beige, pink strips)
BORDER_COLOURS = [
    # (HSV_LO, HSV_HI, name)
    (np.array([ 18,  30, 180]), np.array([ 45, 180, 255]), "yellow"),
    (np.array([  0,  30, 180]), np.array([ 15, 120, 255]), "pink"),
    (np.array([ 85,  20, 180]), np.array([130,  80, 255]), "teal"),
]
BORDER_MIN_FRAC = 0.70   # must cover ≥70% of edge to be trimmed
BORDER_MAX_FRAC = 0.18   # border must be ≤18% of image width


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class ImageSlot:
    top      : float
    bottom   : float
    x0       : float
    x1       : float
    file     : str
    tag      : str
    card_type: str = "diagram"
    source   : str = "opencv"
    width_px : int = 0
    height_px: int = 0


@dataclass
class ContentItem:
    type      : str
    top       : float
    text      : str         = ""
    html      : str         = ""
    plain     : str         = ""
    tag       : str         = ""
    file      : str         = ""
    card_type : str         = ""
    font_sizes: List[float] = field(default_factory=list)


@dataclass
class Paragraph:
    type     : str
    top      : float
    html     : str  = ""
    plain    : str  = ""
    tag      : str  = ""
    file     : str  = ""
    card_type: str  = ""


@dataclass
class PageContent:
    paragraphs    : List[Paragraph]
    image_slots   : List[ImageSlot]
    cv_bboxes     : List[Tuple]
    html          : str
    plain         : str
    answer        : Optional[str]  = None
    page_rgb      : Optional[object] = None   # np.ndarray, rendered at RENDER_DPI
    page_height_pt: float            = 0.0    # original PDF page height in points


# ══════════════════════════════════════════════════════════════════
# PAGE RENDER
# ══════════════════════════════════════════════════════════════════

def _render_page_rgb(
    page: pdfplumber.page.Page,
    dpi : int = RENDER_DPI,
) -> np.ndarray:
    pi      = page.to_image(resolution=dpi)
    pil_img = pi.original
    return np.array(pil_img.convert("RGB"))


# ══════════════════════════════════════════════════════════════════
# PNG SAVE
# ══════════════════════════════════════════════════════════════════

def _save_png(crop_rgb: np.ndarray, path: str) -> Tuple[int, int]:
    h, w = crop_rgb.shape[:2]
    if w < UPSCALE_MIN_PX and h > 10:
        scale    = min(UPSCALE_MIN_PX / w, UPSCALE_MAX_SCALE)
        new_w    = int(w * scale)
        new_h    = int(h * scale)
        crop_rgb = cv2.resize(
            crop_rgb, (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = crop_rgb.shape[:2]
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, PNG_PARAMS)
    return w, h


# ══════════════════════════════════════════════════════════════════
# FIX-1  BLANK DETECTOR
# ══════════════════════════════════════════════════════════════════

def _is_blank(crop: np.ndarray) -> bool:
    """
    Return True if crop is a solid-colour background rectangle.
    Catches: cream boxes, white padding, beige backgrounds.
    """
    if crop.size == 0:
        return True
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Std-dev of pixel values — blank = very low variance
    std = float(np.std(gray))
    if std < 8.0:
        log.debug(f"[blank] std={std:.1f} → skip")
        return True

    # Edge density — blank has almost no edges
    edges     = cv2.Canny(gray, 30, 100)
    edge_frac = np.sum(edges > 0) / (h * w)
    if edge_frac < 0.005:
        log.debug(f"[blank] edge_frac={edge_frac:.4f} → skip")
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# FIX-2  LOGO DETECTOR
# ══════════════════════════════════════════════════════════════════

def _is_logo(crop: np.ndarray) -> bool:
    """
    Detect organisation logos (CDC, WHO, Harrison, etc.)
    Logos typically:
      - Square-ish aspect ratio
      - Dominated by one or two bold colours
      - High contrast but small real content area
      - Blue dominant (CDC blue, WHO blue)
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return False

    aspect = w / max(h, 1)

    # Logos are roughly square to 2:1
    if not (0.5 < aspect < 2.5):
        return False

    # Must be small-ish
    if w > 400 or h > 400:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # CDC blue: deep blue with lines pattern
    deep_blue = cv2.inRange(hsv,
                            np.array([100, 80, 80]),
                            np.array([130, 255, 200]))
    blue_frac = np.sum(deep_blue > 0) / (h * w)
    if blue_frac > 0.40:
        log.debug(f"[logo] deep_blue={blue_frac:.2f} → skip")
        return True

    # Check for very few distinct colours = logo/icon
    # Downscale for speed
    small = cv2.resize(crop, (32, 32))
    pixels = small.reshape(-1, 3).astype(np.float32)
    # Count unique colour clusters roughly
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, _ = cv2.kmeans(
            pixels, 4, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )
        unique_labels = len(np.unique(labels))
        # Real photos have varied colours; logos have 2-3
        if unique_labels <= 2:
            log.debug(f"[logo] low colour diversity → skip")
            return True
    except Exception:
        pass

    return False


# ══════════════════════════════════════════════════════════════════
# FIX-3  UI CHROME DETECTOR
# ══════════════════════════════════════════════════════════════════

def _is_ui_chrome(crop: np.ndarray) -> bool:
    """
    Detect Marrow UI chrome elements:
    - Scrollbar strips (very narrow, teal/blue)
    - Bookmark icons (small, blue/teal square)
    - Navigation arrows
    - Progress bar fragments
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return True

    aspect = w / max(h, 1)

    # Very thin strips = scrollbar / border artifact
    if w < 40 or h < 40:
        log.debug(f"[chrome] tiny {w}×{h} → skip")
        return True

    # Extreme aspect ratio = scrollbar or banner
    if aspect > 8.0 or aspect < 0.12:
        log.debug(f"[chrome] extreme aspect {aspect:.2f} → skip")
        return True

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Teal/cyan strip (Marrow scrollbar / active border)
    teal = cv2.inRange(hsv,
                       np.array([85, 100, 150]),
                       np.array([105, 255, 255]))
    teal_frac = np.sum(teal > 0) / (h * w)
    if teal_frac > 0.35:
        log.debug(f"[chrome] teal_frac={teal_frac:.2f} → skip")
        return True

    # Grey bookmark icon: small, mostly grey
    gray_img  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    grey_mask = cv2.inRange(gray_img, np.array([160]), np.array([220]))
    grey_frac = np.sum(grey_mask > 0) / (h * w)
    if grey_frac > 0.70 and w < 80 and h < 120:
        log.debug(f"[chrome] grey bookmark → skip")
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# FIX-4  BORDER TRIMMER
# ══════════════════════════════════════════════════════════════════

def _trim_borders(crop: np.ndarray) -> np.ndarray:
    """
    Auto-crop solid coloured borders (yellow, pink, teal strips)
    from the edges of clinical photos.

    Handles: yellow left strip on Madura foot image, etc.
    """
    if crop.size == 0:
        return crop

    h, w  = crop.shape[:2]
    hsv   = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    left  = 0
    right = w
    top   = 0
    bot   = h

    max_border_w = int(w * BORDER_MAX_FRAC)
    max_border_h = int(h * BORDER_MAX_FRAC)

    for lo, hi, name in BORDER_COLOURS:
        mask = cv2.inRange(hsv, lo, hi)

        # Check LEFT strip
        for x in range(1, max_border_w):
            col_frac = np.sum(mask[:, :x] > 0) / (h * x)
            if col_frac >= BORDER_MIN_FRAC:
                left = max(left, x)
            else:
                break

        # Check RIGHT strip
        for x in range(1, max_border_w):
            col_frac = np.sum(mask[:, w-x:] > 0) / (h * x)
            if col_frac >= BORDER_MIN_FRAC:
                right = min(right, w - x)
            else:
                break

        # Check TOP strip
        for y in range(1, max_border_h):
            row_frac = np.sum(mask[:y, :] > 0) / (y * w)
            if row_frac >= BORDER_MIN_FRAC:
                top = max(top, y)
            else:
                break

        # Check BOTTOM strip
        for y in range(1, max_border_h):
            row_frac = np.sum(mask[h-y:, :] > 0) / (y * w)
            if row_frac >= BORDER_MIN_FRAC:
                bot = min(bot, h - y)
            else:
                break

    # Only trim if meaningful
    if left > 5 or right < w-5 or top > 5 or bot < h-5:
        trimmed = crop[top:bot, left:right]
        if trimmed.size > 0:
            log.debug(
                f"[trim] l={left} r={w-right} t={top} b={h-bot} "
                f"{w}×{h} → {trimmed.shape[1]}×{trimmed.shape[0]}"
            )
            return trimmed

    return crop


# ══════════════════════════════════════════════════════════════════
# FIX-5  TABLE DETECTOR (improved)
# ══════════════════════════════════════════════════════════════════

def _is_table(crop: np.ndarray) -> bool:
    """
    Return True if crop looks like a data table.
    Catches: beige alternating rows, grid lines, blue headers.
    """
    h, w = crop.shape[:2]
    if h < 60 or w < 100:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Beige/cream row coverage
    beige      = cv2.inRange(hsv, BEIGE_HSV_LO, BEIGE_HSV_HI)
    beige_frac = np.sum(beige > 0) / (h * w)
    if beige_frac > 0.18:
        log.debug(f"[table] beige_frac={beige_frac:.2f} → skip")
        return True

    # Many horizontal lines
    gray  = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hk    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w//5, 20), 1))
    hline = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk)
    hline_count = int(np.sum(np.sum(hline, axis=1) > 0))
    if hline_count > 6:
        log.debug(f"[table] hline_count={hline_count} → skip")
        return True

    # Blue/teal table header row (top 20%)
    blue_header = cv2.inRange(
        hsv[:max(h//5,1), :], TABLE_BLUE_LO, TABLE_BLUE_HI
    )
    if np.sum(blue_header > 0) / max(w * h // 5, 1) > 0.12:
        log.debug(f"[table] blue_header → skip")
        return True

    # Alternating light/dark rows = table
    # Check horizontal band variance
    row_means = np.mean(gray, axis=1)
    if len(row_means) > 10:
        diffs     = np.abs(np.diff(row_means))
        alt_score = float(np.mean(diffs > 8))
        if alt_score > 0.25 and beige_frac > 0.08:
            log.debug(f"[table] alternating rows alt={alt_score:.2f} → skip")
            return True

    return False


# ══════════════════════════════════════════════════════════════════
# NOISE WIDGET DETECTOR
# ══════════════════════════════════════════════════════════════════

def _is_noise_widget(crop: np.ndarray) -> bool:
    """
    Detect: pie chart widget, progress indicators.
    """
    h, w = crop.shape[:2]

    if w < MIN_REAL_W_PX or h < MIN_REAL_H_PX:
        return True

    aspect = w / max(h, 1)
    if aspect > 5.0 and h < 60:
        return True

    hsv        = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    cream      = cv2.inRange(hsv,
                             np.array([20, 5, 220]),
                             np.array([45, 50, 255]))
    green_pie  = cv2.inRange(hsv, PIE_GREEN_LO, PIE_GREEN_HI)
    cream_frac = np.sum(cream > 0) / (h * w)
    green_frac = np.sum(green_pie > 0) / (h * w)

    if cream_frac > 0.50 and green_frac > 0.03:
        return True

    gray       = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    white_frac = np.sum(gray > 245) / (h * w)
    if white_frac > 0.92 and h < 120:
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# MASTER FILTER — run all checks in order
# ══════════════════════════════════════════════════════════════════

def _should_skip(crop: np.ndarray, label: str = "") -> bool:
    """
    Run all filters. Return True = skip this image.
    Order: cheapest checks first.
    """
    if crop is None or crop.size == 0:
        return True
    if _is_blank(crop):
        log.debug(f"[skip:{label}] blank")
        return True
    if _is_ui_chrome(crop):
        log.debug(f"[skip:{label}] ui_chrome")
        return True
    if _is_noise_widget(crop):
        log.debug(f"[skip:{label}] noise_widget")
        return True
    if _is_table(crop):
        log.debug(f"[skip:{label}] table")
        return True
    if _is_logo(crop):
        log.debug(f"[skip:{label}] logo")
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# STRATEGY A — pdfplumber embedded images
# ══════════════════════════════════════════════════════════════════

def _filter_real_images(page: pdfplumber.page.Page) -> List[dict]:
    pw, ph = float(page.width), float(page.height)
    return [
        im for im in page.images
        if (float(im.get("width",  0)) >= IMG_MIN_W_PT
            and float(im.get("height", 0)) >= IMG_MIN_H_PT
            and float(im.get("width",  0)) <= pw * IMG_MAX_W_FRAC
            and float(im.get("height", 0)) <= ph * IMG_MAX_H_FRAC)
    ]


def _crop_pdfplumber_image(
    page    : pdfplumber.page.Page,
    im      : dict,
    page_rgb: np.ndarray,
) -> np.ndarray:
    px_h, px_w = page_rgb.shape[:2]
    sx = px_w / float(page.width)
    sy = px_h / float(page.height)
    x0 = max(0,    int(float(im["x0"])     * sx) - 3)
    y0 = max(0,    int(float(im["top"])     * sy) - 3)
    x1 = min(px_w, int(float(im["x1"])     * sx) + 3)
    y1 = min(px_h, int(float(im["bottom"]) * sy) + 3)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return page_rgb[y0:y1, x0:x1]


# ══════════════════════════════════════════════════════════════════
# STRATEGY B — OpenCV connected components
# ══════════════════════════════════════════════════════════════════

def _remove_marrow_noise(roi: np.ndarray) -> np.ndarray:
    result = roi.copy()
    hsv    = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, w   = roi.shape[:2]
    wm = cv2.inRange(hsv, np.array([100,10,180]), np.array([130,80,255]))
    wm = cv2.dilate(wm, cv2.getStructuringElement(cv2.MORPH_RECT,(50,20)))
    result[wm > 0] = [255,255,255]
    green = cv2.inRange(hsv, np.array([60,80,80]),    np.array([90,255,255]))
    red1  = cv2.inRange(hsv, np.array([0,100,100]),   np.array([10,255,255]))
    red2  = cv2.inRange(hsv, np.array([170,100,100]), np.array([180,255,255]))
    icons = cv2.bitwise_or(green, cv2.bitwise_or(red1, red2))
    cnts, _ = cv2.findContours(icons, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if 100 < cv2.contourArea(c) < 3000:
            x_,y_,bw_,bh_ = cv2.boundingRect(c)
            result[y_:y_+bh_, x_:x_+bw_] = [255,255,255]
    beige = cv2.inRange(hsv, np.array([20,8,210]), np.array([45,70,255]))
    rows  = np.where(beige.sum(axis=1) > w*0.4*255)[0]
    if len(rows):
        result[rows[0]:rows[-1]+2,:] = [255,255,255]
    result[:int(h*0.03),:] = [255,255,255]
    return result


def _is_pearl_card(crop: np.ndarray) -> bool:
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    teal = cv2.inRange(hsv, TEAL_HSV_LO, TEAL_HSV_HI)
    h, w = crop.shape[:2]
    edge = max(8, min(16, h // 15))
    bpx  = (int(teal[:edge,:].sum()) + int(teal[-edge:,:].sum()) +
            int(teal[:,:edge].sum()) + int(teal[:,-edge:].sum())) // 255
    return bpx > 2*(w+h)*0.08


def _classify_card(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    if w < MIN_CARD_DIM_PX or h < MIN_CARD_DIM_PX:
        return 'text'
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if _is_pearl_card(crop):                           return 'pearl'
    pink = cv2.inRange(hsv, np.array([0,50,150]),  np.array([15,200,255]))
    pb   = (int(pink[:6,:].sum())+int(pink[-6:,:].sum()) +
            int(pink[:,:6].sum())+int(pink[:,-6:].sum())) // 255
    if pb > 2*(w+h)*0.10:                             return 'clinical'
    if np.sum(gray < 80)  / gray.size > 0.45:         return 'usg'
    if np.sum(gray > 240) / gray.size > 0.65:         return 'diagram'
    return 'text'


def _merge_continuous_cards(
    cards: List[Tuple], gap: int = PEARL_GAP_PX
) -> List[Tuple]:
    if not cards:
        return cards
    cards  = sorted(cards, key=lambda c: c[0])
    merged = []
    i = 0
    while i < len(cards):
        cur = cards[i]
        if i+1 < len(cards):
            nxt = cards[i+1]
            if (nxt[0]-cur[1] <= gap
                    and cur[5] in ('diagram','pearl')
                    and nxt[5] in ('diagram','pearl')):
                tw = max(cur[4].shape[1], nxt[4].shape[1])
                def _pad(a, tw=tw):
                    if a.shape[1] < tw:
                        p = np.full(
                            (a.shape[0], tw-a.shape[1], 3),
                            255, dtype=np.uint8
                        )
                        return np.hstack([a, p])
                    return a
                combined = np.vstack([_pad(cur[4]), _pad(nxt[4])])
                merged.append((
                    cur[0], nxt[1],
                    min(cur[2],nxt[2]), max(cur[3],nxt[3]),
                    combined, 'diagram+pearl'
                ))
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def _detect_opencv_cards(
    page    : pdfplumber.page.Page,
    page_rgb: np.ndarray,
) -> List[Tuple]:
    px_h, px_w = page_rgb.shape[:2]
    sx = float(page.width)  / px_w
    sy = float(page.height) / px_h

    top_px   = int(px_h * SAFE_TOP_FRAC)
    bot_px   = int(px_h * SAFE_BOT_FRAC)
    roi      = page_rgb[top_px:bot_px, :]
    roi_h, roi_w = roi.shape[:2]

    roi_clean = _remove_marrow_noise(roi)
    gray      = cv2.cvtColor(roi_clean, cv2.COLOR_RGB2GRAY)
    _, mask   = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY_INV)
    k         = cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (MORPH_CLOSE_PX, MORPH_CLOSE_PX))
    mask      = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask      = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
        iterations=2,
    )

    n_labels,_,stats,_ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    page_area  = roi_h * roi_w
    cards_raw  : List[Tuple] = []
    seen_boxes : List[Tuple] = []

    for i in range(1, n_labels):
        x   = stats[i, cv2.CC_STAT_LEFT]
        y   = stats[i, cv2.CC_STAT_TOP]
        cw  = stats[i, cv2.CC_STAT_WIDTH]
        ch_ = stats[i, cv2.CC_STAT_HEIGHT]
        area= stats[i, cv2.CC_STAT_AREA]

        if area < page_area * MIN_AREA_FRAC:               continue
        if cw < MIN_CARD_DIM_PX or ch_ < MIN_CARD_DIM_PX: continue
        if cw / roi_w > MAX_WIDTH_FRAC:                    continue
        if ch_ < MIN_HEIGHT_PX:                            continue
        if max(cw,ch_) / max(min(cw,ch_),1) > MAX_ASPECT: continue

        overlap = any(
            max(0, min(x+cw,  sx_+sw) - max(x,  sx_)) *
            max(0, min(y+ch_, sy_+sh) - max(y,  sy_))
            > OVERLAP_THRESH * cw * ch_
            for sx_,sy_,sw,sh in seen_boxes
        )
        if overlap:
            continue
        seen_boxes.append((x, y, cw, ch_))

        pad  = 8
        crop = roi[
            max(0, y-pad)   : min(roi_h, y+ch_+pad),
            max(0, x-pad)   : min(roi_w, x+cw+pad),
        ]

        # ── Master filter ─────────────────────────────────────────
        if _should_skip(crop, label="cv"):
            continue

        # ── Trim coloured borders ─────────────────────────────────
        crop = _trim_borders(crop)
        if crop.size == 0:
            continue

        card_type = _classify_card(crop)
        if card_type == 'text':
            continue

        y_top_pt = (y       + top_px) * sy
        y_bot_pt = (y + ch_ + top_px) * sy
        x0_pt    =  x                 * sx
        x1_pt    = (x + cw)           * sx

        cards_raw.append((
            y_top_pt, y_bot_pt, x0_pt, x1_pt, crop, card_type
        ))

    return _merge_continuous_cards(cards_raw)


# ══════════════════════════════════════════════════════════════════
# EXTRACT IMAGE SLOTS
# ══════════════════════════════════════════════════════════════════

def extract_image_slots(
    page        : pdfplumber.page.Page,
    output_dir  : str,
    fname_prefix: str,
    seen_hashes : set,
    dpi         : int = RENDER_DPI,
) -> Tuple[List[ImageSlot], List[Tuple]]:
    os.makedirs(output_dir, exist_ok=True)
    page_rgb = _render_page_rgb(page, dpi=dpi)

    slots    : List[ImageSlot] = []
    cv_bboxes: List[Tuple]     = []
    counter  = 0

    # Strategy A: pdfplumber embedded images
    for im in sorted(_filter_real_images(page), key=lambda x: x["top"]):
        crop = _crop_pdfplumber_image(page, im, page_rgb)
        if _should_skip(crop, label="img-A"):
            continue
        crop = _trim_borders(crop)
        if crop.size == 0:
            continue

        img_hash = hashlib.md5(crop.tobytes()).hexdigest()[:12]
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)

        card_type = _classify_card(crop)
        if card_type == 'text':
            continue

        counter += 1
        fname    = (
            f"{fname_prefix}_img{counter:02d}_{card_type}_{img_hash}.png"
        )
        w, h = _save_png(crop, os.path.join(output_dir, fname))

        top, bottom = float(im["top"]),  float(im["bottom"])
        x0,  x1     = float(im["x0"]),   float(im["x1"])

        slots.append(ImageSlot(
            top=top, bottom=bottom, x0=x0, x1=x1,
            file=fname, tag=f"[IMG:{fname}]",
            card_type=card_type, source="pdfplumber",
            width_px=w, height_px=h,
        ))
        cv_bboxes.append((x0, top, x1, bottom, card_type))
        log.debug(f"[img-A] saved {fname} {w}×{h}px")

    # Strategy B: OpenCV cards
    for y_top_pt, y_bot_pt, x0_pt, x1_pt, crop, card_type in \
            _detect_opencv_cards(page, page_rgb):

        already = any(
            abs(s.top - y_top_pt) < 20
            and abs(s.bottom - y_bot_pt) < 20
            for s in slots
        )
        if already:
            continue

        img_hash = hashlib.md5(crop.tobytes()).hexdigest()[:12]
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)

        counter += 1
        fname    = (
            f"{fname_prefix}_img{counter:02d}_{card_type}_{img_hash}.png"
        )
        w, h = _save_png(crop, os.path.join(output_dir, fname))

        slots.append(ImageSlot(
            top=y_top_pt, bottom=y_bot_pt,
            x0=x0_pt,     x1=x1_pt,
            file=fname,   tag=f"[IMG:{fname}]",
            card_type=card_type, source="opencv",
            width_px=w, height_px=h,
        ))
        cv_bboxes.append((x0_pt, y_top_pt, x1_pt, y_bot_pt, card_type))
        log.debug(f"[img-B] saved {fname} {w}×{h}px")

    slots.sort(key=lambda s: s.top)
    return slots, cv_bboxes


# ══════════════════════════════════════════════════════════════════
# BUILD CONTENT STREAM
# ══════════════════════════════════════════════════════════════════

def build_content_stream(
    text_lines : List[dict],
    image_slots: List[ImageSlot],
) -> List[ContentItem]:
    stream: List[ContentItem] = []
    for ln in text_lines:
        stream.append(ContentItem(
            type       = "text",
            top        = float(ln["top"]),
            text       = ln.get("plain", ""),
            html       = ln.get("html",  ""),
            plain      = ln.get("plain", ""),
            font_sizes = ln.get("font_sizes", []),
        ))
    for im in image_slots:
        stream.append(ContentItem(
            type      = "image",
            top       = im.top,
            tag       = im.tag,
            file      = im.file,
            card_type = im.card_type,
        ))
    stream.sort(key=lambda x: x.top)
    return stream


# ═════════════════════���════════════════════════════════════════════
# PARAGRAPH HEALER
# ══════════════════════════════════════════════════════════════════

def heal_paragraphs(stream: List[ContentItem]) -> List[Paragraph]:
    OPTION_RE      = re.compile(r'^[A-D][\.\)]\s*\S')
    EXPLANATION_RE = re.compile(
        r'^(Explanation|Solution|Discussion)\b', re.IGNORECASE
    )
    paragraphs: List[Paragraph] = []
    cur_html  : List[str]       = []
    cur_plain : List[str]       = []
    cur_top   : float           = 0.0
    cur_bot   : float           = 0.0

    def _flush():
        nonlocal cur_html, cur_plain, cur_top, cur_bot
        if cur_html:
            paragraphs.append(Paragraph(
                type  = "text",
                top   = cur_top,
                html  = " ".join(cur_html).strip(),
                plain = " ".join(cur_plain).strip(),
            ))
        cur_html, cur_plain = [], []
        cur_top = cur_bot   = 0.0

    for item in stream:
        if item.type == "image":
            _flush()
            paragraphs.append(Paragraph(
                type      = item.type,
                top       = item.top,
                tag       = item.tag,
                file      = item.file,
                card_type = item.card_type,
            ))
            continue

        plain = item.plain.strip()
        html  = item.html.strip()
        if not plain:
            continue

        force_break = (
            OPTION_RE.match(plain)
            or EXPLANATION_RE.match(plain)
        )
        max_sz = max(item.font_sizes) if item.font_sizes else 12

        if not cur_html:
            cur_html  = [html]
            cur_plain = [plain]
            cur_top   = item.top
            cur_bot   = item.top + max_sz * 1.25
        else:
            gap = item.top - cur_bot
            if force_break or gap > LINE_GAP_JOIN_PT:
                _flush()
                cur_html  = [html]
                cur_plain = [plain]
                cur_top   = item.top
                cur_bot   = item.top + max_sz * 1.25
            else:
                sep = "<br>" if gap > 4 else " "
                cur_html.append(sep + html)
                cur_plain.append(("\n" if gap > 4 else " ") + plain)
                cur_bot = item.top + max_sz * 1.25

    _flush()
    return paragraphs


def paragraphs_to_html(paragraphs, img_prefix="images/"):
    parts = []
    for p in paragraphs:
        if p.type == "image":
            parts.append(
                f'<figure>'
                f'<img src="{img_prefix}{p.file}" '
                f'alt="{p.card_type}" style="max-width:100%">'
                f'</figure>'
            )
        elif p.html.strip():
            parts.append(f"<p>{p.html}</p>")
    return "\n".join(parts)


def paragraphs_to_plain(paragraphs):
    parts = []
    for p in paragraphs:
        if p.type == "image":
            parts.append(p.tag)
        elif p.plain.strip():
            parts.append(p.plain.strip())
    return "\n\n".join(parts)



# ══════════════════════════════════════════════════════════════════
# PAGE REGION CROPPER — question.png + explanation.png
# ══════════════════════════════════════════════════════════════════

import re as _re
_EXP_RE = _re.compile(
    r'^(Explanation|Solution|Discussion|Ans|Answer)',
    _re.IGNORECASE,
)
_OPT_RE = _re.compile(r'^[A-D][\.\)]\s*\S')


def crop_page_regions(
    page_rgb  : np.ndarray,
    paragraphs: List[Paragraph],
    page_height_pt: float,
    dpi       : int = RENDER_DPI,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Split a rendered page into two crops:
      question_crop   — stem + options  (top of page → explanation start)
      explanation_crop — explanation text + diagrams (explanation start → bottom)

    Returns (question_crop, explanation_crop).
    Either may be None if the page has no clear split.

    Coordinate mapping: PDF points → pixels = pt * (dpi / 72)
    """
    px_h, px_w = page_rgb.shape[:2]
    scale      = px_h / max(page_height_pt, 1.0)   # pixels per PDF point

    # ── Find explanation split y (in PDF points) ─────────────────
    exp_top_pt : Optional[float] = None

    for p in paragraphs:
        if p.type == "text" and _EXP_RE.match(p.plain.strip()):
            exp_top_pt = p.top
            break

    # If no explicit "Explanation" header, split after last option row
    if exp_top_pt is None:
        last_opt_top: Optional[float] = None
        for p in paragraphs:
            if p.type == "text" and _OPT_RE.match(p.plain.strip()):
                last_opt_top = p.top
        if last_opt_top is not None:
            # Give a small gap below last option
            exp_top_pt = last_opt_top + 30

    # ── Convert to pixel row ──────────────────────────────────────
    # Preserve a 6px margin above the split
    MARGIN_PX = 6
    if exp_top_pt is not None:
        split_px = max(MARGIN_PX, min(int(exp_top_pt * scale) - MARGIN_PX, px_h - MARGIN_PX))
    else:
        # No split found — whole page is question
        return page_rgb.copy(), None

    question_crop    = page_rgb[:split_px]
    explanation_crop = page_rgb[split_px:]

    # Drop crops that are basically empty
    if question_crop.shape[0] < 40:
        question_crop = None
    if explanation_crop is not None and explanation_crop.shape[0] < 40:
        explanation_crop = None

    return question_crop, explanation_crop


def save_region(
    crop      : Optional[np.ndarray],
    path      : str,
    min_height: int = 40,
) -> bool:
    """Save a crop as PNG. Returns True if saved, False if skipped."""
    if crop is None or crop.shape[0] < min_height:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _save_png(crop, path)
    return True

# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════

def process_full_page(
    page        : pdfplumber.page.Page,
    output_dir  : str,
    fname_prefix: str,
    seen_hashes : set,
    dpi         : int = RENDER_DPI,
) -> Optional[PageContent]:
    if is_schema_page(page):
        return None

    slots, cv_bboxes = extract_image_slots(
        page, output_dir, fname_prefix, seen_hashes, dpi
    )
    noise_result = run_pipeline(page, image_bboxes=cv_bboxes)
    if noise_result is None:
        return None

    stream     = build_content_stream(noise_result.lines, slots)
    paragraphs = heal_paragraphs(stream)

    # Render full page for region cropping (used by extract.py)
    page_rgb = _render_page_rgb(page, dpi=dpi)

    return PageContent(
        paragraphs   = paragraphs,
        image_slots  = slots,
        cv_bboxes    = cv_bboxes,
        html         = paragraphs_to_html(paragraphs),
        plain        = paragraphs_to_plain(paragraphs),
        answer       = noise_result.answer,
        page_rgb     = page_rgb,
        page_height_pt = float(page.height),
    )