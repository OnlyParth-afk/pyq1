"""
run_auto.py — P2E Headless Batch Processor
═══════════════════════════════════════════
Processes an entire Marrow PDF automatically, no browser needed.

Usage:
    py run_auto.py
    py run_auto.py --pdf "D:\\path\\to\\file.pdf"
    py run_auto.py --pdf "..." --redo 12,45,88    (redo specific pages)
    py run_auto.py --pdf "..." --redo-flagged      (redo all flagged pages)

At the end opens review.html showing:
    • Every question: stem / explanation thumbnail + extracted data
    • Each page scored: GOOD / WARN / BAD
    • Redo list you can paste into --redo next run

Confidence scoring (per page):
    GOOD  — all 4 options found, MCQ ID found, sensible cut zones
    WARN  — 1–2 options missing OR MCQ ID missing OR borderline zones
    BAD   — 3+ options missing OR options zone too small/large
             OR explanation too short (crop probably wrong)
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys, time, traceback
from pathlib import Path

# ── Thresholds for flagging ───────────────────────────────────────
MIN_OPT_ZONE   = 0.025   # r2-r1 must be >= this  (else options zone too tiny)
MAX_OPT_ZONE   = 0.30    # r2-r1 must be <= this  (else explanation bled in)
MIN_EXP_ZONE   = 0.12    # b2-r2 must be >= this  (else explanation too short)
MIN_STEM_ZONE  = 0.008   # r1-b1 must be >= this  (else stem too tiny)


# ─────────────────────────────────────────────────────────────────
#  Bootstrap — reuse all logic from splitter.py
# ─────────────────────────────────────────────────────────────────
def _bootstrap(pdf_path: str, out_dir: str | None, dpi: int):
    """Initialise splitter.py STATE and return it ready to use."""
    # Import splitter from same directory
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    import splitter

    import pypdfium2 as pdfium
    doc   = pdfium.PdfDocument(pdf_path)
    total = len(doc)
    doc.close()

    stem     = Path(pdf_path).stem
    base_out = out_dir or str(Path(pdf_path).parent.parent / "p2e" / "output")
    img_dir  = os.path.join(base_out, stem, "images")
    os.makedirs(img_dir, exist_ok=True)

    splits_path = os.path.join(base_out, stem, "splits.json")
    existing    = {}
    if os.path.exists(splits_path):
        with open(splits_path, encoding="utf-8") as f:
            existing = json.load(f)
        q_max = max(
            (v.get("q_num", 0) for v in existing.values() if isinstance(v, dict)),
            default=0,
        )
    else:
        q_max = 0

    splitter.STATE.update({
        "pdf_path"   : pdf_path,
        "total"      : total,
        "current"    : 0,
        "q_counter"  : q_max,
        "img_dir"    : img_dir,
        "splits_path": splits_path,
        "splits"     : existing,
        "dpi"        : dpi,
        "page_cache" : {},
    })
    splitter._load_cut_model()
    splitter.auto_skip_to_first_mcq()
    return splitter


# ─────────────────────────────────────────────────────────────────
#  Confidence scoring
# ─────────────────────────────────────────────────────────────────
def score_page(entry: dict) -> tuple[str, list[str]]:
    """
    Returns (grade, issues) where grade is 'GOOD' / 'WARN' / 'BAD'.
    issues is a list of human-readable problem strings.
    """
    issues = []

    b1 = entry.get("b1", 0)
    r1 = entry.get("r1", 0)
    r2 = entry.get("r2", 0)
    b2 = entry.get("b2", 1)
    opts = entry.get("options", {})
    mcq  = entry.get("mcq_id", "")

    opt_zone  = r2 - r1
    exp_zone  = b2 - r2
    stem_zone = r1 - b1

    missing_opts = [l for l in "ABCD" if l not in opts]

    if missing_opts:
        issues.append(f"Missing options: {', '.join(missing_opts)}")
    if not mcq:
        issues.append("MCQ ID not found")
    if opt_zone < MIN_OPT_ZONE:
        issues.append(f"Options zone too small ({opt_zone:.3f} < {MIN_OPT_ZONE})")
    if opt_zone > MAX_OPT_ZONE:
        issues.append(f"Options zone too large ({opt_zone:.3f} > {MAX_OPT_ZONE})")
    if exp_zone < MIN_EXP_ZONE:
        issues.append(f"Explanation zone too short ({exp_zone:.3f} < {MIN_EXP_ZONE})")
    if stem_zone < MIN_STEM_ZONE:
        issues.append(f"Stem zone tiny ({stem_zone:.3f}) — possible blank page")

    bad_count = len(missing_opts) + (1 if opt_zone < MIN_OPT_ZONE or opt_zone > MAX_OPT_ZONE else 0)
    if bad_count >= 3 or (len(missing_opts) >= 3) or opt_zone < 0.01:
        grade = "BAD"
    elif issues:
        grade = "WARN"
    else:
        grade = "GOOD"

    return grade, issues


# ─────────────────────────────────────────────────────────────────
#  Image crop helper
# ─────────────────────────────────────────────────────────────────
def _crop_b64(splitter, page_idx: int, y0_frac: float, y1_frac: float,
              thumb_w: int = 340) -> str:
    """Render a y-fraction crop of a page and return as base64 JPEG."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image
        import numpy as np
        doc = pdfium.PdfDocument(splitter.STATE["pdf_path"])
        bmp = doc[page_idx].render(scale=100/72.0)
        img = bmp.to_pil().convert("RGB")
        doc.close()
        W, H = img.size
        y0 = max(0, int(y0_frac * H))
        y1 = min(H, int(y1_frac * H))
        if y1 <= y0:
            return ""
        crop = img.crop((0, y0, W, y1))
        # Resize to thumb width
        ratio = thumb_w / W
        crop  = crop.resize((thumb_w, max(1, int(crop.height * ratio))),
                            Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return ""


# ─────────────────────────────────────────────────────────────────
#  HTML report generator
# ─────────────────────────────────────────────────────────────────
def build_report(results: list[dict], pdf_path: str, out_path: str):
    good  = [r for r in results if r["grade"] == "GOOD"]
    warn  = [r for r in results if r["grade"] == "WARN"]
    bad   = [r for r in results if r["grade"] == "BAD"]
    redo_pages = sorted(r["page_idx"] + 1 for r in warn + bad)  # 1-based

    cards = []
    for r in results:
        color = {"GOOD": "#16a34a", "WARN": "#d97706", "BAD": "#dc2626"}[r["grade"]]
        bg    = {"GOOD": "#f0fdf4", "WARN": "#fffbeb", "BAD": "#fef2f2"}[r["grade"]]

        issues_html = ""
        if r["issues"]:
            issues_html = "<ul style='margin:4px 0 0 0;padding-left:16px;font-size:11px;color:#6b7280;'>" + \
                "".join(f"<li>{i}</li>" for i in r["issues"]) + "</ul>"

        opts_html = ""
        for ltr in "ABCD":
            txt = r["options"].get(ltr, "")
            cl  = "color:#16a34a;font-weight:700;" if ltr == r.get("answer") else ""
            opts_html += (
                f"<div style='font-size:11px;margin:1px 0;{cl}'>"
                f"<b>{ltr}.</b> {txt or '<i style=\"color:#9ca3af\">missing</i>'}</div>"
            )

        stem_img = f"<img src='data:image/jpeg;base64,{r['stem_b64']}' style='width:100%;border-radius:4px;margin-bottom:4px;'>" if r.get("stem_b64") else ""
        exp_img  = f"<img src='data:image/jpeg;base64,{r['exp_b64']}' style='width:100%;border-radius:4px;margin-top:4px;'>" if r.get("exp_b64") else ""

        cards.append(f"""
        <div style="background:{bg};border:2px solid {color};border-radius:10px;
                    padding:12px;break-inside:avoid;margin-bottom:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;font-size:13px;">
              Q{r['q_num']} &nbsp;·&nbsp; p{r['page_idx']+1}
            </span>
            <span style="background:{color};color:#fff;padding:2px 10px;
                         border-radius:12px;font-size:11px;font-weight:700;">
              {r['grade']}
            </span>
          </div>
          <div style="font-size:11px;color:#374151;margin-bottom:4px;">
            <b>ID:</b> {r.get('mcq_id') or '<i style="color:#9ca3af">—</i>'}
            &nbsp;·&nbsp;
            <b>Topic:</b> {r.get('topic') or '<i style="color:#9ca3af">—</i>'}
          </div>
          <div style="font-size:10px;color:#6b7280;margin-bottom:6px;">
            b1={r['b1']:.3f} r1={r['r1']:.3f} r2={r['r2']:.3f} b2={r['b2']:.3f}
            &nbsp;·&nbsp;
            opt_zone={r['r2']-r['r1']:.3f} exp_zone={r['b2']-r['r2']:.3f}
          </div>
          {stem_img}
          {opts_html}
          {issues_html}
          {exp_img}
        </div>""")

    redo_str = ",".join(str(p) for p in redo_pages)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>P2E Review — {Path(pdf_path).stem}</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:system-ui,sans-serif; background:#111; color:#f9fafb; padding:24px; }}
    h1   {{ font-size:22px; margin-bottom:4px; }}
    .sub {{ color:#9ca3af; font-size:13px; margin-bottom:20px; }}
    .summary {{ display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }}
    .stat {{ background:#1f2937; border-radius:10px; padding:14px 22px; text-align:center; }}
    .stat .n {{ font-size:36px; font-weight:800; }}
    .stat .l {{ font-size:12px; color:#9ca3af; margin-top:2px; }}
    .redo-box {{ background:#1f2937; border-radius:10px; padding:16px; margin-bottom:24px; }}
    .redo-box h2 {{ font-size:14px; margin-bottom:8px; color:#f59e0b; }}
    .redo-cmd {{ font-family:monospace; font-size:12px; background:#111;
                 padding:10px 14px; border-radius:6px; color:#6ee7b7;
                 user-select:all; word-break:break-all; }}
    .filters {{ display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; }}
    .filters button {{ padding:6px 16px; border-radius:20px; border:none;
                       cursor:pointer; font-size:12px; font-weight:600; }}
    .grid {{ columns:3 340px; column-gap:16px; }}
    @media(max-width:800px) {{ .grid {{ columns:1; }} }}
  </style>
</head>
<body>
  <h1>⚡ P2E Auto-Process Review</h1>
  <div class="sub">{Path(pdf_path).stem} &nbsp;·&nbsp; {len(results)} questions processed</div>

  <div class="summary">
    <div class="stat"><div class="n" style="color:#16a34a">{len(good)}</div><div class="l">GOOD</div></div>
    <div class="stat"><div class="n" style="color:#d97706">{len(warn)}</div><div class="l">WARN</div></div>
    <div class="stat"><div class="n" style="color:#dc2626">{len(bad)}</div> <div class="l">BAD</div></div>
    <div class="stat"><div class="n">{len(results)}</div><div class="l">TOTAL</div></div>
    <div class="stat"><div class="n" style="color:#6ee7b7">{sum(1 for r in results if r.get('mcq_id'))}</div><div class="l">MCQ IDs found</div></div>
    <div class="stat"><div class="n" style="color:#6ee7b7">{sum(1 for r in results if len(r.get('options',{}))==4)}</div><div class="l">All 4 opts found</div></div>
  </div>

  {'<div class="redo-box"><h2>⚠ Pages needing review (' + str(len(redo_pages)) + ')</h2><div class="redo-cmd">py run_auto.py --pdf &quot;' + pdf_path + '&quot; --redo ' + redo_str + '</div><p style="font-size:11px;color:#9ca3af;margin-top:8px;">Copy the command above to re-process only flagged pages, or open splitter.py and jump to each page manually.</p></div>' if redo_pages else '<div class="redo-box" style="border:2px solid #16a34a"><h2 style="color:#16a34a">✓ All pages look good — no redo needed</h2></div>'}

  <div class="filters">
    <button onclick="filter('all')"  style="background:#374151;color:#fff">All ({len(results)})</button>
    <button onclick="filter('GOOD')" style="background:#16a34a;color:#fff">Good ({len(good)})</button>
    <button onclick="filter('WARN')" style="background:#d97706;color:#fff">Warn ({len(warn)})</button>
    <button onclick="filter('BAD')"  style="background:#dc2626;color:#fff">Bad ({len(bad)})</button>
  </div>

  <div class="grid" id="grid">
    {''.join(cards)}
  </div>

  <script>
    const cards = document.querySelectorAll('#grid > div');
    function filter(grade) {{
      cards.forEach(c => {{
        const badge = c.querySelector('span:last-child')?.textContent?.trim();
        c.style.display = (grade === 'all' || badge === grade) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📊 Report saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="P2E Headless Auto-Processor")
    ap.add_argument("--pdf",         default=None, help="Path to Marrow PDF")
    ap.add_argument("--out",         default=None, help="Output directory")
    ap.add_argument("--dpi",         type=int, default=150)
    ap.add_argument("--redo",        default=None,
                    help="Comma-separated 1-based page numbers to reprocess, e.g. 12,45,88")
    ap.add_argument("--redo-flagged",action="store_true",
                    help="Redo all pages previously marked WARN or BAD")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Predict lines but don't save — just show what would be done")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip processing, just regenerate the review report from existing splits.json")
    ap.add_argument("--no-thumbs",   action="store_true",
                    help="Skip rendering thumbnails for faster report")
    args = ap.parse_args()

    # ── PDF picker ────────────────────────────────────────────────
    if not args.pdf:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.wm_attributes("-topmost", True)
            args.pdf = filedialog.askopenfilename(
                title="Select Marrow PDF",
                filetypes=[("PDF files","*.pdf"),("All","*.*")],
                initialdir=r"D:\Study meteial\NEET PG QBANK\pyq",
            )
            root.destroy()
        except Exception:
            pass

    if not args.pdf:
        args.pdf = input("PDF path: ").strip().strip('"').strip("'")

    if not args.pdf or not os.path.exists(args.pdf):
        print(f"ERROR: not found — {args.pdf!r}"); sys.exit(1)

    # ── Bootstrap splitter ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  P2E Auto-Processor")
    print(f"  PDF : {args.pdf}")
    print(f"{'='*60}\n")

    sp = _bootstrap(args.pdf, args.out, args.dpi)

    splits_path = sp.STATE["splits_path"]
    stem        = Path(args.pdf).stem
    base_out    = str(Path(splits_path).parent)
    report_path = os.path.join(base_out, "review.html")
    log_path    = os.path.join(base_out, "auto_log.txt")

    total = sp.STATE["total"]

    # ── Determine which pages to process ─────────────────────────
    if args.report_only:
        todo = []
        print("--report-only: skipping processing, building report from splits.json")
    elif args.redo:
        # Explicit redo list (1-based page numbers)
        todo = [int(p.strip()) - 1 for p in args.redo.split(",") if p.strip().isdigit()]
        print(f"Redo mode: {len(todo)} pages — {args.redo}")
    elif args.redo_flagged:
        existing = sp.STATE["splits"]
        todo = []
        for k, v in existing.items():
            if isinstance(v, dict):
                grade, _ = score_page(v)
                if grade in ("WARN", "BAD"):
                    todo.append(int(k))
        todo.sort()
        print(f"Redo-flagged mode: {len(todo)} WARN/BAD pages")
    else:
        # Normal run: process all unsaved pages
        saved_keys = set(sp.STATE["splits"].keys())
        todo = []
        for i in range(total):
            if str(i) in saved_keys:
                continue
            try:
                ptype = sp.classify_page(i)
            except Exception:
                ptype = "mcq"
            if ptype == "skip":
                print(f"  skip p{i+1} (cover/catalog/schema)")
                continue
            todo.append(i)
        print(f"Pages to process: {len(todo)}  (already done: {len(sp.STATE['splits'])})\n")

    # ── Process ───────────────────────────────────────────────────
    log_lines = []
    t_start   = time.time()

    for n, page_idx in enumerate(todo, 1):
        t0 = time.time()
        print(f"  [{n:4d}/{len(todo)}] p{page_idx+1:4d} ", end="", flush=True)

        try:
            pred = sp._predict_lines(page_idx)
            if pred is None:
                print("SKIP (no prediction)")
                log_lines.append(f"p{page_idx+1}: SKIP no prediction")
                continue

            b1,r1,r2,b2 = pred["b1"],pred["r1"],pred["r2"],pred["b2"]
            text = sp.extract_page_text(page_idx)
            opts = text.get("options",{})
            mcq  = text.get("mcq_id","")
            topic= text.get("topic","")

            entry = {
                "page_idx":page_idx,"b1":b1,"r1":r1,"r2":r2,"b2":b2,
                "options":opts,"mcq_id":mcq,"topic":topic,
                "answer":"","session_meta":{},"appended":0,"auto":True,
            }
            grade, issues = score_page(entry)

            if not args.dry_run:
                q_num = sp.STATE["q_counter"] + 1
                sp.STATE["q_counter"] = q_num
                entry["q_num"] = q_num
                sp.save_crops(page_idx, q_num, b1, r1, r2, b2, appended_b64s=[])
                sp.STATE["splits"][str(page_idx)] = entry
                sp.STATE["current"] = page_idx + 1
                sp._write_splits()

            dt  = time.time() - t0
            sym = {"GOOD":"✓","WARN":"⚠","BAD":"✗"}[grade]
            print(f"{sym} {grade:<4}  id={mcq or '—':8s}  opts={''.join(opts.keys()):4s}  "
                  f"r1={r1:.3f} r2={r2:.3f}  ({dt:.1f}s)")
            if issues:
                for iss in issues:
                    print(f"          → {iss}")
            log_lines.append(f"p{page_idx+1} Q{entry.get('q_num','?')} {grade}: {'; '.join(issues) or 'ok'}")

        except Exception as e:
            print(f"ERROR — {e}")
            log_lines.append(f"p{page_idx+1}: ERROR {e}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    splits  = {k:v for k,v in sp.STATE["splits"].items() if isinstance(v, dict)}
    print(f"\n{'='*60}")
    print(f"  Processed {len(todo)} pages in {elapsed:.0f}s")
    print(f"  Total questions in splits.json: {len(splits)}")
    print(f"{'='*60}\n")

    # ── Save log ──────────────────────────────────────────────────
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"P2E Auto-Process Log — {stem}\n")
        f.write(f"Processed {len(todo)} pages  ({elapsed:.0f}s)\n\n")
        f.write("\n".join(log_lines))
    print(f"📝 Log saved → {log_path}")

    # ── Score ALL splits (not just this run) ──────────────────────
    all_results = []
    for k, v in sorted(splits.items(), key=lambda x: x[1].get("q_num", 0)):
        grade, issues = score_page(v)
        entry = {**v, "grade": grade, "issues": issues,
                 "stem_b64": "", "exp_b64": ""}
        if not args.no_thumbs and not args.dry_run:
            try:
                entry["stem_b64"] = _crop_b64(sp, v["page_idx"],
                                               v["b1"], v["r1"])
                entry["exp_b64"]  = _crop_b64(sp, v["page_idx"],
                                               v["r2"], v["b2"])
            except Exception:
                pass
        all_results.append(entry)

    good_n = sum(1 for r in all_results if r["grade"]=="GOOD")
    warn_n = sum(1 for r in all_results if r["grade"]=="WARN")
    bad_n  = sum(1 for r in all_results if r["grade"]=="BAD")
    print(f"\n  Grades across all {len(all_results)} questions:")
    print(f"    ✓ GOOD : {good_n}")
    print(f"    ⚠ WARN : {warn_n}")
    print(f"    ✗ BAD  : {bad_n}")

    redo_pages = sorted(r["page_idx"]+1 for r in all_results if r["grade"] in ("WARN","BAD"))
    if redo_pages:
        print(f"\n  To redo flagged pages:")
        print(f"    py run_auto.py --pdf \"{args.pdf}\" --redo {','.join(str(p) for p in redo_pages)}")
    else:
        print("\n  ✓ All questions scored GOOD — no redo needed")

    # ── Build HTML report ─────────────────────────────────────────
    if not args.dry_run:
        build_report(all_results, args.pdf, report_path)
        try:
            import webbrowser
            webbrowser.open(report_path)
        except Exception:
            pass

    print(f"\n  Done. Open review.html to inspect results.")
    print(f"  {report_path}\n")


if __name__ == "__main__":
    main()
