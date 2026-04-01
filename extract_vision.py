"""
extract_vision.py — Local Vision Extractor for Marrow NEET PG PDFs
═══════════════════════════════════════════════════════════════════
Uses Ollama (free, runs locally) to extract MCQ data from each page.
No API key. No quota. No cost. Works offline.

Setup (one time):
  1. Install Ollama  → https://ollama.com/download  (Windows installer)
  2. Open CMD and run:
       ollama pull llava
  3. pip install pillow pypdfium2 requests

Usage:
  py extract_vision.py
  py extract_vision.py --pdf "D:\\path\\to.pdf"
  py extract_vision.py --pdf "..." --redo 12,45,88
  py extract_vision.py --pdf "..." --redo-flagged
  py extract_vision.py --pdf "..." --dry-run
  py extract_vision.py --pdf "..." --model llava:13b
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys, time, traceback, requests
from pathlib import Path

PROMPT = """You are a data extractor. Look at this MCQ exam page image carefully.
Return ONLY a JSON object with these exact keys. No explanation, no markdown.

{
  "page_type": "mcq" or "skip",
  "question": "full question stem text here",
  "options": {
    "A": "option A text",
    "B": "option B text",
    "C": "option C text",
    "D": "option D text"
  },
  "mcq_id": "MF7457 or MC1234 etc, empty string if not visible",
  "topic": "topic tag text, empty string if not visible",
  "cuts": {
    "b1": 0.02,
    "r1": 0.15,
    "r2": 0.22,
    "b2": 0.90
  }
}

Rules:
- page_type = "skip" for cover page, table of contents, schema/data-log page
- page_type = "mcq" for all question pages
- question: stem text only, no A/B/C/D labels
- options: clean text only, strip [84%] (21%) tick cross icons
- mcq_id: format like MF7457 MC3178 MA1234, near orange horizontal line
- topic: subject tag e.g. Rickettsia, HPV, Corynebacterium
- cuts: estimate 4 horizontal cut lines as fractions 0.0-1.0:
    b1 = where question text starts (after top bar)
    r1 = where options start (just above A.)
    r2 = where explanation starts (just below orange line)
    b2 = where explanation ends (bottom of text)
- IMPORTANT: Return ONLY the raw JSON object. No markdown fences. No extra text."""

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "moondream"   # fast, 1.6GB — or try: llava-phi3 (2.9GB), llava (4GB)
DPI           = 200
WEBP_QUALITY  = 92
MAX_RETRIES   = 3


def check_ollama(model: str) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        base   = model.split(":")[0]
        if not any(base in m for m in models):
            print(f"\n  Model '{model}' not pulled yet.")
            print(f"  Run:  ollama pull {model}")
            print(f"  Available: {models}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print("\n  Ollama not running — start with:  ollama serve")
        return False


def score_page(entry: dict) -> tuple[str, list[str]]:
    issues = []
    b1, r1, r2, b2 = entry.get("b1",0), entry.get("r1",0), entry.get("r2",0), entry.get("b2",1)
    opts = entry.get("options", {})
    mcq  = entry.get("mcq_id", "")
    missing = [l for l in "ABCD" if l not in opts]
    if missing:          issues.append(f"Missing options: {', '.join(missing)}")
    if not mcq:          issues.append("MCQ ID not found")
    if (r2-r1) < 0.015:  issues.append(f"Options zone too small ({r2-r1:.3f})")
    if (r2-r1) > 0.35:   issues.append(f"Options zone too large ({r2-r1:.3f})")
    if (b2-r2) < 0.10:   issues.append(f"Explanation too short ({b2-r2:.3f})")
    bad   = len(missing) >= 3 or (r2-r1) < 0.01
    grade = "BAD" if bad else ("WARN" if issues else "GOOD")
    return grade, issues


def render_page(pdf_path: str, page_idx: int, dpi: int = DPI):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    bmp = doc[page_idx].render(scale=dpi/72.0)
    img = bmp.to_pil().convert("RGB")
    doc.close()
    return img


def save_crops(img, q_num: int, b1, r1, r2, b2, img_dir: str):
    W, H  = img.size
    b1_px = max(0,       int(H * b1))
    r1_px = max(b1_px+2, int(H * r1))
    r2_px = max(r1_px+2, int(H * r2))
    b2_px = min(H,       int(H * b2))
    key   = f"q{q_num:03d}"
    os.makedirs(img_dir, exist_ok=True)
    img.crop((0, b1_px, W, r1_px)).save(
        os.path.join(img_dir, f"{key}_question.webp"),
        format="WEBP", quality=WEBP_QUALITY, method=4)
    img.crop((0, r2_px, W, b2_px)).save(
        os.path.join(img_dir, f"{key}_explanation.webp"),
        format="WEBP", quality=WEBP_QUALITY, method=4)


def call_ollama(img, model: str, retries: int = MAX_RETRIES) -> dict | None:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "model":   model,
        "prompt":  PROMPT,
        "images":  [img_b64],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 600},
    }

    for attempt in range(retries):
        try:
            r    = requests.post(OLLAMA_URL, json=payload, timeout=300)
            r.raise_for_status()
            text = r.json().get("response", "").strip()

            # Strip markdown fences
            if "```" in text:
                for part in text.split("```"):
                    p = part.strip().lstrip("json").strip()
                    if p.startswith("{") or p.startswith("["):
                        text = p; break

            # Extract JSON object (prefer {...} over [...])
            s_obj = text.find("{")
            s_arr = text.find("[")
            e_obj = text.rfind("}") + 1
            e_arr = text.rfind("]") + 1

            if s_obj >= 0 and e_obj > s_obj:
                text = text[s_obj:e_obj]
            elif s_arr >= 0 and e_arr > s_arr:
                text = text[s_arr:e_arr]

            data = json.loads(text)

            # Normalize: handle array format [{"page_type":...}] or [{...}]
            if isinstance(data, list):
                data = data[0] if data else {}

            # Normalize options: handle ["A","B","C","D"] or ["A: text", ...]
            # or {"A": "text"} (correct format)
            opts = data.get("options", {})
            if isinstance(opts, list):
                norm = {}
                for item in opts:
                    item = str(item).strip()
                    # "A: some text" or "A. some text" or just "A"
                    import re
                    m = re.match(r'^([A-D])[.:)\s]\s*(.+)', item)
                    if m:
                        norm[m.group(1)] = m.group(2).strip()
                    elif item in "ABCD" and len(item) == 1:
                        norm[item] = ""  # placeholder, will be WARN
                data["options"] = norm

            return data

        except json.JSONDecodeError as e:
            raw = r.json().get("response","")[:200] if "r" in dir() else "?"
            print(f"\n    [JSON error {attempt+1}] {e}  raw: {raw[:100]}")
            if attempt < retries-1: time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"\n    [Timeout {attempt+1}]")
            if attempt < retries-1: time.sleep(5)
        except Exception as ex:
            print(f"\n    [Error {attempt+1}] {ex}")
            if attempt < retries-1: time.sleep(3)
    return None


def write_splits(splits, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


def build_report(results, pdf_path, out_path, model):
    good = [r for r in results if r["grade"]=="GOOD"]
    warn = [r for r in results if r["grade"]=="WARN"]
    bad  = [r for r in results if r["grade"]=="BAD"]
    redo = sorted(r["page_idx"]+1 for r in warn+bad)

    cards = []
    for r in results:
        c  = {"GOOD":"#16a34a","WARN":"#d97706","BAD":"#dc2626"}[r["grade"]]
        bg = {"GOOD":"#f0fdf4","WARN":"#fffbeb","BAD":"#fef2f2"}[r["grade"]]
        opts_html = "".join(
            f"<div style='font-size:11px'><b>{l}.</b> "
            f"{r['options'].get(l,'<i style=\"color:#9ca3af\">—</i>')}</div>"
            for l in "ABCD")
        iss_html = ("".join(f"<li>{i}</li>" for i in r["issues"]))
        si = f"<img src='data:image/webp;base64,{r['stem_b64']}' style='width:100%;border-radius:4px;margin-bottom:4px;'>" if r.get("stem_b64") else ""
        ei = f"<img src='data:image/webp;base64,{r['exp_b64']}' style='width:100%;border-radius:4px;margin-top:4px;'>" if r.get("exp_b64") else ""
        q  = r.get("question","")
        qp = f"<div style='font-size:11px;color:#374151;font-style:italic;margin-bottom:6px'>{q[:120]}{'…' if len(q)>120 else ''}</div>" if q else ""
        cards.append(f"""<div style="background:{bg};border:2px solid {c};border-radius:10px;padding:12px;break-inside:avoid;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <b>Q{r['q_num']} · p{r['page_idx']+1}</b>
    <span style="background:{c};color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;">{r['grade']}</span>
  </div>
  <div style="font-size:11px;color:#374151;margin-bottom:4px;"><b>ID:</b> {r.get('mcq_id') or '—'} &nbsp;·&nbsp; <b>Topic:</b> {r.get('topic') or '—'}</div>
  <div style="font-size:10px;color:#6b7280;margin-bottom:4px;">b1={r['b1']:.3f} r1={r['r1']:.3f} r2={r['r2']:.3f} b2={r['b2']:.3f}</div>
  {qp}{si}{opts_html}{'<ul style="margin:4px 0 0;padding-left:16px;font-size:11px;color:#6b7280;">'+iss_html+'</ul>' if iss_html else ''}{ei}
</div>""")

    redo_str = ",".join(str(p) for p in redo)
    redo_block = (f'<div class="redo"><h2 style="color:#f59e0b;font-size:14px;margin-bottom:4px">⚠ {len(redo)} pages need review</h2>'
                  f'<div class="cmd">py extract_vision.py --pdf &quot;{pdf_path}&quot; --redo {redo_str}</div></div>'
                  if redo else
                  '<div class="redo" style="border:2px solid #16a34a"><h2 style="color:#16a34a;font-size:14px">✓ All pages GOOD</h2></div>')

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>P2E Vision — {Path(pdf_path).stem}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#111;color:#f9fafb;padding:24px}}
h1{{font-size:22px;margin-bottom:4px}}.sub{{color:#9ca3af;font-size:13px;margin-bottom:20px}}
.summary{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
.stat{{background:#1f2937;border-radius:10px;padding:14px 22px;text-align:center}}
.stat .n{{font-size:36px;font-weight:800}}.stat .l{{font-size:12px;color:#9ca3af;margin-top:2px}}
.redo{{background:#1f2937;border-radius:10px;padding:16px;margin-bottom:24px}}
.cmd{{font-family:monospace;font-size:12px;background:#000;padding:10px;border-radius:6px;color:#6ee7b7;word-break:break-all;margin-top:8px}}
.filters{{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}}
.filters button{{padding:6px 16px;border-radius:20px;border:none;cursor:pointer;font-size:12px;font-weight:600}}
.grid{{columns:3 340px;column-gap:16px}}
</style></head><body>
<h1>⚡ P2E Vision Review</h1>
<div class="sub">{Path(pdf_path).stem} · {len(results)} questions · {model}</div>
<div class="summary">
  <div class="stat"><div class="n" style="color:#16a34a">{len(good)}</div><div class="l">GOOD</div></div>
  <div class="stat"><div class="n" style="color:#d97706">{len(warn)}</div><div class="l">WARN</div></div>
  <div class="stat"><div class="n" style="color:#dc2626">{len(bad)}</div><div class="l">BAD</div></div>
  <div class="stat"><div class="n">{len(results)}</div><div class="l">TOTAL</div></div>
  <div class="stat"><div class="n" style="color:#6ee7b7">{sum(1 for r in results if r.get('mcq_id'))}</div><div class="l">MCQ IDs</div></div>
  <div class="stat"><div class="n" style="color:#6ee7b7">{sum(1 for r in results if len(r.get('options',{{}}))==4)}</div><div class="l">All 4 opts</div></div>
</div>
{redo_block}
<div class="filters">
  <button onclick="filter('all')" style="background:#374151;color:#fff">All ({len(results)})</button>
  <button onclick="filter('GOOD')" style="background:#16a34a;color:#fff">Good ({len(good)})</button>
  <button onclick="filter('WARN')" style="background:#d97706;color:#fff">Warn ({len(warn)})</button>
  <button onclick="filter('BAD')" style="background:#dc2626;color:#fff">Bad ({len(bad)})</button>
</div>
<div class="grid" id="grid">{''.join(cards)}</div>
<script>
const cards=document.querySelectorAll('#grid > div');
function filter(g){{cards.forEach(c=>{{const b=c.querySelector('span')?.textContent?.trim();c.style.display=(g==='all'||b===g)?'':'none';}});}}
</script></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📊 Report → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",          default=None)
    ap.add_argument("--out",          default=None)
    ap.add_argument("--dpi",          type=int, default=DPI)
    ap.add_argument("--model",        default=DEFAULT_MODEL)
    ap.add_argument("--redo",         default=None)
    ap.add_argument("--redo-flagged", action="store_true")
    ap.add_argument("--start",        type=int, default=None)
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--no-thumbs",    action="store_true")
    ap.add_argument("--report-only",  action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        if not check_ollama(args.model):
            print("\nSetup:\n  1. https://ollama.com/download")
            print(f"  2. ollama pull {args.model}")
            sys.exit(1)
        print(f"✓ Ollama ready  |  model={args.model}")

    if not args.pdf:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            root.wm_attributes("-topmost", True)
            args.pdf = filedialog.askopenfilename(
                title="Select Marrow PDF",
                filetypes=[("PDF","*.pdf"),("All","*.*")],
                initialdir=r"D:\Study meteial\NEET PG QBANK\pyq")
            root.destroy()
        except Exception: pass
    if not args.pdf:
        args.pdf = input("PDF path: ").strip().strip('"').strip("'")
    if not args.pdf or not os.path.exists(args.pdf):
        print(f"ERROR: {args.pdf!r}"); sys.exit(1)

    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(args.pdf); total = len(doc); doc.close()

    stem        = Path(args.pdf).stem
    base_out    = args.out or str(Path(args.pdf).parent.parent / "p2e" / "output")
    img_dir     = os.path.join(base_out, stem, "images")
    splits_path = os.path.join(base_out, stem, "splits.json")
    report_path = os.path.join(base_out, stem, "review_vision.html")
    log_path    = os.path.join(base_out, stem, "vision_log.txt")
    os.makedirs(img_dir, exist_ok=True)

    existing = {}
    if os.path.exists(splits_path):
        with open(splits_path, encoding="utf-8") as f:
            existing = json.load(f)
    q_max = max((v.get("q_num",0) for v in existing.values() if isinstance(v,dict)), default=0)

    print(f"\n{'='*60}\n  PDF: {args.pdf}\n  Pages: {total}  Done: {len(existing)}  q={q_max}\n{'='*60}\n")

    if args.report_only:
        todo = []
    elif args.redo:
        todo = [int(p.strip())-1 for p in args.redo.split(",") if p.strip().isdigit()]
        print(f"Redo {len(todo)} pages")
    elif getattr(args, "redo_flagged"):
        todo = sorted(int(k) for k,v in existing.items()
                      if isinstance(v,dict) and score_page(v)[0] in ("WARN","BAD"))
        print(f"Redo-flagged: {len(todo)} pages")
    else:
        saved = set(existing.keys())
        start = (args.start-1) if args.start else 0
        todo  = [i for i in range(start, total) if str(i) not in saved]
        if args.dry_run: todo = todo[:5]; print("DRY RUN — 5 pages")
        print(f"To process: {len(todo)}")

    log_lines = []; t_start = time.time()

    for n, page_idx in enumerate(todo, 1):
        print(f"  [{n:4d}/{len(todo)}] p{page_idx+1:4d} ", end="", flush=True)
        t0 = time.time()
        try:
            img  = render_page(args.pdf, page_idx, args.dpi)
            data = call_ollama(img, args.model)
            if data is None:
                print("ERROR"); log_lines.append(f"p{page_idx+1}: ERROR"); continue
            if data.get("page_type") == "skip":
                print("SKIP"); log_lines.append(f"p{page_idx+1}: SKIP"); continue

            cuts  = data.get("cuts", {})
            b1    = max(0.005, min(float(cuts.get("b1", 0.02)),  0.12))
            r1    = max(b1+0.005, min(float(cuts.get("r1", 0.15)), 0.82))
            r2    = max(r1+0.01,  min(float(cuts.get("r2", r1+0.12)), 0.91))
            b2    = max(r2+0.01,  min(float(cuts.get("b2", 0.92)), 0.985))
            opts  = {k:str(v).strip() for k,v in data.get("options",{}).items()
                     if k in "ABCD" and str(v).strip()}
            mcq   = str(data.get("mcq_id","")).strip()
            topic = str(data.get("topic","")).strip()
            quest = str(data.get("question","")).strip()

            grade, issues = score_page({"b1":b1,"r1":r1,"r2":r2,"b2":b2,"options":opts,"mcq_id":mcq})

            if not args.dry_run:
                q_num = q_max+1; q_max += 1
                save_crops(img, q_num, b1, r1, r2, b2, img_dir)
                existing[str(page_idx)] = {
                    "page_idx":page_idx,"q_num":q_num,
                    "b1":b1,"r1":r1,"r2":r2,"b2":b2,
                    "options":opts,"answer":"","mcq_id":mcq,
                    "topic":topic,"question":quest,
                    "session_meta":{},"appended":0,"auto":True,"fmt":"webp"}
                write_splits(existing, splits_path)

            sym = {"GOOD":"✓","WARN":"⚠","BAD":"✗"}[grade]
            print(f"{sym} {grade:<4}  id={mcq or '—':10s}  opts={''.join(opts.keys()):4s}  ({time.time()-t0:.1f}s)")
            for iss in issues: print(f"          → {iss}")
            log_lines.append(f"p{page_idx+1} Q{q_max} {grade}: {'; '.join(issues) or 'ok'}")

        except Exception as e:
            print(f"ERROR — {e}"); log_lines.append(f"p{page_idx+1}: ERROR {e}")
            traceback.print_exc()

    elapsed = time.time()-t_start
    splits  = {k:v for k,v in existing.items() if isinstance(v,dict)}
    print(f"\n{'='*60}\n  {len(todo)} pages in {elapsed/60:.1f}min  |  Total: {len(splits)} questions\n{'='*60}")

    with open(log_path,"w",encoding="utf-8") as f:
        f.write(f"Vision Log — {stem}\n{len(todo)} pages {elapsed:.0f}s\n\n"+"\n".join(log_lines))
    print(f"📝 {log_path}")

    all_results = []
    for k,v in sorted(splits.items(), key=lambda x: x[1].get("q_num",0)):
        grade,issues = score_page(v)
        entry = {**v,"grade":grade,"issues":issues,"stem_b64":"","exp_b64":""}
        if not args.no_thumbs and not args.dry_run:
            try:
                from PIL import Image as PILImage
                for suf,key in [("question","stem_b64"),("explanation","exp_b64")]:
                    p = os.path.join(img_dir, f"q{v['q_num']:03d}_{suf}.webp")
                    if os.path.exists(p):
                        pil = PILImage.open(p).convert("RGB"); pil.thumbnail((340,9999))
                        buf = io.BytesIO(); pil.save(buf,format="WEBP",quality=75)
                        entry[key] = base64.b64encode(buf.getvalue()).decode()
            except Exception: pass
        all_results.append(entry)

    gn = sum(1 for r in all_results if r["grade"]=="GOOD")
    wn = sum(1 for r in all_results if r["grade"]=="WARN")
    bn = sum(1 for r in all_results if r["grade"]=="BAD")
    redo = sorted(r["page_idx"]+1 for r in all_results if r["grade"] in ("WARN","BAD"))
    print(f"\n  ✓ GOOD={gn}  ⚠ WARN={wn}  ✗ BAD={bn}")
    if redo:
        print(f'\n  py extract_vision.py --pdf "{args.pdf}" --redo {",".join(str(p) for p in redo)}')

    if not args.dry_run:
        build_report(all_results, args.pdf, report_path, args.model)
        try: import webbrowser; webbrowser.open(report_path)
        except Exception: pass
    print("\n  Done ✓\n")


if __name__ == "__main__":
    main()
