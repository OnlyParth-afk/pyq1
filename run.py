"""
run.py — Single command launcher
python run.py  →  extract + DB + web + browser
"""
from __future__ import annotations
import argparse, logging, os, sys, threading, time, webbrowser
from pathlib import Path

import db          as DB
import progress_db as PDB
from app import create_app

DEFAULT_PDF_DIR  = r"D:\Study meteial\NEET PG QBANK\pyq"
DEFAULT_OUT_DIR  = r"D:\Study meteial\NEET PG QBANK\p2e\output"
DEFAULT_PORT     = 5000

def main():
    ap = argparse.ArgumentParser(description="P2E — Full Stack Runner")
    ap.add_argument("--pdfs",           default=DEFAULT_PDF_DIR)
    ap.add_argument("--out",            default=DEFAULT_OUT_DIR)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--skip-extract",   action="store_true")
    ap.add_argument("--no-browser",     action="store_true")
    ap.add_argument("--debug",          action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img_dir      = os.path.join(args.out, "images")
    marrow_db    = os.path.join(args.out, "marrow.db")
    progress_db  = os.path.join(args.out, "progress.db")
    os.makedirs(img_dir, exist_ok=True)

    logging.basicConfig(
        level   = logging.DEBUG if args.debug else logging.INFO,
        format  = "%(levelname)s %(name)s: %(message)s",
        handlers= [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.out, "run.log"),
                                encoding="utf-8"),
        ],
    )
    log = logging.getLogger("run")

    # ── EXTRACTION ──────────────────────────────────────────────────
    if not args.skip_extract:
        from marrow_pipeline import process_folder
        log.info("Starting extraction…")
        process_folder(args.pdfs, args.out)
        log.info("Extraction done.")

    # ── INIT FSRS CARDS ─────────────────────────────────────────────
    log.info("Initialising FSRS cards…")
    mconn = DB.init_db(marrow_db)
    pconn = PDB.init_progress_db(progress_db)
    rows  = mconn.execute("SELECT question_id FROM questions").fetchall()
    qids  = [r["question_id"] for r in rows]
    PDB.init_new_cards(pconn, qids)
    mconn.close(); pconn.close()
    log.info(f"FSRS cards ready for {len(qids)} questions.")

    # ── FLASK APP ───────────────────────────────────────────────────
    app = create_app(
    marrow_db_path   = "output/marrow.db",
    progress_db_path = "output/progress.db",
    images_dir       = "output",
    output_dir       = "output",
    )
    url = f"http://127.0.0.1:{args.port}"

    print(f"\n{'='*55}")
    print(f"  URL:      {url}")
    print(f"  DB:       {marrow_db}")
    print(f"  Progress: {progress_db}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*55}\n")

    if not args.no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    app.run(host="0.0.0.0", port=args.port,
            debug=args.debug, use_reloader=False)

if __name__ == "__main__":
    main()