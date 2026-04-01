"""
loader.py — P2E
Loads output/<stem>/clean.json (or raw.json / splits.json) into marrow.db.

Usage:
    python loader.py --stem "5.Micro PYQ (2017-2022)"
    python loader.py --all
    python loader.py --all --replace   # overwrite existing rows
"""
from __future__ import annotations
import argparse, json, logging, os, sys, sqlite3
from datetime import datetime, timezone
from pathlib import Path

import db as DB

log = logging.getLogger(__name__)
DEFAULT_DB  = r"D:\Study meteial\NEET PG QBANK\p2e\output\marrow.db"
DEFAULT_OUT = r"D:\Study meteial\NEET PG QBANK\p2e\output"


def _read(path):
    if not os.path.exists(path): return None
    with open(path, encoding="utf-8") as f: return json.load(f)

def _load_pdf_data(stem_dir):
    for n in ("clean.json", "raw.json"):
        d = _read(os.path.join(stem_dir, n))
        if d: return d
    return None

def _now(): return datetime.now(timezone.utc).isoformat()


def _upsert_pdf(conn, stem, subject, exam_type, total_q):
    conn.execute("""
        INSERT INTO pdfs (stem,subject,exam_type,total_q,status,loaded_at)
        VALUES (?,?,?,?,'loaded',datetime('now'))
        ON CONFLICT(stem) DO UPDATE SET
          subject=excluded.subject, exam_type=excluded.exam_type,
          total_q=excluded.total_q, status='loaded', loaded_at=excluded.loaded_at
    """, (stem, subject or "", exam_type or "", total_q))
    conn.commit()


def load_stem(stem, output_dir, db_path, replace=False, dry_run=False):
    """Load one stem. Returns count inserted."""
    stem_dir = os.path.join(output_dir, stem)
    conn     = DB.init_db(db_path)
    inserted = 0

    # ── Try clean.json / raw.json first ──────────────────────────
    data = _load_pdf_data(stem_dir)
    if data:
        questions = data.get("questions", [])
        for q in questions:
            qid = q.get("question_id")
            if not qid: continue
            if not replace and conn.execute(
                "SELECT 1 FROM questions WHERE question_id=?", (qid,)
            ).fetchone():
                continue
            if dry_run: inserted += 1; continue
            q.setdefault("subject", data.get("subject",""))
            DB.insert_question(conn, q)
            conn.execute("INSERT OR IGNORE INTO fsrs_cards (question_id,due) VALUES (?,?)",
                         (qid, _now()))
            inserted += 1
        conn.commit()
        if not dry_run:
            sm = {}
            for q in questions:
                sm = q.get("session_meta") or {}
                if sm: break
            _upsert_pdf(conn, stem, data.get("subject",""),
                        sm.get("Exam","") or data.get("exam_type",""), len(questions))
        log.info(f"[{stem}] {inserted} from JSON")
        return inserted

    # ── Try splits.json (splitter.py output) ─────────────────────
    splits = _read(os.path.join(stem_dir, "splits.json"))
    if splits:
        img_dir = os.path.join(stem_dir, "images")
        sm_last = {}
        for entry in splits.values():
            if isinstance(entry, dict) and entry.get("session_meta"):
                sm_last = entry["session_meta"]
                break

        for page_key, entry in splits.items():
            if not isinstance(entry, dict): continue
            q_num  = entry.get("q_num", 0)
            key    = f"q{q_num:03d}"
            mcq_id = (entry.get("mcq_id") or "").strip()
            qid    = mcq_id if mcq_id else f"{stem}_{key}"
            if not replace and conn.execute(
                "SELECT 1 FROM questions WHERE question_id=?", (qid,)
            ).fetchone():
                continue
            if dry_run: inserted += 1; continue

            opts = entry.get("options") or {}
            sm   = entry.get("session_meta") or {}
            q_img  = f"{stem}/images/{key}_question.png"
            e_img  = f"{stem}/images/{key}_explanation.png"
            if not os.path.exists(os.path.join(img_dir, f"{key}_question.png")):    q_img = None
            if not os.path.exists(os.path.join(img_dir, f"{key}_explanation.png")): e_img = None

            q = {
                "question_id":       qid,
                "pdf_stem":          stem,
                "subject":           (sm.get("Subject") or "").lower(),
                "topic":             (entry.get("topic") or "").strip() or None,
                "pearl_id":          mcq_id or None,
                "mcq_id":            mcq_id or None,
                "page":              entry.get("page_idx", 0) + 1,
                "question":          None,
                "question_html":     None,
                "question_image":    q_img,
                "options":           opts,
                "option_a":          opts.get("A"), "option_b": opts.get("B"),
                "option_c":          opts.get("C"), "option_d": opts.get("D"),
                "answer":            (entry.get("answer") or "").upper() or None,
                "explanation":       None,
                "explanation_html":  None,
                "explanation_image": e_img,
                "images":            [],
                "topic_tags":        [],
                "schema_topics":     [],
                "flags":             [],
                "exam_type":         sm.get("Exam") or None,
            }
            DB.insert_question(conn, q)
            conn.execute("INSERT OR IGNORE INTO fsrs_cards (question_id,due) VALUES (?,?)",
                         (qid, _now()))
            inserted += 1

        conn.commit()
        if not dry_run:
            _upsert_pdf(conn, stem, (sm_last.get("Subject","") or "").lower(),
                        sm_last.get("Exam","") or "", inserted)
        log.info(f"[{stem}] {inserted} from splits.json")
        return inserted

    log.warning(f"[{stem}] No data files found")
    return 0


def main():
    ap = argparse.ArgumentParser(description="P2E Loader")
    ap.add_argument("--db",      default=DEFAULT_DB)
    ap.add_argument("--out",     default=DEFAULT_OUT)
    ap.add_argument("--stem",    default=None)
    ap.add_argument("--all",     action="store_true")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.stem and not args.all:
        ap.print_help(); sys.exit(1)
    # DB will be created automatically by init_db if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)

    stems = [args.stem] if args.stem else [
        d for d in os.listdir(args.out)
        if os.path.isdir(os.path.join(args.out, d)) and any(
            os.path.exists(os.path.join(args.out, d, f))
            for f in ("clean.json","raw.json","splits.json"))
    ]

    total = sum(
        load_stem(s, args.out, args.db, args.replace, args.dry_run)
        for s in sorted(stems)
    )
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done — {total} questions total")


if __name__ == "__main__":
    main()
