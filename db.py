"""
db.py - Questions database (marrow.db)
Handles all question storage and retrieval.
"""
from __future__ import annotations

import sqlite3
import os
from typing import Optional

# ══════════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════════

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ══════════════════════════════════════════════════════════════════
# INIT — create all tables
# ══════════════════════════════════════════════════════════════════

def init_db(db_path: str) -> sqlite3.Connection:
    conn = get_conn(db_path)
    c    = conn.cursor()

    # ── Questions ─────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            question_id      TEXT PRIMARY KEY,
            pdf_stem         TEXT,
            subject          TEXT,
            topic            TEXT,
            pearl_id         TEXT,
            page             INTEGER,
            question         TEXT,
            question_html    TEXT,
            question_image   TEXT,
            option_a         TEXT,
            option_b         TEXT,
            option_c         TEXT,
            option_d         TEXT,
            answer           TEXT,
            explanation      TEXT,
            explanation_html  TEXT,
            explanation_image TEXT,
            images           TEXT,
            topic_tags       TEXT,
            schema_topics    TEXT,
            flags            TEXT,
            exam_type        TEXT,
            mcq_id           TEXT,
            created_at       TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Migration: add new columns to existing DBs ─────────────
    _migrate_columns(conn)

    # ── FSRS cards ────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS fsrs_cards (
            question_id     TEXT PRIMARY KEY,
            state           INTEGER DEFAULT 0,
            due             TEXT,
            stability       REAL    DEFAULT 0.0,
            difficulty      REAL    DEFAULT 0.0,
            elapsed_days    INTEGER DEFAULT 0,
            scheduled_days  INTEGER DEFAULT 0,
            reps            INTEGER DEFAULT 0,
            lapses          INTEGER DEFAULT 0,
            last_review     TEXT,
            FOREIGN KEY (question_id) REFERENCES questions(question_id)
        )
    """)

    # ── Review log ────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS review_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id     TEXT,
            session_id      TEXT,
            mode            TEXT,
            rating          INTEGER,
            answer_given    TEXT,
            is_correct      INTEGER,
            confidence      INTEGER,
            time_taken_sec  INTEGER,
            guessed         INTEGER DEFAULT 0,
            state           INTEGER,
            due             TEXT,
            stability       REAL,
            difficulty      REAL,
            elapsed_days    INTEGER,
            scheduled_days  INTEGER,
            reviewed_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Sessions ──────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id      TEXT PRIMARY KEY,
            mode            TEXT,
            subject         TEXT,
            total_questions INTEGER DEFAULT 0,
            correct         INTEGER DEFAULT 0,
            incorrect       INTEGER DEFAULT 0,
            unattempted     INTEGER DEFAULT 0,
            score           REAL    DEFAULT 0.0,
            duration_sec    INTEGER DEFAULT 0,
            started_at      TEXT DEFAULT (datetime('now')),
            ended_at        TEXT
        )
    """)

    # ── PDFs registry ─────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            stem         TEXT PRIMARY KEY,
            subject      TEXT,
            exam_type    TEXT,
            total_q      INTEGER DEFAULT 0,
            status       TEXT    DEFAULT 'raw',
            extracted_at TEXT,
            loaded_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_q_subject ON questions(subject)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_q_pdf    ON questions(pdf_stem)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_q_answer ON questions(answer)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_q_topic  ON questions(topic)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fsrs_due ON fsrs_cards(due)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_log_qid  ON review_log(question_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_log_sess ON review_log(session_id)")

    conn.commit()
    return conn


def _migrate_columns(conn):
    """Idempotently add any missing columns to existing databases."""
    c = conn.cursor()
    existing = {row[1] for row in c.execute("PRAGMA table_info(questions)")}
    additions = [
        ("question_image",    "TEXT"),
        ("explanation_image", "TEXT"),
        ("exam_type",         "TEXT"),
        ("mcq_id",            "TEXT"),
    ]
    for col, typ in additions:
        if col not in existing:
            c.execute(f"ALTER TABLE questions ADD COLUMN {col} {typ}")
    # pdfs table
    existing_pdfs = {row[1] for row in c.execute("PRAGMA table_info(pdfs)")}
    if "exam_type" not in existing_pdfs:
        try:
            c.execute("ALTER TABLE pdfs ADD COLUMN exam_type TEXT")
        except Exception:
            pass
    conn.commit()


# ══════════════════════════════════════════════════════════════════
# QUESTIONS — insert / update / query
# ══════════════════════════════════════════════════════════════════

def insert_question(conn: sqlite3.Connection, q: dict) -> None:
    import json
    conn.execute("""
        INSERT OR REPLACE INTO questions (
            question_id, pdf_stem, subject, topic, pearl_id, page,
            question, question_html, question_image,
            option_a, option_b, option_c, option_d,
            answer, explanation, explanation_html, explanation_image,
            images, topic_tags, schema_topics, flags,
            exam_type, mcq_id
        ) VALUES (
            :question_id, :pdf_stem, :subject, :topic, :pearl_id, :page,
            :question, :question_html, :question_image,
            :option_a, :option_b, :option_c, :option_d,
            :answer, :explanation, :explanation_html, :explanation_image,
            :images, :topic_tags, :schema_topics, :flags,
            :exam_type, :mcq_id
        )
    """, {
        "question_id"     : q.get("question_id", ""),
        "pdf_stem"        : q.get("pdf_stem", ""),
        "subject"         : q.get("subject", ""),
        "topic"           : q.get("topic", ""),
        "pearl_id"        : q.get("pearl_id", ""),
        "page"            : q.get("page", 0),
        "question"        : q.get("question", ""),
        "question_html"   : q.get("question_html", ""),
        "option_a"        : (q.get("options") or {}).get("A", ""),
        "option_b"        : (q.get("options") or {}).get("B", ""),
        "option_c"        : (q.get("options") or {}).get("C", ""),
        "option_d"        : (q.get("options") or {}).get("D", ""),
        "answer"          : q.get("answer", ""),
        "explanation"      : q.get("explanation", ""),
        "explanation_html" : q.get("explanation_html", ""),
        "explanation_image": q.get("explanation_image", ""),
        "question_image"   : q.get("question_image", ""),
        "images"           : json.dumps(q.get("images", [])),
        "topic_tags"      : json.dumps(q.get("topic_tags", [])),
        "schema_topics"   : json.dumps(q.get("schema_topics", [])),
        "flags"           : json.dumps(q.get("flags", [])),
        "exam_type"        : q.get("exam_type", "") or q.get("session_meta", {}).get("Exam", ""),
        "mcq_id"           : q.get("mcq_id", ""),
    })


def get_question(
    conn: sqlite3.Connection, question_id: str
) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM questions WHERE question_id = ?",
        (question_id,)
    ).fetchone()
    return _row_to_q(row) if row else None


def get_questions(
    conn       : sqlite3.Connection,
    subject    : str  = "",
    answer     : str  = "",
    search     : str  = "",
    topic      : str  = "",
    limit      : int  = 50,
    offset     : int  = 0,
    **kwargs,
) -> tuple[list[dict], int]:
    where  = []
    params = []

    if subject:
        where.append("subject = ?")
        params.append(subject)
    if answer:
        where.append("answer = ?")
        params.append(answer)
    if topic:
        where.append("topic LIKE ?")
        params.append(f"%{topic}%")
    if search:
        where.append("question LIKE ?")
        params.append(f"%{search}%")
    exam_type = kwargs.get("exam_type", "")
    if exam_type:
        where.append("exam_type = ?")
        params.append(exam_type)

    clause = ("WHERE " + " AND ".join(where)) if where else ""

    total = conn.execute(
        f"SELECT COUNT(*) FROM questions {clause}", params
    ).fetchone()[0]

    rows = conn.execute(
        f"SELECT * FROM questions {clause} "
        f"ORDER BY question_id LIMIT ? OFFSET ?",
        params + [limit, offset]
    ).fetchall()

    return [_row_to_q(r) for r in rows], total


# Alias used by app.py
def search_questions(
    conn      : sqlite3.Connection,
    subject   : str = "",
    topic     : str = "",
    answer    : str = "",
    search    : str = "",
    limit     : int = 50,
    offset    : int = 0,
    exam_type : str = "",
    **kwargs,
) -> list[dict]:
    """Alias for get_questions that returns only the list (app.py compat)."""
    rows, _ = get_questions(conn, subject, answer, search, topic, limit, offset, **kwargs)
    return rows


def get_subjects(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT subject, COUNT(*) as count
        FROM questions
        WHERE subject != ''
        GROUP BY subject
        ORDER BY count DESC
    """).fetchall()
    return [{"subject": r["subject"], "count": r["count"]} for r in rows]


def get_questions_for_session(
    conn      : sqlite3.Connection,
    subject   : str  = "",
    count     : int  = 20,
    mode      : str  = "practice",
    fsrs_due  : bool = False,
    wrong_only: bool = False,
    exam_type : str  = "",
) -> list[dict]:
    import json
    from datetime import datetime, timezone

    where  = ["1=1"]
    params = []

    if subject:
        where.append("q.subject = ?")
        params.append(subject)

    if exam_type:
        where.append("q.exam_type = ?")
        params.append(exam_type)

    if wrong_only:
        where.append("""
            q.question_id IN (
                SELECT question_id FROM review_log
                WHERE is_correct = 0
            )
        """)

    if fsrs_due:
        now = datetime.now(timezone.utc).isoformat()
        where.append("""
            q.question_id IN (
                SELECT question_id FROM fsrs_cards
                WHERE due <= ?
            )
        """)
        params.append(now)

    clause = "WHERE " + " AND ".join(where)

    rows = conn.execute(f"""
        SELECT q.* FROM questions q
        {clause}
        ORDER BY RANDOM()
        LIMIT ?
    """, params + [count]).fetchall()

    return [_row_to_q(r) for r in rows]


def _row_to_q(row: sqlite3.Row) -> dict:
    import json
    if row is None:
        return {}
    d = dict(row)
    d["options"] = {
        "A": d.pop("option_a", "") or "",
        "B": d.pop("option_b", "") or "",
        "C": d.pop("option_c", "") or "",
        "D": d.pop("option_d", "") or "",
    }
    for field in ("images", "topic_tags", "schema_topics", "flags"):
        val = d.get(field)
        if isinstance(val, str):
            try:
                d[field] = json.loads(val)
            except Exception:
                d[field] = []
        elif val is None:
            d[field] = []
    return d


# ══════════════════════════════════════════════════════════════════
# FSRS CARDS
# ══════════════════════════════════════════════════════════════════

def get_fsrs_card(
    conn: sqlite3.Connection, question_id: str
) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM fsrs_cards WHERE question_id = ?",
        (question_id,)
    ).fetchone()
    return dict(row) if row else None


def upsert_fsrs_card(conn: sqlite3.Connection, card_dict: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO fsrs_cards (
            question_id, state, due, stability, difficulty,
            elapsed_days, scheduled_days, reps, lapses, last_review
        ) VALUES (
            :question_id, :state, :due, :stability, :difficulty,
            :elapsed_days, :scheduled_days, :reps, :lapses, :last_review
        )
    """, card_dict)


def get_due_cards(
    conn : sqlite3.Connection,
    now  : str,
    limit: int = 500,
) -> list[dict]:
    rows = conn.execute("""
        SELECT f.*, q.subject, q.topic
        FROM fsrs_cards f
        JOIN questions q ON f.question_id = q.question_id
        WHERE f.due <= ?
        ORDER BY f.due ASC
        LIMIT ?
    """, (now, limit)).fetchall()
    return [dict(r) for r in rows]


def init_fsrs_cards(conn: sqlite3.Connection) -> int:
    """Create FSRS card rows for any questions that don't have one yet."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO fsrs_cards (question_id, due)
        SELECT question_id, ? FROM questions
    """, (now,))
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM fsrs_cards"
    ).fetchone()[0]
    return count


# ══════════════════════════════════════════════════════════════════
# REVIEW LOG
# ══════════════════════════════════════════════════════════════════

def insert_review_log(conn: sqlite3.Connection, log: dict) -> None:
    conn.execute("""
        INSERT INTO review_log (
            question_id, session_id, mode, rating,
            answer_given, is_correct, confidence,
            time_taken_sec, guessed, state, due,
            stability, difficulty, elapsed_days, scheduled_days
        ) VALUES (
            :question_id, :session_id, :mode, :rating,
            :answer_given, :is_correct, :confidence,
            :time_taken_sec, :guessed, :state, :due,
            :stability, :difficulty, :elapsed_days, :scheduled_days
        )
    """, log)


# ══════════════════════════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════════════════════════

def create_session(conn: sqlite3.Connection, sess: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO sessions (
            session_id, mode, subject,
            total_questions, started_at
        ) VALUES (
            :session_id, :mode, :subject,
            :total_questions, :started_at
        )
    """, sess)


def end_session(conn: sqlite3.Connection, sess: dict) -> None:
    conn.execute("""
        UPDATE sessions SET
            correct         = :correct,
            incorrect       = :incorrect,
            unattempted     = :unattempted,
            score           = :score,
            duration_sec    = :duration_sec,
            ended_at        = :ended_at
        WHERE session_id = :session_id
    """, sess)


def get_sessions(
    conn : sqlite3.Connection,
    limit: int = 20,
) -> list[dict]:
    rows = conn.execute("""
        SELECT * FROM sessions
        WHERE ended_at IS NOT NULL
        ORDER BY started_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════

def get_stats(conn: sqlite3.Connection) -> dict:
    from datetime import datetime, timezone, timedelta

    total_q = conn.execute(
        "SELECT COUNT(*) FROM questions"
    ).fetchone()[0]

    total_reviews = conn.execute(
        "SELECT COUNT(*) FROM review_log"
    ).fetchone()[0]

    correct_reviews = conn.execute(
        "SELECT COUNT(*) FROM review_log WHERE is_correct = 1"
    ).fetchone()[0]

    overall_accuracy = (
        round(correct_reviews / total_reviews * 100)
        if total_reviews > 0 else 0
    )

    today = datetime.now(timezone.utc).date().isoformat()
    today_count = conn.execute(
        "SELECT COUNT(*) FROM review_log WHERE reviewed_at LIKE ?",
        (today + "%",)
    ).fetchone()[0]

    total_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL"
    ).fetchone()[0]

    now = datetime.now(timezone.utc).isoformat()
    due_today = conn.execute(
        "SELECT COUNT(*) FROM fsrs_cards WHERE due <= ?",
        (now,)
    ).fetchone()[0]

    # Streak — one GROUP BY query instead of 365 separate queries
    active_days = {
        row[0] for row in conn.execute("""
            SELECT DATE(reviewed_at)
            FROM review_log
            WHERE reviewed_at >= DATE('now','-365 days')
            GROUP BY DATE(reviewed_at)
        """).fetchall()
    }
    streak    = 0
    check_day = datetime.now(timezone.utc).date()
    for _ in range(365):
        if check_day.isoformat() in active_days:
            streak    += 1
            check_day -= timedelta(days=1)
        else:
            break

    return {
        "total_questions"  : total_q,
        "total_reviews"    : total_reviews,
        "overall_accuracy" : overall_accuracy,
        "today_count"      : today_count,
        "total_sessions"   : total_sessions,
        "due_today"        : due_today,
        "streak"           : streak,
    }