"""
progress_db.py — User Progress Database
════════════════════════════════════════
Stores: FSRS cards, reviews, sessions,
        topic stats, daily stats, error log.
Separate from marrow.db (questions).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


# ══════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════

def init_progress_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    log.info(f"[progress_db] opened: {db_path}")
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );

    -- FSRS card state per question
    CREATE TABLE IF NOT EXISTS fsrs_cards (
        question_id     TEXT PRIMARY KEY,
        due             TEXT NOT NULL,
        stability       REAL DEFAULT 0.0,
        difficulty      REAL DEFAULT 5.0,
        elapsed_days    INTEGER DEFAULT 0,
        scheduled_days  INTEGER DEFAULT 0,
        reps            INTEGER DEFAULT 0,
        lapses          INTEGER DEFAULT 0,
        state           INTEGER DEFAULT 0,
        last_review     TEXT
    );

    -- Every individual review event
    CREATE TABLE IF NOT EXISTS reviews (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id     TEXT NOT NULL,
        session_id      TEXT,
        mode            TEXT,
        rating          INTEGER,
        answer_given    TEXT,
        is_correct      INTEGER DEFAULT 0,
        confidence      INTEGER DEFAULT 3,
        time_taken_sec  INTEGER DEFAULT 0,
        guessed         INTEGER DEFAULT 0,
        created_at      TEXT NOT NULL
    );

    -- Test sessions
    CREATE TABLE IF NOT EXISTS sessions (
        id              TEXT PRIMARY KEY,
        mode            TEXT NOT NULL,
        subject_filter  TEXT,
        topic_filter    TEXT,
        total_questions INTEGER DEFAULT 0,
        correct         INTEGER DEFAULT 0,
        incorrect       INTEGER DEFAULT 0,
        unattempted     INTEGER DEFAULT 0,
        score           REAL    DEFAULT 0,
        max_score       REAL    DEFAULT 0,
        duration_sec    INTEGER DEFAULT 0,
        started_at      TEXT,
        ended_at        TEXT
    );

    -- Aggregate topic performance
    CREATE TABLE IF NOT EXISTS topic_stats (
        subject         TEXT NOT NULL,
        topic           TEXT NOT NULL,
        total_seen      INTEGER DEFAULT 0,
        total_correct   INTEGER DEFAULT 0,
        total_incorrect INTEGER DEFAULT 0,
        current_streak  INTEGER DEFAULT 0,
        best_streak     INTEGER DEFAULT 0,
        avg_confidence  REAL    DEFAULT 0,
        avg_time_sec    REAL    DEFAULT 0,
        last_seen       TEXT,
        fsrs_retention  REAL    DEFAULT 0,
        PRIMARY KEY (subject, topic)
    );

    -- Daily aggregates (streak, target tracking)
    CREATE TABLE IF NOT EXISTS daily_stats (
        date            TEXT PRIMARY KEY,
        questions_done  INTEGER DEFAULT 0,
        correct         INTEGER DEFAULT 0,
        score           REAL    DEFAULT 0,
        study_time_sec  INTEGER DEFAULT 0,
        streak_day      INTEGER DEFAULT 0
    );

    -- Error pattern log
    CREATE TABLE IF NOT EXISTS error_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        question_id     TEXT,
        error_type      TEXT,
        subject         TEXT,
        topic           TEXT,
        session_id      TEXT,
        created_at      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_reviews_qid      ON reviews(question_id);
    CREATE INDEX IF NOT EXISTS idx_reviews_session  ON reviews(session_id);
    CREATE INDEX IF NOT EXISTS idx_reviews_date     ON reviews(created_at);
    CREATE INDEX IF NOT EXISTS idx_fsrs_due         ON fsrs_cards(due);
    CREATE INDEX IF NOT EXISTS idx_fsrs_state       ON fsrs_cards(state);
    CREATE INDEX IF NOT EXISTS idx_topic_subject    ON topic_stats(subject);
    """)
    conn.commit()


# ══════════════════════════════════════════════════════════════════
# FSRS CARDS
# ══════════════════════════════════════════════════════════════════

def get_card(conn: sqlite3.Connection, question_id: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT * FROM fsrs_cards WHERE question_id=?", (question_id,)
    ).fetchone()
    return dict(row) if row else None


def upsert_card(conn: sqlite3.Connection,
                question_id: str, card_dict: dict) -> None:
    conn.execute("""
        INSERT INTO fsrs_cards
            (question_id, due, stability, difficulty,
             elapsed_days, scheduled_days, reps,
             lapses, state, last_review)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(question_id) DO UPDATE SET
            due            = excluded.due,
            stability      = excluded.stability,
            difficulty     = excluded.difficulty,
            elapsed_days   = excluded.elapsed_days,
            scheduled_days = excluded.scheduled_days,
            reps           = excluded.reps,
            lapses         = excluded.lapses,
            state          = excluded.state,
            last_review    = excluded.last_review
    """, (
        question_id,
        card_dict["due"],
        card_dict["stability"],
        card_dict["difficulty"],
        card_dict["elapsed_days"],
        card_dict["scheduled_days"],
        card_dict["reps"],
        card_dict["lapses"],
        card_dict["state"],
        card_dict["last_review"],
    ))
    conn.commit()


def get_due_questions(
    conn       : sqlite3.Connection,
    limit      : int  = 100,
    subject    : str  = None,
    topic      : str  = None,
    new_only   : bool = False,
    review_only: bool = False,
) -> List[str]:
    """Return question_ids due for review today."""
    now = datetime.now(timezone.utc).isoformat()
    sql = "SELECT question_id FROM fsrs_cards WHERE due <= ?"
    params: list = [now]
    if new_only:
        sql += " AND state = 0"
    if review_only:
        sql += " AND state = 2"
    sql += " ORDER BY due ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [r["question_id"] for r in rows]


def get_new_question_ids(
    conn : sqlite3.Connection,
    limit: int = 50,
) -> List[str]:
    """Questions never seen (not in fsrs_cards yet)."""
    rows = conn.execute("""
        SELECT question_id FROM fsrs_cards
        WHERE state = 0
        ORDER BY RANDOM()
        LIMIT ?
    """, (limit,)).fetchall()
    return [r["question_id"] for r in rows]


def init_new_cards(
    conn        : sqlite3.Connection,
    question_ids: List[str],
) -> None:
    """Insert new cards for questions not yet in fsrs_cards."""
    now = datetime.now(timezone.utc).isoformat()
    existing = {
        r["question_id"]
        for r in conn.execute("SELECT question_id FROM fsrs_cards").fetchall()
    }
    new_ids = [q for q in question_ids if q not in existing]
    conn.executemany("""
        INSERT OR IGNORE INTO fsrs_cards
            (question_id, due, stability, difficulty,
             elapsed_days, scheduled_days, reps, lapses, state)
        VALUES (?,?,0,5,0,0,0,0,0)
    """, [(qid, now) for qid in new_ids])
    conn.commit()
    if new_ids:
        log.info(f"[fsrs] initialised {len(new_ids)} new cards")


# ══════════════════════════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════════════════════════

def create_session(
    conn          : sqlite3.Connection,
    mode          : str,
    total_questions: int,
    subject_filter: str = None,
    topic_filter  : str = None,
) -> str:
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO sessions
            (id, mode, subject_filter, topic_filter,
             total_questions, max_score, started_at)
        VALUES (?,?,?,?,?,?,?)
    """, (
        sid, mode, subject_filter, topic_filter,
        total_questions, total_questions * 4, now,
    ))
    conn.commit()
    return sid


def end_session(
    conn    : sqlite3.Connection,
    session_id: str,
    correct : int,
    incorrect: int,
    unattempted: int,
    duration_sec: int,
) -> None:
    score = correct * 4 - incorrect * 1
    now   = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE sessions SET
            correct      = ?,
            incorrect    = ?,
            unattempted  = ?,
            score        = ?,
            duration_sec = ?,
            ended_at     = ?
        WHERE id = ?
    """, (correct, incorrect, unattempted, score, duration_sec, now, session_id))
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT * FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def get_recent_sessions(
    conn: sqlite3.Connection, limit: int = 20
) -> List[Dict]:
    rows = conn.execute("""
        SELECT * FROM sessions
        WHERE ended_at IS NOT NULL
        ORDER BY started_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# REVIEWS
# ══════════════════════════════════════════════════════════════════

def save_review(
    conn          : sqlite3.Connection,
    question_id   : str,
    session_id    : str,
    mode          : str,
    rating        : int,
    answer_given  : Optional[str],
    is_correct    : bool,
    confidence    : int,
    time_taken_sec: int,
    guessed       : bool,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO reviews
            (question_id, session_id, mode, rating,
             answer_given, is_correct, confidence,
             time_taken_sec, guessed, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        question_id, session_id, mode, rating,
        answer_given, int(is_correct), confidence,
        time_taken_sec, int(guessed), now,
    ))
    conn.commit()


def get_question_history(
    conn       : sqlite3.Connection,
    question_id: str,
) -> List[Dict]:
    rows = conn.execute("""
        SELECT * FROM reviews
        WHERE question_id=?
        ORDER BY created_at DESC LIMIT 20
    """, (question_id,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# TOPIC STATS
# ══════════════════════════════════════════════════════════════════

def update_topic_stats(
    conn      : sqlite3.Connection,
    subject   : str,
    topic     : str,
    is_correct: bool,
    confidence: int,
    time_sec  : int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    # Upsert
    conn.execute("""
        INSERT INTO topic_stats (subject, topic, last_seen)
        VALUES (?,?,?)
        ON CONFLICT(subject, topic) DO UPDATE SET
            last_seen = excluded.last_seen
    """, (subject, topic, now))

    # Update counts
    if is_correct:
        conn.execute("""
            UPDATE topic_stats SET
                total_seen      = total_seen + 1,
                total_correct   = total_correct + 1,
                current_streak  = current_streak + 1,
                best_streak     = MAX(best_streak, current_streak + 1),
                avg_confidence  = (avg_confidence * total_seen + ?)
                                  / (total_seen + 1),
                avg_time_sec    = (avg_time_sec * total_seen + ?)
                                  / (total_seen + 1)
            WHERE subject=? AND topic=?
        """, (confidence, time_sec, subject, topic))
    else:
        conn.execute("""
            UPDATE topic_stats SET
                total_seen      = total_seen + 1,
                total_incorrect = total_incorrect + 1,
                current_streak  = 0,
                avg_confidence  = (avg_confidence * total_seen + ?)
                                  / (total_seen + 1),
                avg_time_sec    = (avg_time_sec * total_seen + ?)
                                  / (total_seen + 1)
            WHERE subject=? AND topic=?
        """, (confidence, time_sec, subject, topic))
    conn.commit()


def get_weak_topics(
    conn   : sqlite3.Connection,
    limit  : int = 20,
    subject: str = None,
) -> List[Dict]:
    sql = """
        SELECT subject, topic, total_seen,
               total_correct, total_incorrect,
               CASE WHEN total_seen > 0
                    THEN ROUND(total_correct * 100.0 / total_seen, 1)
                    ELSE 0 END AS accuracy,
               current_streak, best_streak,
               avg_confidence, avg_time_sec, last_seen
        FROM topic_stats
        WHERE total_seen > 0
    """
    params = []
    if subject:
        sql += " AND subject=?"
        params.append(subject)
    sql += " ORDER BY accuracy ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_topic_heatmap(
    conn   : sqlite3.Connection,
    subject: str = None,
) -> List[Dict]:
    sql = """
        SELECT subject, topic, total_seen, total_correct,
               CASE WHEN total_seen > 0
                    THEN ROUND(total_correct * 100.0 / total_seen, 1)
                    ELSE 0 END AS accuracy,
               fsrs_retention
        FROM topic_stats
    """
    params = []
    if subject:
        sql += " WHERE subject=?"
        params.append(subject)
    sql += " ORDER BY subject, topic"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# DAILY STATS
# ══════════════════════════════════════════════════════════════════

def update_daily_stats(
    conn      : sqlite3.Connection,
    is_correct: bool,
    score_delta: float,
    time_sec  : int,
) -> None:
    today = date.today().isoformat()
    conn.execute("""
        INSERT INTO daily_stats (date, questions_done, correct,
                                 score, study_time_sec)
        VALUES (?,1,?,?,?)
        ON CONFLICT(date) DO UPDATE SET
            questions_done = questions_done + 1,
            correct        = correct + ?,
            score          = score + ?,
            study_time_sec = study_time_sec + ?
    """, (
        today, int(is_correct), score_delta, time_sec,
        int(is_correct), score_delta, time_sec,
    ))
    conn.commit()
    _update_streak(conn, today)


def _update_streak(conn: sqlite3.Connection, today: str) -> None:
    from datetime import timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    prev = conn.execute(
        "SELECT streak_day FROM daily_stats WHERE date=?", (yesterday,)
    ).fetchone()
    streak = (prev["streak_day"] + 1) if prev else 1
    conn.execute(
        "UPDATE daily_stats SET streak_day=? WHERE date=?",
        (streak, today)
    )
    conn.commit()


def get_daily_stats(
    conn: sqlite3.Connection, days: int = 30
) -> List[Dict]:
    rows = conn.execute("""
        SELECT * FROM daily_stats
        ORDER BY date DESC LIMIT ?
    """, (days,)).fetchall()
    return [dict(r) for r in rows]


def get_streak(conn: sqlite3.Connection) -> int:
    row = conn.execute("""
        SELECT streak_day FROM daily_stats
        ORDER BY date DESC LIMIT 1
    """).fetchone()
    return row["streak_day"] if row else 0


def get_today_count(conn: sqlite3.Connection) -> int:
    today = date.today().isoformat()
    row = conn.execute(
        "SELECT questions_done FROM daily_stats WHERE date=?", (today,)
    ).fetchone()
    return row["questions_done"] if row else 0


# ══════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════

def get_accuracy_trend(
    conn   : sqlite3.Connection,
    days   : int = 30,
    subject: str = None,
) -> List[Dict]:
    if subject:
        rows = conn.execute("""
            SELECT DATE(r.created_at) as day,
                   COUNT(*) as total,
                   SUM(r.is_correct) as correct,
                   ROUND(AVG(r.is_correct)*100, 1) as accuracy
            FROM reviews r
            JOIN (SELECT question_id, subject FROM
                  main.questions LIMIT 0) dummy ON 1=0
            WHERE DATE(r.created_at) >= DATE('now', ? || ' days')
            GROUP BY day ORDER BY day ASC
        """, (f"-{days}",)).fetchall()
    else:
        rows = conn.execute("""
            SELECT DATE(created_at) as day,
                   COUNT(*) as total,
                   SUM(is_correct) as correct,
                   ROUND(AVG(is_correct)*100, 1) as accuracy
            FROM reviews
            WHERE DATE(created_at) >= DATE('now', ? || ' days')
            GROUP BY day ORDER BY day ASC
        """, (f"-{days}",)).fetchall()
    return [dict(r) for r in rows]


def get_subject_accuracy(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("""
        SELECT subject,
               SUM(total_seen)     as total,
               SUM(total_correct)  as correct,
               ROUND(SUM(total_correct)*100.0/MAX(SUM(total_seen),1),1)
                   as accuracy
        FROM topic_stats
        GROUP BY subject
        ORDER BY accuracy ASC
    """).fetchall()
    return [dict(r) for r in rows]


def get_error_breakdown(
    conn: sqlite3.Connection
) -> List[Dict]:
    rows = conn.execute("""
        SELECT error_type,
               COUNT(*) as count,
               subject
        FROM error_log
        GROUP BY error_type, subject
        ORDER BY count DESC
    """).fetchall()
    return [dict(r) for r in rows]


def get_overall_stats(conn: sqlite3.Connection) -> Dict:
    total = conn.execute(
        "SELECT COUNT(*) FROM reviews"
    ).fetchone()[0]
    correct = conn.execute(
        "SELECT SUM(is_correct) FROM reviews"
    ).fetchone()[0] or 0
    sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL"
    ).fetchone()[0]
    due_today = conn.execute(
        "SELECT COUNT(*) FROM fsrs_cards WHERE due <= ?",
        (datetime.now(timezone.utc).isoformat(),)
    ).fetchone()[0]
    streak = get_streak(conn)
    today  = get_today_count(conn)
    return {
        "total_reviews"  : total,
        "total_correct"  : correct,
        "total_sessions" : sessions,
        "due_today"      : due_today,
        "streak"         : streak,
        "today_count"    : today,
        "overall_accuracy":
            round(correct * 100 / max(total, 1), 1),
    }