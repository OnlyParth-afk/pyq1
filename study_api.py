"""
study_api.py — Flask Blueprint: /api/study  /api/fsrs
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, g
from fsrs import FSRS, Card, Rating, State
import progress_db as PDB

log = logging.getLogger(__name__)
study_bp = Blueprint("study", __name__)
_fsrs    = FSRS()


def _pconn():
    return g.progress_conn


# ══════════════════════════════════════════════════════════════════
# FSRS — due questions
# ══════════════════════════════════════════════════════════════════

@study_bp.route("/api/fsrs/due")
def fsrs_due():
    conn    = _pconn()
    limit   = int(request.args.get("limit", 50))
    subject = request.args.get("subject")
    ids     = PDB.get_due_questions(conn, limit=limit, subject=subject)
    return jsonify({"due": ids, "count": len(ids)})


@study_bp.route("/api/fsrs/review", methods=["POST"])
def fsrs_review():
    data        = request.json
    question_id = data["question_id"]
    rating_int  = int(data["rating"])          # 1-4
    session_id  = data.get("session_id", "")
    mode        = data.get("mode", "practice")
    answer      = data.get("answer_given")
    is_correct  = bool(data.get("is_correct", False))
    confidence  = int(data.get("confidence", 3))
    time_sec    = int(data.get("time_taken_sec", 30))
    guessed     = bool(data.get("guessed", False))

    conn = _pconn()

    # Load or create card
    card_dict = PDB.get_card(conn, question_id)
    if card_dict:
        card = Card.from_dict(card_dict)
    else:
        card = Card()

    # Run FSRS
    rating       = Rating(rating_int)
    card, review = _fsrs.review(card, rating)

    # Save card
    PDB.upsert_card(conn, question_id, card.to_dict())

    # Save review
    PDB.save_review(
        conn, question_id, session_id, mode,
        rating_int, answer, is_correct,
        confidence, time_sec, guessed,
    )

    # Daily stats
    score_delta = 4 if is_correct else -1
    PDB.update_daily_stats(conn, is_correct, score_delta, time_sec)

    return jsonify({
        "next_due"      : card.due.isoformat(),
        "stability"     : round(card.stability, 2),
        "difficulty"    : round(card.difficulty, 2),
        "scheduled_days": card.scheduled_days,
        "state"         : int(card.state),
    })


@study_bp.route("/api/fsrs/stats/<question_id>")
def fsrs_card_stats(question_id: str):
    conn = _pconn()
    card_dict = PDB.get_card(conn, question_id)
    if not card_dict:
        return jsonify({"error": "not found"}), 404
    card        = Card.from_dict(card_dict)
    retention   = _fsrs.get_retrievability(card)
    history     = PDB.get_question_history(conn, question_id)
    return jsonify({
        "card"       : card_dict,
        "retention"  : round(retention, 3),
        "history"    : history,
    })


# ══════════════════════════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════════════════════════

@study_bp.route("/api/study/session/start", methods=["POST"])
def session_start():
    data    = request.json
    mode    = data.get("mode", "practice")
    subject = data.get("subject")
    topic   = data.get("topic")
    count   = int(data.get("count", 20))
    conn    = _pconn()
    mconn   = g.marrow_conn

    # Build question list
    question_ids = _build_question_list(
        mconn, conn, data, count
    )

    if not question_ids:
        return jsonify({"error": "No questions available"}), 400

    # Init any new cards
    PDB.init_new_cards(conn, question_ids)

    sid = PDB.create_session(conn, mode, len(question_ids), subject, topic)

    return jsonify({
        "session_id"  : sid,
        "mode"        : mode,
        "question_ids": question_ids,
        "total"       : len(question_ids),
    })


@study_bp.route("/api/study/session/end", methods=["POST"])
def session_end():
    data         = request.json
    session_id   = data["session_id"]
    correct      = int(data.get("correct",      0))
    incorrect    = int(data.get("incorrect",    0))
    unattempted  = int(data.get("unattempted",  0))
    duration_sec = int(data.get("duration_sec", 0))
    conn         = _pconn()

    PDB.end_session(
        conn, session_id,
        correct, incorrect, unattempted, duration_sec
    )
    session = PDB.get_session(conn, session_id)
    return jsonify({"session": session})


@study_bp.route("/api/study/sessions")
def session_list():
    conn  = _pconn()
    limit = int(request.args.get("limit", 20))
    rows  = PDB.get_recent_sessions(conn, limit)
    return jsonify({"sessions": rows})


@study_bp.route("/api/study/session/<session_id>")
def session_detail(session_id: str):
    conn = _pconn()
    s    = PDB.get_session(conn, session_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s)


# ══════════════════════════════════════════════════════════════════
# TOPIC STATS
# ══════════════════════════════════════════════════════════════════

@study_bp.route("/api/study/topic-stats", methods=["POST"])
def save_topic_stats():
    data       = request.json
    subject    = data["subject"]
    topic      = data["topic"]
    is_correct = bool(data.get("is_correct", False))
    confidence = int(data.get("confidence", 3))
    time_sec   = int(data.get("time_sec", 30))
    PDB.update_topic_stats(
        _pconn(), subject, topic, is_correct, confidence, time_sec
    )
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _build_question_list(
    mconn, pconn, params: dict, count: int
) -> list:
    mode       = params.get("mode", "practice")
    subject    = params.get("subject")
    topic      = params.get("topic")
    fsrs_only  = params.get("fsrs_due", False)
    wrong_only = params.get("wrong_only", False)
    weak_topic = params.get("weak_topics", False)

    ids = []

    if fsrs_only:
        ids = PDB.get_due_questions(pconn, limit=count, subject=subject)
        if len(ids) < count:
            # Pad with new questions
            new_ids = _get_questions_from_marrow(
                mconn, subject, topic, count - len(ids)
            )
            ids += new_ids
        return ids[:count]

    if wrong_only:
        rows = pconn.execute("""
            SELECT DISTINCT question_id FROM reviews
            WHERE is_correct = 0
            ORDER BY created_at DESC LIMIT ?
        """, (count,)).fetchall()
        return [r["question_id"] for r in rows]

    if weak_topic:
        weak = PDB.get_weak_topics(pconn, limit=5, subject=subject)
        for wt in weak:
            rows = _get_questions_from_marrow(
                mconn, wt["subject"], wt["topic"],
                max(1, count // len(weak))
            )
            ids += rows
        return ids[:count]

    # Default: from marrow.db
    return _get_questions_from_marrow(mconn, subject, topic, count)


def _get_questions_from_marrow(
    conn   : object,
    subject: str,
    topic  : str,
    count  : int,
) -> list:
    sql    = "SELECT question_id FROM questions WHERE 1=1"
    params = []
    if subject:
        sql    += " AND subject=?"
        params.append(subject)
    if topic:
        sql    += " AND topic LIKE ?"
        params.append(f"%{topic}%")
    sql    += " ORDER BY RANDOM() LIMIT ?"
    params.append(count)
    rows   = conn.execute(sql, params).fetchall()
    return [r["question_id"] for r in rows]