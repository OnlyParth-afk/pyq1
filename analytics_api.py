"""
analytics_api.py — Flask Blueprint: /api/analytics
"""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request, g
import progress_db as PDB
from fsrs import FSRS, Card
from datetime import datetime, timezone, timedelta

log          = logging.getLogger(__name__)
analytics_bp = Blueprint("analytics", __name__)
_fsrs        = FSRS()


def _pconn(): return g.progress_conn
def _mconn(): return g.marrow_conn


@analytics_bp.route("/api/analytics/overview")
def overview():
    conn  = _pconn()
    stats = PDB.get_overall_stats(conn)
    daily = PDB.get_daily_stats(conn, days=30)
    return jsonify({"stats": stats, "daily": daily})


@analytics_bp.route("/api/analytics/heatmap")
def heatmap():
    conn    = _pconn()
    subject = request.args.get("subject")
    data    = PDB.get_topic_heatmap(conn, subject)
    return jsonify({"heatmap": data})


@analytics_bp.route("/api/analytics/trends")
def trends():
    conn    = _pconn()
    days    = int(request.args.get("days", 30))
    subject = request.args.get("subject")
    data    = PDB.get_accuracy_trend(conn, days, subject)
    return jsonify({"trends": data})


@analytics_bp.route("/api/analytics/weak-topics")
def weak_topics():
    conn    = _pconn()
    limit   = int(request.args.get("limit", 20))
    subject = request.args.get("subject")
    data    = PDB.get_weak_topics(conn, limit, subject)
    return jsonify({"weak_topics": data})


@analytics_bp.route("/api/analytics/subject-accuracy")
def subject_accuracy():
    conn = _pconn()
    data = PDB.get_subject_accuracy(conn)
    return jsonify({"subjects": data})


@analytics_bp.route("/api/analytics/fsrs-curve")
def fsrs_curve():
    """Retention curve for a single question or overall burndown."""
    conn        = _pconn()
    question_id = request.args.get("question_id")

    if question_id:
        card_dict = PDB.get_card(conn, question_id)
        if not card_dict:
            return jsonify({"curve": []})
        card   = Card.from_dict(card_dict)
        now    = datetime.now(timezone.utc)
        points = []
        for d in range(0, 31):
            r = _fsrs.get_retrievability(
                card, now + timedelta(days=d)
            )
            points.append({"day": d, "retention": round(r, 3)})
        return jsonify({"curve": points})

    # Overall burndown
    rows  = conn.execute("SELECT * FROM fsrs_cards").fetchall()
    cards = [Card.from_dict(dict(r)) for r in rows]
    data  = _fsrs.burndown(cards, days=30)
    return jsonify({"burndown": data})


@analytics_bp.route("/api/analytics/error-patterns")
def error_patterns():
    conn = _pconn()
    data = PDB.get_error_breakdown(conn)
    return jsonify({"errors": data})


@analytics_bp.route("/api/analytics/concept-frequency")
def concept_frequency():
    """Most frequently tested topics across all questions."""
    conn  = _mconn()
    rows  = conn.execute("""
        SELECT topic, subject, COUNT(*) as frequency
        FROM questions
        WHERE topic != ''
        GROUP BY topic, subject
        ORDER BY frequency DESC
        LIMIT 50
    """).fetchall()
    return jsonify({"concepts": [dict(r) for r in rows]})

@analytics_bp.route("/api/analytics/most-repeated")
def most_repeated():
    """
    Most repeated topics per subject across all PYQ questions.
    Groups by subject → sorted list of topics with count.
    Query params:
      subject  — filter to one subject
      min_count — only topics appearing >= n times (default 2)
      limit     — max topics per subject (default 20)
    """
    conn      = _mconn()
    subject   = request.args.get("subject", "")
    min_count = int(request.args.get("min_count", 2))
    limit     = int(request.args.get("limit", 30))

    where  = ["topic IS NOT NULL", "topic != ''"]
    params = []
    if subject:
        where.append("subject = ?")
        params.append(subject)

    clause = "WHERE " + " AND ".join(where)

    rows = conn.execute(f"""
        SELECT subject, topic,
               COUNT(*) as frequency,
               GROUP_CONCAT(DISTINCT exam_type) as exams
        FROM questions
        {clause}
        GROUP BY subject, topic
        HAVING COUNT(*) >= ?
        ORDER BY subject ASC, frequency DESC
    """, params + [min_count]).fetchall()

    # Group by subject
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        subj = r["subject"] or "unknown"
        grouped[subj].append({
            "topic"    : r["topic"],
            "frequency": r["frequency"],
            "exams"    : [e for e in (r["exams"] or "").split(",") if e],
        })

    # Respect per-subject limit
    result = []
    for subj in sorted(grouped.keys()):
        topics = grouped[subj][:limit]
        result.append({
            "subject": subj,
            "topics" : topics,
            "total"  : len(grouped[subj]),
        })

    # Subject-level summary (for overview cards)
    summary = conn.execute(f"""
        SELECT subject, COUNT(DISTINCT topic) as unique_topics,
               COUNT(*) as total_questions,
               MAX(cnt) as max_repeat
        FROM (
            SELECT subject, topic, COUNT(*) as cnt
            FROM questions
            WHERE topic IS NOT NULL AND topic != ''
            GROUP BY subject, topic
        )
        GROUP BY subject
        ORDER BY total_questions DESC
    """).fetchall()

    return jsonify({
        "subjects" : result,
        "summary"  : [dict(r) for r in summary],
    })
