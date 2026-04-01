"""
review_api.py — Flask Blueprint: /api/review/*
All edit endpoints require audit mode ON.
All saves are atomic (write .tmp → rename).
All changes logged to manifest.json edit_log.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, g
from audit import require_audit
from auto_flag import auto_flag_question, flag_summary

log       = logging.getLogger(__name__)
review_bp = Blueprint("review", __name__)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _out_dir() -> str:
    from flask import current_app
    return current_app.config["OUTPUT_DIR"]


def _pdf_dir(stem: str) -> str:
    return os.path.join(_out_dir(), stem)


def _raw_path(stem: str)     -> str:
    return os.path.join(_pdf_dir(stem), "raw.json")

def _clean_path(stem: str)   -> str:
    return os.path.join(_pdf_dir(stem), "clean.json")

def _manifest_path(stem: str)-> str:
    return os.path.join(_pdf_dir(stem), "manifest.json")

def _index_path()            -> str:
    return os.path.join(_out_dir(), "index.json")


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pdf_data(stem: str) -> dict:
    """Load clean.json if exists, else raw.json."""
    cp = _clean_path(stem)
    rp = _raw_path(stem)
    if os.path.exists(cp):
        return _read_json(cp)
    if os.path.exists(rp):
        return _read_json(rp)
    return None


def _save_pdf_data(stem: str, data: dict) -> None:
    """Always save to clean.json (never touch raw.json)."""
    _write_json(_clean_path(stem), data)


def _load_manifest(stem: str) -> dict:
    return _read_json(_manifest_path(stem))


def _save_manifest(stem: str, manifest: dict) -> None:
    _write_json(_manifest_path(stem), manifest)


def _log_edit(stem: str, q_id: str, field: str,
              old_val: Any, new_val: Any, action: str) -> None:
    manifest = _load_manifest(stem)
    if "edit_log" not in manifest:
        manifest["edit_log"] = []
    manifest["edit_log"].append({
        "time"     : datetime.now(timezone.utc).isoformat(),
        "q_id"     : q_id,
        "field"    : field,
        "old_value": old_val,
        "new_value": new_val,
        "action"   : action,
    })
    manifest["status"]      = "edited"
    manifest["last_edited"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(stem, manifest)
    # Update index.json status
    _update_index_status(stem, "edited")


def _update_manifest_counts(stem: str, questions: list) -> None:
    manifest = _load_manifest(stem)
    manifest["total_q"]    = len(questions)
    manifest["missing_ans"]= sum(1 for q in questions if not q.get("answer"))
    manifest["bleed_count"]= sum(
        1 for q in questions
        if any(f.startswith("bleed") for f in q.get("flags", []))
    )
    manifest["flag_summary"] = flag_summary(questions)
    _save_manifest(stem, manifest)


def _update_index_status(stem: str, status: str) -> None:
    ip = _index_path()
    if not os.path.exists(ip):
        return
    index = _read_json(ip)
    for p in index.get("pdfs", []):
        if p["stem"] == stem:
            p["status"] = status
            break
    _write_json(ip, index)


# ═══════════════════════��══════════════════════════════════════════
# PDF LIST
# ══════════════════════════════════════════════════════════════════

@review_bp.route("/api/review/pdfs")
def review_pdf_list():
    ip = _index_path()
    if not os.path.exists(ip):
        return jsonify({"pdfs": []})
    index = _read_json(ip)
    pdfs  = index.get("pdfs", [])
    # Attach fresh manifest data
    for p in pdfs:
        mp = _manifest_path(p["stem"])
        if os.path.exists(mp):
            m = _read_json(mp)
            p["status"]      = m.get("status", "raw")
            p["missing_ans"] = m.get("missing_ans", 0)
            p["bleed_count"] = m.get("bleed_count", 0)
            p["flag_summary"]= m.get("flag_summary", {})
    return jsonify({"pdfs": pdfs})


@review_bp.route("/api/review/pdf/<stem>")
def review_pdf_detail(stem: str):
    data = _load_pdf_data(stem)
    if not data:
        return jsonify({"error": "not found"}), 404
    manifest  = _load_manifest(stem)
    # Filter params
    flag      = request.args.get("flag")
    questions = data.get("questions", [])
    if flag == "missing_answer":
        questions = [q for q in questions if not q.get("answer")]
    elif flag == "bleeding":
        questions = [q for q in questions
                     if any(f.startswith("bleed") for f in q.get("flags",[]))]
    elif flag == "flagged":
        questions = [q for q in questions if q.get("flags")]
    elif flag == "approved":
        questions = [q for q in questions if "approved" in q.get("flags",[])]
    return jsonify({
        "stem"     : stem,
        "manifest" : manifest,
        "questions": questions,
        "total"    : len(questions),
    })


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>")
def review_question_detail(stem: str, q_id: str):
    data = _load_pdf_data(stem)
    if not data:
        return jsonify({"error": "pdf not found"}), 404
    q = next(
        (q for q in data.get("questions", []) if q["question_id"] == q_id),
        None
    )
    if not q:
        return jsonify({"error": "question not found"}), 404
    return jsonify(q)


# ══════════════════════════════════════════════════════════════════
# EDIT ENDPOINTS — all require audit mode
# ══════════════════════════════════════════════════════════════════

@review_bp.route("/api/review/pdf/<stem>/question/<q_id>",
                 methods=["PATCH"])
@require_audit
def edit_question(stem: str, q_id: str):
    """
    Generic field editor. Body can contain any of:
      answer, question, question_html, explanation,
      explanation_html, subject, topic, options
    """
    data = _load_pdf_data(stem)
    if not data:
        return jsonify({"error": "pdf not found"}), 404

    questions = data.get("questions", [])
    q_index   = next(
        (i for i, q in enumerate(questions) if q["question_id"] == q_id),
        None
    )
    if q_index is None:
        return jsonify({"error": "question not found"}), 404

    body    = request.json
    q       = questions[q_index]
    changes = []

    EDITABLE = [
        "answer", "question", "question_html",
        "explanation", "explanation_html",
        "subject", "topic",
    ]

    for field in EDITABLE:
        if field in body:
            old = q.get(field)
            new = body[field]
            if old != new:
                q[field] = new
                changes.append((field, old, new, f"edit_{field}"))

    # Options edit (partial or full)
    if "options" in body:
        old_opts = dict(q.get("options", {}))
        new_opts = dict(q.get("options", {}))
        for letter, text in body["options"].items():
            new_opts[letter] = text
        if old_opts != new_opts:
            q["options"] = new_opts
            changes.append(("options", old_opts, new_opts, "edit_options"))

    if not changes:
        return jsonify({"ok": True, "message": "No changes"})

    # Re-run auto-flag
    q["flags"] = [
        f for f in q.get("flags", [])
        if not f.startswith("bleed") and f != "missing_answer"
        and f != "empty_stem" and f != "short_stem"
        and f != "missing_options" and f != "answer_not_in_options"
    ]
    q["flags"] = list(set(q["flags"]) | set(auto_flag_question(q)))

    questions[q_index]  = q
    data["questions"]   = questions
    _save_pdf_data(stem, data)
    _update_manifest_counts(stem, questions)

    for field, old, new, action in changes:
        _log_edit(stem, q_id, field, old, new, action)

    return jsonify({"ok": True, "saved_to": "clean.json", "changes": len(changes)})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/answer",
                 methods=["POST"])
@require_audit
def set_answer(stem: str, q_id: str):
    """Quick endpoint: set the correct answer letter."""
    body   = request.json
    answer = (body.get("answer") or "").upper().strip()
    if answer not in ("A", "B", "C", "D"):
        return jsonify({"error": "Invalid answer. Must be A/B/C/D"}), 400

    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error": "not found"}), 404

    old = questions[q_index].get("answer")
    questions[q_index]["answer"] = answer

    # Remove missing_answer flag
    questions[q_index]["flags"] = [
        f for f in questions[q_index].get("flags", [])
        if f != "missing_answer"
    ]

    data["questions"] = questions
    _save_pdf_data(stem, data)
    _update_manifest_counts(stem, questions)
    _log_edit(stem, q_id, "answer", old, answer, "set_answer")

    return jsonify({"ok": True, "answer": answer})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/remove-bleed",
                 methods=["POST"])
@require_audit
def remove_bleed(stem: str, q_id: str):
    """
    Remove a specific bleeding text token from a field.
    Body: { "field": "question", "token": "©MARROW" }
    """
    body    = request.json
    field   = body.get("field", "question")
    token   = body.get("token", "")
    pattern = body.get("pattern")   # optional regex

    if field not in ("question","explanation","question_html","explanation_html"):
        return jsonify({"error": "invalid field"}), 400

    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error": "not found"}), 404

    old_text = questions[q_index].get(field, "") or ""
    if pattern:
        new_text = re.sub(pattern, "", old_text, flags=re.IGNORECASE).strip()
    else:
        new_text = old_text.replace(token, "").strip()

    questions[q_index][field] = new_text
    # Re-flag
    questions[q_index]["flags"] = list(
        set(questions[q_index].get("flags",[])) |
        set(auto_flag_question(questions[q_index]))
    )

    data["questions"] = questions
    _save_pdf_data(stem, data)
    _update_manifest_counts(stem, questions)
    _log_edit(stem, q_id, field, old_text, new_text, "remove_bleed")

    return jsonify({"ok": True, "new_text": new_text})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/merge-options",
                 methods=["POST"])
@require_audit
def merge_options(stem: str, q_id: str):
    """
    Merge option B text into option A, delete B.
    Body: { "from": "C", "into": "B" }
    """
    body  = request.json
    frm   = body.get("from", "").upper()
    into  = body.get("into", "").upper()
    if frm not in "ABCD" or into not in "ABCD" or frm == into:
        return jsonify({"error": "invalid letters"}), 400

    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error": "not found"}), 404

    opts = questions[q_index].get("options", {})
    if frm not in opts or into not in opts:
        return jsonify({"error": "option not found"}), 400

    old_opts = dict(opts)
    opts[into] = (opts.get(into,"") + " " + opts.get(frm,"")).strip()
    del opts[frm]
    questions[q_index]["options"] = opts
    questions[q_index]["flags"]   = list(
        set(auto_flag_question(questions[q_index]))
    )

    data["questions"] = questions
    _save_pdf_data(stem, data)
    _log_edit(stem, q_id, "options", old_opts, opts, "merge_options")

    return jsonify({"ok": True, "options": opts})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/move-image",
                 methods=["POST"])
@require_audit
def move_image(stem: str, q_id: str):
    """
    Reorder images list.
    Body: { "filename": "q001_img01.png", "direction": "up" | "down" }
    """
    body      = request.json
    filename  = body.get("filename", "")
    direction = body.get("direction", "up")

    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error": "not found"}), 404

    imgs = list(questions[q_index].get("images", []))
    if filename not in imgs:
        return jsonify({"error": "image not found"}), 404

    idx = imgs.index(filename)
    if direction == "up" and idx > 0:
        imgs[idx], imgs[idx-1] = imgs[idx-1], imgs[idx]
    elif direction == "down" and idx < len(imgs)-1:
        imgs[idx], imgs[idx+1] = imgs[idx+1], imgs[idx]

    old = questions[q_index]["images"]
    questions[q_index]["images"] = imgs
    data["questions"] = questions
    _save_pdf_data(stem, data)
    _log_edit(stem, q_id, "images", old, imgs, "move_image")

    return jsonify({"ok": True, "images": imgs})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/remove-image",
                 methods=["POST"])
@require_audit
def remove_image(stem: str, q_id: str):
    body     = request.json
    filename = body.get("filename", "")

    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error":"not found"}), 404

    old  = list(questions[q_index].get("images", []))
    imgs = [i for i in old if i != filename]
    questions[q_index]["images"] = imgs
    data["questions"] = questions
    _save_pdf_data(stem, data)
    _log_edit(stem, q_id, "images", old, imgs, "remove_image")

    return jsonify({"ok": True, "images": imgs})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/flag",
                 methods=["POST"])
@require_audit
def flag_question(stem: str, q_id: str):
    body  = request.json
    flag  = body.get("flag", "bad_extraction")
    data  = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error":"not found"}), 404
    flags = list(set(questions[q_index].get("flags",[]) + [flag]))
    questions[q_index]["flags"] = flags
    data["questions"] = questions
    _save_pdf_data(stem, data)
    _log_edit(stem, q_id, "flags", None, flag, "add_flag")
    return jsonify({"ok": True, "flags": flags})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/unflag",
                 methods=["POST"])
@require_audit
def unflag_question(stem: str, q_id: str):
    body  = request.json
    flag  = body.get("flag", "")
    data  = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error":"not found"}), 404
    flags = [f for f in questions[q_index].get("flags",[]) if f != flag]
    questions[q_index]["flags"] = flags
    data["questions"] = questions
    _save_pdf_data(stem, data)
    _log_edit(stem, q_id, "flags", flag, None, "remove_flag")
    return jsonify({"ok": True, "flags": flags})


@review_bp.route("/api/review/pdf/<stem>/question/<q_id>/approve",
                 methods=["POST"])
@require_audit
def approve_question(stem: str, q_id: str):
    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is None:
        return jsonify({"error":"not found"}), 404
    flags = list(set(
        [f for f in questions[q_index].get("flags",[])
         if f != "approved"] + ["approved"]
    ))
    questions[q_index]["flags"] = flags
    data["questions"] = questions
    _save_pdf_data(stem, data)
    _log_edit(stem, q_id, "flags", None, "approved", "approve_question")
    return jsonify({"ok": True})


@review_bp.route("/api/review/pdf/<stem>/approve-all",
                 methods=["POST"])
@require_audit
def approve_all_clean(stem: str):
    """Approve all questions that have no error flags."""
    data      = _load_pdf_data(stem)
    questions = data["questions"]
    count     = 0
    for q in questions:
        bad = [f for f in q.get("flags",[])
               if not f.startswith("bleed") and f != "missing_answer"
               and f != "bad_extraction"]
        if not bad and "approved" not in q.get("flags",[]):
            q["flags"] = list(set(q.get("flags",[]) + ["approved"]))
            count += 1
    data["questions"] = questions
    _save_pdf_data(stem, data)
    _update_manifest_counts(stem, questions)
    return jsonify({"ok": True, "approved": count})


@review_bp.route("/api/review/pdf/<stem>/undo",
                 methods=["POST"])
@require_audit
def undo_last(stem: str):
    """Revert the last edit_log entry."""
    manifest = _load_manifest(stem)
    log_entries = manifest.get("edit_log", [])
    if not log_entries:
        return jsonify({"error": "Nothing to undo"}), 400

    last  = log_entries[-1]
    q_id  = last["q_id"]
    field = last["field"]
    old   = last["old_value"]

    data      = _load_pdf_data(stem)
    questions = data["questions"]
    q_index   = next(
        (i for i,q in enumerate(questions) if q["question_id"]==q_id), None
    )
    if q_index is not None and field not in ("flags",):
        questions[q_index][field] = old
        data["questions"] = questions
        _save_pdf_data(stem, data)

    manifest["edit_log"] = log_entries[:-1]
    _save_manifest(stem, manifest)

    return jsonify({"ok": True, "reverted": last})