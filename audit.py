"""
audit.py — Audit Mode Control
Session-based: stored in Flask session.
No password needed (single user, offline).
"""
from __future__ import annotations
from flask import session, jsonify
from functools import wraps


AUDIT_SESSION_KEY = "audit_mode"


def is_audit_mode() -> bool:
    return bool(session.get(AUDIT_SESSION_KEY, False))


def require_audit(f):
    """Decorator: endpoint returns 403 if audit mode is off."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_audit_mode():
            return jsonify({
                "error"  : "audit_mode_required",
                "message": "Enable Audit Mode to make changes.",
            }), 403
        return f(*args, **kwargs)
    return decorated


def audit_routes(app):
    """Register audit toggle endpoints on the app."""

    @app.route("/api/audit/status")
    def audit_status():
        return jsonify({
            "audit_mode": is_audit_mode(),
        })

    @app.route("/api/audit/enable", methods=["POST"])
    def audit_enable():
        session[AUDIT_SESSION_KEY]    = True
        session.permanent             = False
        return jsonify({"audit_mode": True, "message": "Audit mode enabled."})

    @app.route("/api/audit/disable", methods=["POST"])
    def audit_disable():
        session[AUDIT_SESSION_KEY] = False
        return jsonify({"audit_mode": False, "message": "Audit mode disabled."})