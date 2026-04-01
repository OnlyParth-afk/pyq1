"""
app.py — Main Flask Application
Registers all blueprints, serves templates + static files.
"""
from __future__ import annotations
import logging
import os
import sqlite3
from flask import Flask, g, jsonify, render_template, \
    request, send_from_directory, abort

import db as DB
import progress_db as PDB
from study_api     import study_bp
from analytics_api import analytics_bp
from review_api    import review_bp
from audit         import audit_routes

log = Flask(__name__)


# ══════════════════════════════════════════════════════════════════
# APP FACTORY
# ══════════════════════════════════════════════════════════════════

def create_app(
    marrow_db_path  : str,
    progress_db_path: str,
    images_dir      : str,
    output_dir      : str = "output",
) -> Flask:
    app = Flask(
        __name__,
        template_folder = "templates",
        static_folder   = "static",
    )
    app.config["MARROW_DB"]   = marrow_db_path
    app.config["PROGRESS_DB"] = progress_db_path
    app.config["IMAGES_DIR"]  = images_dir
    app.config["OUTPUT_DIR"]  = output_dir

    # Register blueprints
    app.secret_key = os.environ.get("P2E_SECRET", "p2e-dev-secret-2025")
    app.register_blueprint(study_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(review_bp)
    audit_routes(app)

    # ── DB connections per request ─────────────────────────────────
    @app.before_request
    def open_dbs():
        g.marrow_conn   = sqlite3.connect(
            app.config["MARROW_DB"], check_same_thread=False
        )
        g.marrow_conn.row_factory = sqlite3.Row
        g.progress_conn = PDB.init_progress_db(app.config["PROGRESS_DB"])

    @app.teardown_request
    def close_dbs(exc=None):
        if hasattr(g, "marrow_conn"):
            g.marrow_conn.close()
        if hasattr(g, "progress_conn"):
            g.progress_conn.close()

    # ── Static images (compressed) ─────────────────────────────────
    @app.route("/images/<path:filename>")
    def serve_image(filename):
        import io
        from PIL import Image
        images_dir = app.config["IMAGES_DIR"]
        full_path  = os.path.join(images_dir, filename)
        if not os.path.exists(full_path):
            abort(404)

        # Check for cached compressed version
        cache_dir  = os.path.join(images_dir, "_cache")
        cache_path = os.path.join(cache_dir, filename.replace("/", "_").replace(".png", ".jpg"))
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(cache_path):
            img = Image.open(full_path).convert("RGB")
            # Resize to max 1400px wide, keep aspect ratio
            max_w = 1400
            if img.width > max_w:
                ratio = max_w / img.width
                img   = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            img.save(cache_path, "JPEG", quality=82, optimize=True)

        from flask import send_file
        return send_file(cache_path, mimetype="image/jpeg",
                         max_age=86400)  # cache 1 day in browser

    # ── EXISTING question API ──────────────────────────────────────
    @app.route("/api/stats")
    def api_stats():
        return jsonify(DB.get_stats(g.marrow_conn))

    @app.route("/api/subjects")
    def api_subjects():
        rows = g.marrow_conn.execute("""
            SELECT subject, COUNT(*) as count
            FROM questions GROUP BY subject ORDER BY count DESC
        """).fetchall()
        return jsonify([dict(r) for r in rows])

    @app.route("/api/questions")
    def api_questions():
        subject   = request.args.get("subject", "")
        topic     = request.args.get("topic",   "")
        answer    = request.args.get("answer",  "")
        q         = request.args.get("q",       "")
        exam_type = request.args.get("exam_type", "")
        limit     = min(int(request.args.get("limit", 50)), 200)
        offset    = int(request.args.get("offset", 0))

        qs = DB.search_questions(
            g.marrow_conn, subject, topic, answer, q,
            limit, offset, exam_type=exam_type,
        )

        # Count with same filters
        where, params = [], []
        if subject:   where.append("subject = ?");   params.append(subject)
        if exam_type: where.append("exam_type = ?"); params.append(exam_type)
        if answer:    where.append("answer = ?");    params.append(answer)
        if topic:     where.append("topic LIKE ?");  params.append(f"%{topic}%")
        if q:         where.append("question LIKE ?"); params.append(f"%{q}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        total  = g.marrow_conn.execute(
            f"SELECT COUNT(*) FROM questions {clause}", params
        ).fetchone()[0]

        return jsonify({
            "total": total, "offset": offset,
            "limit": limit, "questions": qs,
        })

    @app.route("/api/questions/<question_id>")
    def api_question(question_id):
        q = DB.get_question(g.marrow_conn, question_id)
        if not q:
            abort(404)
        # Attach FSRS card data
        card = PDB.get_card(g.progress_conn, question_id)
        if card:
            from fsrs import FSRS, Card
            fsrs_obj  = FSRS()
            card_obj  = Card.from_dict(card)
            retention = fsrs_obj.get_retrievability(card_obj)
            q["fsrs"] = {**card, "retention": round(retention, 3)}
        return jsonify(q)

    @app.route("/api/pdfs")
    def api_pdfs():
        rows = g.marrow_conn.execute(
            "SELECT * FROM pdfs ORDER BY stem"
        ).fetchall()
        return jsonify([dict(r) for r in rows])

    @app.route("/api/errors")
    def api_errors():
        rows = g.marrow_conn.execute(
            "SELECT * FROM errors ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        return jsonify([dict(r) for r in rows])

    # ── HTML PAGES ─────────────────────────────────────────────────
    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/questions")
    def questions_page():
        return render_template("questions.html")

    @app.route("/q/<question_id>")
    def question_detail(question_id):
        return render_template("question_detail.html",
                               question_id=question_id)

    @app.route("/study")
    def study_page():
        return render_template("study.html")

    @app.route("/session")
    def session_page():
        return render_template("session.html")

    @app.route("/analytics")
    def analytics_page():
        return render_template("analytics.html")

    @app.route("/topics")
    def topics_page():
        return render_template("topics.html")

    @app.route("/most-repeated")
    def most_repeated_page():
        return render_template("most_repeated.html")

    @app.route("/review")
    def review_page():
        return render_template("review.html")

    @app.route("/review/<stem>")
    def review_pdf_page(stem):
        return render_template("review_pdf.html", stem=stem)

    @app.route("/review/question")
    def review_question_page():
        return render_template("review_question.html")

    # ── Loader API ─────────────────────────────────────────────
    @app.route("/api/loader/load/<stem>", methods=["POST"])
    def api_load_stem(stem):
        from audit import is_audit_mode
        if not is_audit_mode():
            return jsonify({"error": "audit_mode_required"}), 403
        try:
            import loader
            n = loader.load_stem(
                stem,
                output_dir = app.config.get("OUTPUT_DIR", "output"),
                db_path    = app.config["MARROW_DB"],
            )
            return jsonify({"ok": True, "loaded": n})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app