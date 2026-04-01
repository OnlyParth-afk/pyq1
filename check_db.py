"""Quick DB check — run from p2e folder: python check_db.py"""
import sqlite3, os

DB = r"D:\Study meteial\NEET PG QBANK\p2e\marrow.db"

if not os.path.exists(DB):
    print("DB does not exist at:", DB)
else:
    size = os.path.getsize(DB)
    print(f"DB exists — size: {size} bytes")
    conn = sqlite3.connect(DB)
    count = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    print(f"Questions in DB: {count}")
    if count > 0:
        rows = conn.execute("SELECT question_id, subject, topic, answer FROM questions LIMIT 5").fetchall()
        for r in rows:
            print(" ", r)
    conn.close()
