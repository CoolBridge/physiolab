"""Minimal registration + usage tracking (SQLite). Swap for Supabase/PostgreSQL later.
Patient files are NEVER stored here; only account and usage metadata."""
import sqlite3, os
from datetime import datetime, timezone

DB = os.environ.get("PHYSIOLAB_DB", "physiolab.db")


def _c():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY, name TEXT, institution TEXT, category TEXT, country TEXT, specialty TEXT, created TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS events(email TEXT, modality TEXT, status TEXT, ts TEXT)")
    return c


def register(name, email, institution, category, country, specialty=""):
    with _c() as c:
        c.execute("INSERT OR REPLACE INTO users VALUES(?,?,?,?,?,?,?)",
                  (email.strip().lower(), name, institution, category, country, specialty,
                   datetime.now(timezone.utc).isoformat(timespec="seconds")))


def log_event(email, modality, status):
    with _c() as c:
        c.execute("INSERT INTO events VALUES(?,?,?,?)", (email, modality, status,
                  datetime.now(timezone.utc).isoformat(timespec="seconds")))


def stats():
    with _c() as c:
        q = lambda s: c.execute(s).fetchall()
        return {
            "registered_users": q("SELECT COUNT(*) FROM users")[0][0],
            "institutions": q("SELECT COUNT(DISTINCT institution) FROM users")[0][0],
            "countries": q("SELECT COUNT(DISTINCT country) FROM users")[0][0],
            "successful_analyses": q("SELECT COUNT(*) FROM events WHERE status='success'")[0][0],
            "failed_uploads": q("SELECT COUNT(*) FROM events WHERE status='failed'")[0][0],
            "uploads_by_modality": dict(q("SELECT modality, COUNT(*) FROM events GROUP BY modality")),
            "by_category": dict(q("SELECT category, COUNT(*) FROM users GROUP BY category")),
        }