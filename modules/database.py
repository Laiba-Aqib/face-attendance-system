# modules/database.py
"""
DATABASE MODULE — SESSION-AWARE
================================
Every attendance entry belongs to a SESSION.
A session has: ID, subject name, teacher, date, start time, duration.

Key design:
  - Sessions table: one row per class/lecture
  - Attendance table: one row per student per session
  - When app launches, teacher picks an existing active session OR creates new one
  - "Active" = session whose (start_time + duration) has not passed yet
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import List, Optional


DB_PATH = os.path.join("attendance", "attendance.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db():
    """Create all tables if they don't exist. Safe to call multiple times."""
    os.makedirs("attendance", exist_ok=True)
    conn = _connect()
    c = conn.cursor()

    # ── Sessions table ───────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            subject      TEXT    NOT NULL,
            teacher      TEXT    NOT NULL,
            date         TEXT    NOT NULL,
            start_time   TEXT    NOT NULL,
            duration_min INTEGER NOT NULL,
            expires_at   TEXT    NOT NULL
        )
    """)

    # ── Attendance table ─────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            name       TEXT    NOT NULL,
            user_id    INTEGER NOT NULL,
            marked_at  TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id),
            UNIQUE(session_id, name)        -- one entry per student per session
        )
    """)

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════
#  SESSION FUNCTIONS
# ════════════════════════════════════════════════════════════════

def create_session(subject: str, teacher: str, duration_min: int) -> int:
    """
    Create a new session and return its session_id.
    expires_at = now + duration_min  →  the window during which this session
    is considered 'active' and editable.
    """
    now = datetime.now()
    expires_at = now + timedelta(minutes=duration_min)

    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions (subject, teacher, date, start_time, duration_min, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        subject,
        teacher,
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        duration_min,
        expires_at.strftime("%Y-%m-%d %H:%M:%S")
    ))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_active_sessions() -> List[dict]:
    """
    Return all sessions whose expires_at is still in the future.
    These are sessions the teacher can 'resume' to add late students.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM sessions
        WHERE expires_at > ?
        ORDER BY start_time DESC
    """, (now,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_session(session_id: int) -> Optional[dict]:
    """Return a single session by ID."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def is_session_active(session_id: int) -> bool:
    """True if the session's expiry time has not passed."""
    session = get_session(session_id)
    if not session:
        return False
    expires = datetime.strptime(session["expires_at"], "%Y-%m-%d %H:%M:%S")
    return datetime.now() < expires


def get_seconds_remaining(session_id: int) -> int:
    """Seconds until this session expires. 0 if already expired."""
    session = get_session(session_id)
    if not session:
        return 0
    expires = datetime.strptime(session["expires_at"], "%Y-%m-%d %H:%M:%S")
    remaining = (expires - datetime.now()).total_seconds()
    return max(0, int(remaining))


def get_all_sessions_today() -> List[dict]:
    """All sessions (active or expired) created today."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE date = ? ORDER BY start_time ASC", (today,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# ════════════════════════════════════════════════════════════════
#  ATTENDANCE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def mark_attendance(session_id: int, name: str, user_id: int) -> bool:
    """
    Mark a student present in a session.
    Returns True if newly marked, False if already marked (duplicate).
    Uses INSERT OR IGNORE so duplicates never raise an error.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO attendance (session_id, name, user_id, marked_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, name, user_id, now))
    newly_inserted = c.rowcount == 1   # 1 = new row, 0 = ignored duplicate
    conn.commit()
    conn.close()
    return newly_inserted


def get_attendance_for_session(session_id: int) -> List[dict]:
    """All attendance records for a specific session."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT name, user_id, marked_at
        FROM attendance
        WHERE session_id = ?
        ORDER BY marked_at ASC
    """, (session_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_marked_names_for_session(session_id: int) -> set:
    """Return set of names already marked for this session (for fast duplicate check)."""
    records = get_attendance_for_session(session_id)
    return {r["name"] for r in records}


def get_all_students() -> dict:
    """Load names.json → {id_str: name}"""
    if not os.path.exists("names.json"):
        return {}
    with open("names.json", "r") as f:
        content = f.read().strip()
    return json.loads(content) if content else {}


def get_absent_students(session_id: int) -> List[str]:
    """Return names of registered students NOT yet marked in this session."""
    all_students = set(get_all_students().values())
    present = get_marked_names_for_session(session_id)
    return sorted(all_students - present)

def delete_student(user_id_str: str, user_name: str) -> tuple:
    """
    Completely remove a student from the system.

    Steps:
      1. Remove from names.json
      2. Delete their dataset folder (all 30 training images)
      3. Past attendance records in SQLite are kept (historical record)

    Returns (True, "") on success, (False, error_message) on failure.
    """
    import shutil

    # ── Step 1: Remove from names.json ───────────────────────
    try:
        if not os.path.exists("names.json"):
            return False, "names.json not found."

        with open("names.json", "r") as f:
            content_j = f.read().strip()
        names = json.loads(content_j) if content_j else {}

        if user_id_str not in names:
            return False, f"Student ID {user_id_str} not found in names.json."

        del names[user_id_str]

        with open("names.json", "w") as f:
            json.dump(names, f, indent=2)

    except Exception as e:
        return False, f"Failed to update names.json: {e}"

    # ── Step 2: Delete dataset folder ────────────────────────
    folder = os.path.join("dataset", f"user.{user_id_str}.{user_name}")
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)   # removes folder and all images inside
        except Exception as e:
            # Non-fatal — student is already removed from names.json
            print(f"Warning: could not delete dataset folder: {e}")

    # ── Step 3: Past attendance is intentionally kept ─────────
    # We do NOT delete SQLite attendance rows — historical records
    # should be preserved even if a student leaves.

    return True, ""