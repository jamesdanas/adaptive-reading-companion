"""
db.py
SQLite persistence layer for the ADHD-aware adaptive reading demo.
Keeps things simple and file-based (adhd_reading.db) so the whole
app can run with zero external services.
"""
import sqlite3
import json
from datetime import datetime

DB_PATH = "adhd_reading.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            grade TEXT NOT NULL,
            guardian_name TEXT,
            guardian_role TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL,
            screening_score INTEGER,
            avg_reaction_time REAL,
            reaction_variability REAL,
            memory_score REAL,
            attention_profile TEXT,
            confidence REAL,
            settings_json TEXT,
            created_at TEXT,
            FOREIGN KEY (child_id) REFERENCES children (id)
        )
    """)

    # Migration for existing databases created before memory_score existed.
    try:
        cur.execute("ALTER TABLE screenings ADD COLUMN memory_score REAL")
    except sqlite3.OperationalError:
        pass  # column already exists

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL,
            passage_title TEXT,
            quiz_score REAL,
            avg_time_per_chunk REAL,
            breaks_triggered INTEGER,
            created_at TEXT,
            FOREIGN KEY (child_id) REFERENCES children (id)
        )
    """)

    conn.commit()
    conn.close()


def add_child(name, age, grade, guardian_name, guardian_role):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO children (name, age, grade, guardian_name, guardian_role, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, age, grade, guardian_name, guardian_role, datetime.now().isoformat()),
    )
    conn.commit()
    child_id = cur.lastrowid
    conn.close()
    return child_id


def list_children():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM children ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_child(child_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_screening(child_id, screening_score, avg_rt, rt_var, memory_score, profile, confidence, settings):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO screenings
           (child_id, screening_score, avg_reaction_time, reaction_variability,
            memory_score, attention_profile, confidence, settings_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (child_id, screening_score, avg_rt, rt_var, memory_score, profile, confidence,
         json.dumps(settings), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_latest_screening(child_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM screenings WHERE child_id = ? ORDER BY created_at DESC LIMIT 1",
        (child_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["settings"] = json.loads(d["settings_json"])
    return d


def save_session(child_id, passage_title, quiz_score, avg_time_per_chunk, breaks_triggered):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO sessions
           (child_id, passage_title, quiz_score, avg_time_per_chunk, breaks_triggered, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (child_id, passage_title, quiz_score, avg_time_per_chunk, breaks_triggered,
         datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_sessions(child_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE child_id = ? ORDER BY created_at ASC", (child_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def seed_demo_data():
    """
    Ensures one fully-populated demo learner exists, so the Progress
    Monitoring / Dashboard / Recommendations pages have something to show
    even before the presenter registers a live learner. Safe to call on
    every startup -- it only inserts once, checked by guardian_role marker.
    """
    import ml_engine
    from datetime import timedelta

    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM children WHERE guardian_role = 'DEMO_SEED'"
    ).fetchone()
    conn.close()
    if existing:
        return

    demo_id = add_child("Udinah (Demo)", 9, "Developing Reader", "Demo Data", "DEMO_SEED")

    profile, confidence, settings = ml_engine.predict_profile(19, 610, 140, 0.55, 9)
    save_screening(demo_id, 19, 610, 140, 0.55, profile, confidence, settings)

    conn = get_conn()
    cur = conn.cursor()
    demo_sessions = [
        ("The Curious Tortoise", 33.3, 22.0, 3, 6),
        ("The Curious Tortoise", 66.7, 15.0, 1, 4),
        ("Amaka's Market Day", 100.0, 11.0, 0, 2),
        ("Amaka's Market Day", 66.7, 13.5, 1, 1),
    ]
    now = datetime.now()
    for i, (title, score, avg_time, breaks, days_ago) in enumerate(demo_sessions):
        cur.execute(
            """INSERT INTO sessions
               (child_id, passage_title, quiz_score, avg_time_per_chunk, breaks_triggered, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (demo_id, title, score, avg_time, breaks,
             (now - timedelta(days=days_ago)).isoformat()),
        )
    conn.commit()
    conn.close()
