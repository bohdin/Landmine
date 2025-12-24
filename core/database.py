import os
from pathlib import Path
import sqlite3
import time
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite database stored inside data/ to keep everything local and easy to demo
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "landmine.db"

Base = declarative_base()


def get_engine():
    os.makedirs(DATA_DIR, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False, "timeout": 10},
        future=True,
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.close()

    return engine


def run_with_retry(fn, retries: int = 3, delay: float = 0.2):
    last_exc = None
    for _ in range(retries):
        try:
            return fn()
        except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
            last_exc = exc
            if "locked" not in str(exc).lower():
                raise
            time.sleep(delay)
    if last_exc:
        raise last_exc


engine = get_engine()
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,  # keep objects usable after commit to avoid detached errors in simple flows
)


def init_db():
    """Create all tables if they do not exist yet."""
    from core import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_schema()


def _ensure_schema():
    """Apply lightweight schema adjustments for demo SQLite without migrations."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(missions);")
        mission_columns = [row[1] for row in cursor.fetchall()]
        if "track_spacing_m" not in mission_columns:
            cursor.execute("ALTER TABLE missions ADD COLUMN track_spacing_m REAL DEFAULT 5.0;")
        if "waypoint_step_m" not in mission_columns:
            cursor.execute("ALTER TABLE missions ADD COLUMN waypoint_step_m REAL DEFAULT 2.0;")
        if "capture_every_m" not in mission_columns:
            cursor.execute("ALTER TABLE missions ADD COLUMN capture_every_m REAL DEFAULT 2.0;")
        if "seed" not in mission_columns:
            cursor.execute("ALTER TABLE missions ADD COLUMN seed INTEGER;")

        cursor.execute("PRAGMA table_info(detections);")
        columns = [row[1] for row in cursor.fetchall()]
        if "label" not in columns:
            cursor.execute("ALTER TABLE detections ADD COLUMN label TEXT NOT NULL DEFAULT 'review';")
        conn.commit()
    finally:
        conn.close()
