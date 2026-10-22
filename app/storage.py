"""SQLite persistence for counting sessions."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import CROSSCHECK_MODEL_ID


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "sessions.sqlite3"
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, label TEXT NOT NULL, model TEXT NOT NULL,
                    confidence REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS refs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    ordinal INTEGER NOT NULL, filename TEXT NOT NULL, boxes TEXT NOT NULL,
                    width INTEGER NOT NULL, height INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    status TEXT NOT NULL, created_at TEXT NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id),
                    ordinal INTEGER NOT NULL, filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL, sha256 TEXT NOT NULL,
                    count INTEGER, detections TEXT, annotated_filename TEXT,
                    width INTEGER NOT NULL, height INTEGER NOT NULL, elapsed_ms REAL,
                    warning TEXT, processing TEXT, reference_boxes TEXT
                );
                CREATE INDEX IF NOT EXISTS refs_session ON refs(session_id);
                CREATE INDEX IF NOT EXISTS batches_session ON batches(session_id);
                CREATE INDEX IF NOT EXISTS images_batch ON images(batch_id);
            """)
            columns = {column["name"] for column in db.execute("PRAGMA table_info(images)")}
            if "warning" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN warning TEXT")
            if "processing" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN processing TEXT")
            if "reference_boxes" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN reference_boxes TEXT")
            if "search_areas" not in columns:
                db.execute("ALTER TABLE images ADD COLUMN search_areas TEXT")
            batch_columns = {column["name"] for column in db.execute("PRAGMA table_info(batches)")}
            if "count_mode" not in batch_columns:
                db.execute("ALTER TABLE batches ADD COLUMN count_mode TEXT NOT NULL DEFAULT 'standard'")
            for name, definition in {
                "verification": "TEXT", "review_detections": "TEXT", "review_annotated_filename": "TEXT",
                "review_revision": "INTEGER NOT NULL DEFAULT 0", "reviewed_at": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")
            db.execute("UPDATE batches SET status='failed', error=? WHERE status IN ('queued', 'running')",
                       ("Server restarted before this batch finished. Please retry the images.",))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_session(self, session_id, label, model, confidence, refs):
        with self.connection() as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                       (session_id, label, model, confidence, "active", now()))
            db.executemany("INSERT INTO refs VALUES (?,?,?,?,?,?,?)", [
                (r["id"], session_id, i, r["filename"], json.dumps(r["boxes"]),
                 r["width"], r["height"]) for i, r in enumerate(refs)
            ])

    def session_row(self, session_id):
        with self.connection() as db:
            return db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()

    def create_batch(self, session_id, batch_id, images, count_mode="standard"):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            session = db.execute("SELECT status FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session:
                return "missing"
            if session["status"] != "active":
                return "finished"
            busy = db.execute("SELECT 1 FROM batches WHERE session_id=? AND status IN ('queued','running')",
                              (session_id,)).fetchone()
            if busy:
                return "busy"
            hashes = [item["sha256"] for item in images]
            placeholders = ",".join("?" for _ in hashes)
            duplicate = db.execute(f"""SELECT 1 FROM images i JOIN batches b ON b.id=i.batch_id
                WHERE b.session_id=? AND b.status IN ('queued','running','complete')
                AND i.sha256 IN ({placeholders}) LIMIT 1""", (session_id, *hashes)).fetchone()
            if duplicate:
                return "duplicate"
            db.execute("""INSERT INTO batches (id, session_id, status, created_at, error, count_mode)
                VALUES (?,?,?,?,?,?)""", (batch_id, session_id, "queued", now(), None, count_mode))
            db.executemany("""INSERT INTO images (id, batch_id, ordinal, filename,
                original_filename, sha256, count, detections, annotated_filename,
                width, height, elapsed_ms, warning, reference_boxes, search_areas)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", [
                (item["id"], batch_id, i, item["filename"], item["original_filename"],
                 item["sha256"], None, None, None, item["width"], item["height"], None, None,
                 json.dumps(item["reference_boxes"]) if "reference_boxes" in item else None,
                 json.dumps(item["search_areas"]) if "search_areas" in item else None)
                for i, item in enumerate(images)
            ])
        return None

    def set_batch_status(self, batch_id, status, error=None):
        with self.connection() as db:
            db.execute("UPDATE batches SET status=?, error=? WHERE id=?", (status, error, batch_id))

    def complete_batch(self, batch_id, results):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for result in results:
                db.execute("""UPDATE images SET count=?, detections=?, annotated_filename=?,
                    width=?, height=?, elapsed_ms=?, warning=?, processing=?, verification=? WHERE id=? AND batch_id=?""",
                    (result["count"], json.dumps(result["detections"]),
                     result["annotated_filename"], result["width"], result["height"],
                     result["elapsed_ms"], result.get("warning"),
                     json.dumps(result["processing"]) if result.get("processing") is not None else None,
                     json.dumps(result["verification"]) if result.get("verification") is not None else None,
                     result["id"], batch_id))
            db.execute("UPDATE batches SET status='complete', error=NULL WHERE id=?", (batch_id,))

    def finish(self, session_id):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            session = db.execute("SELECT status, model FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session:
                return "missing"
            if session["status"] == "finished":
                return "finished"
            if db.execute("SELECT 1 FROM batches WHERE session_id=? AND status IN ('queued','running')",
                          (session_id,)).fetchone():
                return "busy"
            if session["model"] == CROSSCHECK_MODEL_ID and db.execute("""SELECT 1 FROM images i
                    JOIN batches b ON b.id=i.batch_id WHERE b.session_id=? AND b.status='complete'
                    AND i.reviewed_at IS NULL LIMIT 1""", (session_id,)).fetchone():
                return "unreviewed"
            db.execute("UPDATE sessions SET status='finished' WHERE id=?", (session_id,))
        return None

    def review_image_row(self, session_id, image_id):
        with self.connection() as db:
            return db.execute("""SELECT i.*, b.status AS batch_status FROM images i
                JOIN batches b ON b.id=i.batch_id WHERE b.session_id=? AND i.id=?""",
                (session_id, image_id)).fetchone()

    def save_review(self, session_id, image_id, revision, detections, annotated_filename, reviewed):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT i.review_revision, b.status AS batch_status,
                s.status AS session_status FROM images i JOIN batches b ON b.id=i.batch_id
                JOIN sessions s ON s.id=b.session_id WHERE s.id=? AND i.id=?""",
                (session_id, image_id)).fetchone()
            if row is None:
                return "missing"
            if row["session_status"] != "active":
                return "finished"
            if row["batch_status"] != "complete" or db.execute(
                    "SELECT 1 FROM batches WHERE session_id=? AND status IN ('queued','running')",
                    (session_id,)).fetchone():
                return "busy"
            if row["review_revision"] != revision:
                return "stale"
            db.execute("""UPDATE images SET review_detections=?, review_annotated_filename=?,
                review_revision=review_revision+1, reviewed_at=? WHERE id=?""",
                (json.dumps(detections), annotated_filename, now() if reviewed else None, image_id))
        return None

    def detail(self, session_id):
        with self.connection() as db:
            session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session:
                return None
            refs = db.execute("SELECT * FROM refs WHERE session_id=? ORDER BY ordinal", (session_id,)).fetchall()
            batches = db.execute("SELECT * FROM batches WHERE session_id=? ORDER BY rowid",
                                 (session_id,)).fetchall()
            batch_data = []
            total_count = image_count = reviewed_count = 0
            for batch in batches:
                result_data = []
                batch_count = 0
                if batch["status"] == "complete":
                    images = db.execute("SELECT * FROM images WHERE batch_id=? ORDER BY ordinal",
                                        (batch["id"],)).fetchall()
                    for item in images:
                        reviewed_detections = (json.loads(item["review_detections"])
                                               if item["review_detections"] is not None else None)
                        count = len(reviewed_detections) if reviewed_detections is not None else item["count"]
                        batch_count += count
                        reviewed_count += item["reviewed_at"] is not None
                        result_data.append({
                            "id": item["id"], "filename": item["filename"], "count": count,
                            "annotated_url": self.file_url(session_id, item["review_annotated_filename"]
                                                           or item["annotated_filename"]),
                            "original_url": self.file_url(session_id, item["original_filename"]),
                            "detections": (reviewed_detections if reviewed_detections is not None
                                           else json.loads(item["detections"])),
                            "model_count": item["count"], "model_detections": json.loads(item["detections"]),
                            "model_annotated_url": self.file_url(session_id, item["annotated_filename"]),
                            "reviewed": item["reviewed_at"] is not None, "reviewed_at": item["reviewed_at"],
                            "review_revision": item["review_revision"],
                            "verification": json.loads(item["verification"]) if item["verification"] else None,
                            "width": item["width"], "height": item["height"],
                            "elapsed_ms": item["elapsed_ms"], "warning": item["warning"],
                            "processing": json.loads(item["processing"]) if item["processing"] else None,
                            "reference_boxes": (json.loads(item["reference_boxes"])
                                                if item["reference_boxes"] else []),
                            "search_areas": (json.loads(item["search_areas"])
                                             if item["search_areas"] else []),
                        })
                    total_count += batch_count
                    image_count += len(images)
                batch_data.append({
                    "id": batch["id"], "status": batch["status"],
                    "count_mode": batch["count_mode"],
                    "created_at": batch["created_at"], "error": batch["error"],
                    "total_count": batch_count, "results": result_data,
                })
        return {
            "id": session["id"], "label": session["label"], "model": session["model"],
            "confidence": session["confidence"], "status": session["status"],
            "created_at": session["created_at"], "total_count": total_count,
            "image_count": image_count,
            "reviewed_image_count": reviewed_count, "unreviewed_image_count": image_count - reviewed_count,
            "requires_review": session["model"] == CROSSCHECK_MODEL_ID,
            "references": [{
                "filename": ref["filename"], "url": self.file_url(session_id, ref["filename"]),
                "boxes": json.loads(ref["boxes"]), "width": ref["width"], "height": ref["height"],
            } for ref in refs],
            "batches": batch_data,
        }

    @staticmethod
    def file_url(session_id, filename):
        return f"/api/sessions/{session_id}/files/{filename}"

    def owns_file(self, session_id, filename):
        with self.connection() as db:
            return bool(db.execute("SELECT 1 FROM refs WHERE session_id=? AND filename=?", (session_id, filename)).fetchone()
                or db.execute("""SELECT 1 FROM images i JOIN batches b ON b.id=i.batch_id
                    WHERE b.session_id=? AND (b.status IN ('queued','running','complete') AND i.original_filename=? OR
                    (b.status='complete' AND (i.annotated_filename=? OR i.review_annotated_filename=?)))""",
                    (session_id, filename, filename, filename)).fetchone())
