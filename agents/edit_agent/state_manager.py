"""
agents/edit_agent/state_manager.py
────────────────────────────────────
SQLite-backed state versioning.

Every pipeline run and every edit creates a named snapshot (v001, v002 …).
Snapshots capture all JSON handoff files plus images/ and audio/.
The final video is large so its existence is recorded but the file is not copied.
"""

import os
import shutil
import sqlite3
import threading
from datetime import datetime

STATE_DIR  = os.path.join("data", "state_versions")
STATE_DB   = os.path.join(STATE_DIR, "versions.db")
ASSETS_DIR = os.path.join(STATE_DIR, "assets")

_TRACKED_FILES = [
    os.path.join("data", "outputs", "scene_manifest.json"),
    os.path.join("data", "outputs", "character_db.json"),
    os.path.join("data", "outputs", "phase2_audio_handoff.json"),
    os.path.join("data", "outputs", "phase3_video_handoff.json"),
    os.path.join("data", "outputs", "audio", "timing_manifest.json"),
    os.path.join("data", "outputs", "edit_settings.json"),
]

_TRACKED_DIRS = [
    (os.path.join("data", "outputs", "images"), "images"),
    (os.path.join("data", "outputs", "audio"),  "audio"),
]


class StateManager:
    def __init__(self):
        os.makedirs(STATE_DIR,  exist_ok=True)
        os.makedirs(ASSETS_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._db   = sqlite3.connect(STATE_DB, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                version     TEXT UNIQUE NOT NULL,
                timestamp   TEXT NOT NULL,
                description TEXT NOT NULL,
                has_video   INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._db.commit()

    def snapshot(self, description: str) -> str:
        with self._lock:
            vid       = self._next_id()
            version   = f"v{vid:03d}"
            asset_dir = os.path.join(ASSETS_DIR, version)
            os.makedirs(asset_dir, exist_ok=True)

            # Copy individual JSON files
            for src in _TRACKED_FILES:
                if os.path.exists(src):
                    key = src.replace(os.sep, "__").replace("/", "__")
                    shutil.copy2(src, os.path.join(asset_dir, key))

            # Copy asset directories (images + audio)
            for src_dir, label in _TRACKED_DIRS:
                dst_dir = os.path.join(asset_dir, label)
                if os.path.exists(src_dir) and not os.path.exists(dst_dir):
                    shutil.copytree(src_dir, dst_dir)

            from config import FINAL_VIDEO
            has_video = 1 if os.path.exists(FINAL_VIDEO) else 0

            self._db.execute(
                "INSERT OR IGNORE INTO versions (version, timestamp, description, has_video) "
                "VALUES (?,?,?,?)",
                (version, datetime.now().isoformat(), description, has_video),
            )
            self._db.commit()
            print(f"[StateManager] Snapshot {version}: {description}")
            return version

    def revert(self, version: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT version FROM versions WHERE version=?", (version,)
            ).fetchone()
            if not row:
                raise ValueError(f"Version {version!r} not found")

            asset_dir = os.path.join(ASSETS_DIR, version)
            if not os.path.exists(asset_dir):
                raise FileNotFoundError(f"Assets for {version} missing at {asset_dir}")

            # Restore individual files
            for src in _TRACKED_FILES:
                key    = src.replace(os.sep, "__").replace("/", "__")
                backed = os.path.join(asset_dir, key)
                if os.path.exists(backed):
                    os.makedirs(os.path.dirname(src) or ".", exist_ok=True)
                    shutil.copy2(backed, src)

            # Restore asset directories
            for src_dir, label in _TRACKED_DIRS:
                backed_dir = os.path.join(asset_dir, label)
                if os.path.exists(backed_dir):
                    if os.path.exists(src_dir):
                        shutil.rmtree(src_dir)
                    shutil.copytree(backed_dir, src_dir)

            print(f"[StateManager] Reverted to {version}")
            return True

    def history(self) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, version, timestamp, description, has_video "
                "FROM versions ORDER BY id DESC"
            ).fetchall()
        return [
            {
                "id":          r[0],
                "version":     r[1],
                "timestamp":   r[2],
                "description": r[3],
                "has_video":   bool(r[4]),
            }
            for r in rows
        ]

    def _next_id(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM versions").fetchone()
        return (row[0] or 0) + 1
