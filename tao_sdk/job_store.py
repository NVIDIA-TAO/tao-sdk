# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed job metadata store.

Replaces tao_session_state.json with a persistent, multi-process safe
database. Tracks job lifecycle, outputs, checkpoints, and parent-child
relationships for spec_params resolution (output_dir, resume_model,
parent_model, etc.) without requiring MongoDB.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tao_sdk.checkpoints import build_checkpoint_candidate, select_checkpoint_path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    network_arch TEXT NOT NULL,
    action TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    backend_type TEXT DEFAULT '',
    backend_job_id TEXT DEFAULT '',
    status TEXT DEFAULT 'Pending',
    parent_job_id TEXT REFERENCES jobs(job_id),
    image TEXT DEFAULT '',
    specs TEXT DEFAULT '{}',
    results_dir TEXT DEFAULT '',
    local_results_dir TEXT DEFAULT '',
    base_checkpoint TEXT DEFAULT '',
    plan_id TEXT DEFAULT '',
    step_id INTEGER DEFAULT 0,
    last_epoch INTEGER DEFAULT 0,
    instance_id TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    path TEXT NOT NULL,
    epoch INTEGER DEFAULT 0,
    format TEXT DEFAULT 'safetensors',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_job ON checkpoints(job_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """SQLite-backed job metadata store.

    Thread-safe and multi-process safe via WAL mode. All public methods
    are self-contained transactions.

    Args:
        db_path: Path to the SQLite database file. Created if it doesn't
            exist. Use ":memory:" for testing.
    """

    def __init__(self, db_path: str = "tao_session.db"):
        self._db_path = db_path
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that may be missing from older databases."""
        columns = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for col, ddl in [
            ("plan_id", "TEXT DEFAULT ''"),
            ("step_id", "INTEGER DEFAULT 0"),
            ("instance_id", "TEXT DEFAULT ''"),
        ]:
            if col not in columns:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")
                logger.info("Migrated jobs table: added column %s", col)
        # Create index after columns are guaranteed to exist
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_plan_step ON jobs(plan_id, step_id)"
        )

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        network_arch: str,
        action: str,
        workspace_id: str,
        backend_type: str = "",
        backend_job_id: str = "",
        status: str = "Pending",
        parent_job_id: str | None = None,
        image: str = "",
        specs: dict | None = None,
        results_dir: str = "",
        local_results_dir: str = "",
        base_checkpoint: str = "",
        plan_id: str = "",
        step_id: int = 0,
        created_at: str | None = None,
    ) -> None:
        """Insert a new job record."""
        now = created_at or _now()
        self._conn.execute(
            """INSERT OR REPLACE INTO jobs
               (job_id, network_arch, action, workspace_id, backend_type,
                backend_job_id, status, parent_job_id, image, specs,
                results_dir, local_results_dir, base_checkpoint,
                plan_id, step_id,
                last_epoch, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
            (
                job_id, network_arch, action, workspace_id, backend_type,
                backend_job_id, status, parent_job_id, image,
                json.dumps(specs or {}),
                results_dir, local_results_dir, base_checkpoint,
                plan_id, step_id,
                now, now,
            ),
        )

    def get_job(self, job_id: str) -> dict | None:
        """Fetch a job by ID. Returns dict or None."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_job_status(
        self, job_id: str, status: str, last_epoch: int | None = None
    ) -> None:
        """Update job status and optionally last_epoch."""
        if last_epoch is not None:
            self._conn.execute(
                "UPDATE jobs SET status = ?, last_epoch = ?, updated_at = ? WHERE job_id = ?",
                (status, last_epoch, _now(), job_id),
            )
        else:
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status, _now(), job_id),
            )

    def list_jobs(
        self, status: str | None = None, network_arch: str | None = None,
        action: str | None = None, limit: int | None = None,
    ) -> list[dict]:
        """List jobs with optional filters."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if network_arch:
            query += " AND network_arch = ?"
            params.append(network_arch)
        if action:
            query += " AND action = ?"
            params.append(action)
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_job_by_plan_step(self, plan_id: str, step_id: int) -> dict | None:
        """Find the job for a given plan step. Returns the most recent match."""
        row = self._conn.execute(
            """SELECT * FROM jobs
               WHERE plan_id = ? AND step_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (plan_id, step_id),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Checkpoint tracking
    # ------------------------------------------------------------------

    def add_checkpoint(
        self,
        job_id: str,
        path: str,
        epoch: int = 0,
        format: str = "safetensors",
    ) -> None:
        """Record a checkpoint produced by a job."""
        self._conn.execute(
            """INSERT INTO checkpoints (job_id, path, epoch, format, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, path, epoch, format, _now()),
        )

    def get_checkpoints(self, job_id: str) -> list[dict]:
        """Get all checkpoints for a job, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY epoch DESC",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_checkpoint(self, job_id: str) -> dict | None:
        """Get the most recent checkpoint for a job."""
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY epoch DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_checkpoint_path(
        self,
        job_id: str,
        *,
        model_name: str | None = None,
        epoch: int | None = None,
        step: int | None = None,
        action: str = "resume",
        prefer_directory: bool = False,
        allow_latest: bool = False,
    ) -> str | None:
        """Resolve a checkpoint path for a job/action at an epoch or step."""
        candidates = [
            build_checkpoint_candidate(row["path"])
            for row in self.get_checkpoints(job_id)
            if row.get("path")
        ]
        return select_checkpoint_path(
            candidates,
            model_name=model_name,
            epoch=epoch,
            step=step,
            action=action,
            prefer_directory=prefer_directory,
            allow_latest=allow_latest,
        )

    # ------------------------------------------------------------------
    # Job chain queries
    # ------------------------------------------------------------------

    def get_parent_job(self, job_id: str) -> dict | None:
        """Get the parent job of a given job."""
        row = self._conn.execute(
            """SELECT p.* FROM jobs p
               JOIN jobs c ON c.parent_job_id = p.job_id
               WHERE c.job_id = ?""",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_parent_model_path(
        self,
        job_id: str,
        *,
        epoch: int | None = None,
        step: int | None = None,
        action: str = "resume",
    ) -> str:
        """Resolve a checkpoint path from the parent job."""
        parent = self.get_parent_job(job_id)
        if not parent:
            return ""
        path = self.get_checkpoint_path(
            parent["job_id"],
            model_name=parent.get("network_arch"),
            epoch=epoch,
            step=step,
            action=action,
            allow_latest=epoch is None and step is None,
        )
        return path or ""

    def get_parent_results_dir(self, job_id: str) -> str:
        """Get the results_dir from the parent job."""
        row = self._conn.execute(
            """SELECT p.results_dir FROM jobs p
               JOIN jobs c ON c.parent_job_id = p.job_id
               WHERE c.job_id = ?""",
            (job_id,),
        ).fetchone()
        return row["results_dir"] if row else ""

    # ------------------------------------------------------------------
    # Spec params resolution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inference function registry
    # ------------------------------------------------------------------
    # Mirrors FTMS's CLI_CONFIG_TO_FUNCTIONS in infer_params.py.
    # Each function takes (store, job_id) and returns the resolved value
    # or None. New inference functions are added here — no hardcoding
    # in sdk.py or elsewhere.

    def _infer_output_dir(self, job_id: str) -> str | None:
        """Same as FTMS infer_output_dir — returns the results cloud URI."""
        job = self.get_job(job_id)
        return job["results_dir"] if job else None

    def _infer_resume_model(self, job_id: str) -> str | None:
        """Checkpoint path from the parent job."""
        return self.get_parent_model_path(job_id) or None

    def _infer_resume_model_bool(self, job_id: str) -> str | None:
        """'true' if the parent job has checkpoints."""
        path = self.get_parent_model_path(job_id)
        return "true" if path else "false"

    def _infer_parent_model_folder(self, job_id: str) -> str | None:
        """Results dir of the parent job."""
        return self.get_parent_results_dir(job_id) or None

    def _infer_ptm_if_no_resume_model(self, job_id: str) -> str | None:
        """Base checkpoint if there's no parent job to resume from."""
        job = self.get_job(job_id)
        if job and not job["parent_job_id"]:
            return job["base_checkpoint"] or None
        return None

    # Registry: inference function name → method
    # Add new functions here as they're needed.
    _INFERENCE_FUNCTIONS: dict[str, str] = {
        "output_dir": "_infer_output_dir",
        "automl_output_dir": "_infer_output_dir",
        "resume_model": "_infer_resume_model",
        "parent_model": "_infer_resume_model",
        "resume_model_bool": "_infer_resume_model_bool",
        "parent_model_folder": "_infer_parent_model_folder",
        "ptm_if_no_resume_model": "_infer_ptm_if_no_resume_model",
    }

    def resolve_spec_param(self, job_id: str, inference_fn: str) -> str | None:
        """Resolve a spec_params inference function using stored job context.

        Looks up the inference function in the registry and calls the
        corresponding method. Mirrors FTMS's infer_params dispatch.

        Args:
            job_id: The current job being configured.
            inference_fn: The inference function name from network config
                spec_params (e.g., "output_dir", "resume_model").

        Returns:
            Resolved value, or None if unsupported or data unavailable.
        """
        method_name = self._INFERENCE_FUNCTIONS.get(inference_fn)
        if method_name:
            return getattr(self, method_name)(job_id)

        logger.debug(
            "Unsupported inference function '%s' for job %s", inference_fn, job_id
        )
        return None

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def import_from_state_file(self, state_file: str) -> int:
        """Import jobs from a tao_session_state.json file.

        Returns the number of jobs imported.
        """
        path = Path(state_file)
        if not path.exists():
            return 0
        try:
            with open(path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse state file %s", state_file)
            return 0

        count = 0
        for entry in state.get("active_jobs", []):
            job_id = entry.get("job_id", "")
            if not job_id or self.get_job(job_id):
                continue
            self.create_job(
                job_id=job_id,
                network_arch=entry.get("network_arch", ""),
                action=entry.get("action", ""),
                workspace_id=entry.get("workspace_id", ""),
                backend_type=entry.get("backend_type", ""),
                backend_job_id=entry.get("backend_job_id", ""),
                status=entry.get("status", "Unknown"),
                created_at=entry.get("created_at", _now()),
            )
            count += 1
        logger.info("Imported %d jobs from %s", count, state_file)
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
