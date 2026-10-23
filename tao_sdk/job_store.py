# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed job metadata store.

Replaces tao_session_state.json with a persistent, multi-process safe
database. Tracks job lifecycle, outputs, checkpoints, and parent-child
relationships for spec_params resolution (output_dir, resume_model,
parent_model, etc.) without requiring MongoDB.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

try:  # POSIX interprocess file locking.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by the Windows simulation test.
    _fcntl = None

try:  # Windows interprocess byte-range locking.
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - normal on POSIX.
    _msvcrt = None

from tao_sdk.checkpoints import build_checkpoint_candidate, select_checkpoint_path
from tao_sdk.spec_serialization import normalize_specs

logger = logging.getLogger(__name__)

_TERMINAL_JOB_STATUSES = ("Complete", "Error", "Canceled")
_DEFAULT_STATE_DIR = ".tao_sdk"
_DEFAULT_STATE_FILE = "tao_session_state.json"
_LEGACY_STATE_FILES = ("tao_session_state.json",)
_LEGACY_DB_FILES = ("tao_session_state.db", "tao_session.db")


def _effective_uid() -> int | None:
    """Return the effective uid where the host exposes POSIX ownership."""
    get_euid = getattr(os, "geteuid", None)
    return get_euid() if callable(get_euid) else None


def _set_private_mode(path: Path, mode: int) -> None:
    """Apply POSIX owner-only permissions without misusing Windows chmod."""
    if _effective_uid() is not None:
        os.chmod(path, mode)


def _prepare_private_state_dir(state_dir: Path) -> None:
    """Create or harden the per-user directory that owns durable SDK state."""
    if state_dir.is_symlink():
        raise ValueError("TAO SDK state directory must not be a symbolic link")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = state_dir.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("TAO SDK state path must be a directory")
    effective_uid = _effective_uid()
    if effective_uid is not None:
        owner_uid = getattr(info, "st_uid", None)
        if owner_uid is not None and owner_uid != effective_uid:
            raise PermissionError(
                "TAO SDK state directory must be owned by the current user"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            os.chmod(state_dir, 0o700)


@contextmanager
def _exclusive_migration_lock(lock_path: Path):
    """Hold a cross-process lock across database initialization or migration."""
    with lock_path.open("a+b") as lock_file:
        if _fcntl is not None:
            _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
            return

        if _msvcrt is None:
            raise RuntimeError(
                "No supported interprocess file-locking API is available"
            )

        # ``msvcrt.locking`` locks bytes from the current file position. Keep
        # one byte in the file so every process contends on the same range.
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()

        while True:
            lock_file.seek(0)
            try:
                _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if (
                    exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} and
                    getattr(exc, "winerror", None) != 33
                ):
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            lock_file.seek(0)
            _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)


def merge_legacy_database(source_path: Path, db_path: Path) -> int:
    """Merge missing legacy jobs and their checkpoints into the global DB."""
    source = sqlite3.connect(str(source_path))
    source.row_factory = sqlite3.Row
    target = JobStore(db_path=db_path)
    imported: set[str] = set()
    parent_ids: dict[str, str] = {}
    try:
        has_jobs = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if not has_jobs:
            return 0
        with target._lock:
            target._conn.execute("BEGIN IMMEDIATE")
            try:
                for source_row in source.execute("SELECT * FROM jobs").fetchall():
                    row = dict(source_row)
                    job_id = str(row.get("job_id", "") or "")
                    if not job_id:
                        continue
                    specs = row.get("specs", {})
                    if isinstance(specs, str):
                        try:
                            specs = json.loads(specs)
                        except (TypeError, ValueError):
                            specs = {}
                    if not isinstance(specs, dict):
                        specs = {}
                    inserted = target.create_job_if_absent(
                        job_id=job_id,
                        network_arch=str(row.get("network_arch", "") or ""),
                        action=str(row.get("action", "") or ""),
                        workspace_id=str(row.get("workspace_id", "") or ""),
                        backend_type=str(row.get("backend_type", "") or ""),
                        backend_job_id=str(row.get("backend_job_id", "") or ""),
                        status=str(row.get("status", "Pending") or "Pending"),
                        # Link parents after all missing rows have been inserted.
                        parent_job_id=None,
                        image=str(row.get("image", "") or ""),
                        specs=specs,
                        results_dir=str(row.get("results_dir", "") or ""),
                        local_results_dir=str(row.get("local_results_dir", "") or ""),
                        base_checkpoint=str(row.get("base_checkpoint", "") or ""),
                        plan_id=str(row.get("plan_id", "") or ""),
                        step_id=int(row.get("step_id", 0) or 0),
                        created_at=str(row.get("created_at", "") or _now()),
                    )
                    if not inserted:
                        continue
                    imported.add(job_id)
                    parent_id = str(row.get("parent_job_id", "") or "")
                    if parent_id:
                        parent_ids[job_id] = parent_id
                    target._conn.execute(
                        """UPDATE jobs
                           SET last_epoch = ?, instance_id = ?, updated_at = ?
                           WHERE job_id = ?""",
                        (
                            int(row.get("last_epoch", 0) or 0),
                            str(row.get("instance_id", "") or ""),
                            str(row.get("updated_at", "") or _now()),
                            job_id,
                        ),
                    )

                for job_id, parent_id in parent_ids.items():
                    if target.get_job(parent_id):
                        target._conn.execute(
                            "UPDATE jobs SET parent_job_id = ? WHERE job_id = ?",
                            (parent_id, job_id),
                        )

                has_checkpoints = source.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type = 'table' AND name = 'checkpoints'"""
                ).fetchone()
                if has_checkpoints and imported:
                    placeholders = ", ".join("?" for _ in imported)
                    rows = source.execute(
                        f"SELECT * FROM checkpoints WHERE job_id IN ({placeholders})",
                        tuple(imported),
                    ).fetchall()
                    for checkpoint_row in rows:
                        checkpoint = dict(checkpoint_row)
                        target._conn.execute(
                            """INSERT INTO checkpoints
                               (job_id, path, epoch, format, created_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                checkpoint["job_id"],
                                checkpoint["path"],
                                int(checkpoint.get("epoch", 0) or 0),
                                str(
                                    checkpoint.get("format", "safetensors") or
                                    "safetensors"
                                ),
                                str(checkpoint.get("created_at", "") or _now()),
                            ),
                        )
                target._conn.execute("COMMIT")
            except BaseException:
                target._conn.execute("ROLLBACK")
                raise
        return len(imported)
    finally:
        target.close()
        source.close()


def _migrate_legacy_default_state(state_path: Path, db_path: Path) -> None:
    """Move cwd-relative pre-upgrade state into the stable default location."""
    lock_path = state_path.parent / ".legacy-state-migration.lock"
    with _exclusive_migration_lock(lock_path):
        legacy_json = next(
            (
                path
                for name in _LEGACY_STATE_FILES
                if (path := Path.cwd() / name).is_file() and
                path.resolve() != state_path.resolve()
            ),
            None,
        )
        if legacy_json is not None and not state_path.exists():
            tmp_state = state_path.with_name(
                f".{state_path.name}.{os.getpid()}.migrating"
            )
            try:
                tmp_state.write_bytes(legacy_json.read_bytes())
                _set_private_mode(tmp_state, 0o600)
                os.replace(tmp_state, state_path)
            finally:
                tmp_state.unlink(missing_ok=True)

        legacy_databases = [
            path
            for name in _LEGACY_DB_FILES
            if (path := Path.cwd() / name).is_file() and
            path.resolve() != db_path.resolve()
        ]
        if legacy_databases and not db_path.exists():
            legacy_db = legacy_databases.pop(0)
            tmp_db = db_path.with_name(
                f".{db_path.name}.{os.getpid()}.migrating"
            )
            tmp_db.unlink(missing_ok=True)
            source = sqlite3.connect(str(legacy_db))
            destination = sqlite3.connect(str(tmp_db))
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
            _set_private_mode(tmp_db, 0o600)
            os.replace(tmp_db, db_path)

        for legacy_db in legacy_databases:
            imported = merge_legacy_database(legacy_db, db_path)
            if imported:
                logger.info(
                    "Merged %d jobs from legacy SDK database %s",
                    imported,
                    legacy_db,
                )

        if legacy_json is not None:
            # ``JobStore`` is defined by the time this function is called.
            store = JobStore(db_path=db_path)
            try:
                imported = store.import_from_state_file(str(legacy_json))
                if imported:
                    logger.info(
                        "Merged %d jobs from legacy SDK state %s",
                        imported,
                        legacy_json,
                    )
            finally:
                store.close()


def resolve_sdk_state_paths(
    state_file: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Return the monitor-state and SQLite paths for an SDK instance.

    An omitted path uses the stable per-user ``TAO_SDK_STATE_DIR`` (or
    ``~/.tao_sdk``). An explicitly supplied path retains the historical
    spelling and ``.json``-to-``.db`` conversion, including relative paths.
    """
    if state_file is None:
        configured_dir = os.environ.get("TAO_SDK_STATE_DIR", "").strip()
        state_dir = (
            Path(configured_dir).expanduser()
            if configured_dir
            else Path.home() / _DEFAULT_STATE_DIR
        )
        if not state_dir.is_absolute():
            raise ValueError("TAO_SDK_STATE_DIR must be an absolute path")
        _prepare_private_state_dir(state_dir)
        state_path_obj = state_dir / _DEFAULT_STATE_FILE
        db_path_obj = state_path_obj.with_suffix(".db")
        _migrate_legacy_default_state(state_path_obj, db_path_obj)
        state_path = str(state_path_obj)
    else:
        state_path = os.fspath(state_file)

    db_path = (
        state_path[:-5] + ".db"
        if state_path.endswith(".json")
        else state_path + ".db"
    )
    return state_path, db_path


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


def _locked_operation(method):
    """Serialize one complete operation on a shared SQLite connection."""
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


class JobStore:
    """SQLite-backed job metadata store.

    Thread-safe and multi-process safe via WAL mode. All public methods
    are self-contained transactions.

    Args:
        db_path: Path to the SQLite database file. When omitted, uses the
            stable ``TAO_SDK_STATE_DIR``/per-user SDK state directory. Created
            if it doesn't exist. Use ":memory:" for testing.
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None):
        if db_path is None:
            _, resolved_db_path = resolve_sdk_state_paths()
            db_path = resolved_db_path
        db_path = os.fspath(db_path)
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        # Journal-mode conversion precedes SQLite migration transactions and
        # can race with another opener's schema writes. Serialize the complete
        # initialization boundary, not just ALTER TABLE. In-memory databases
        # are private to their connections and need no filesystem sidecar.
        initialization_lock = (
            nullcontext() if db_path == ":memory:" else
            _exclusive_migration_lock(Path(str(Path(db_path).resolve()) + ".init.lock"))
        )
        try:
            with initialization_lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.executescript(_SCHEMA)
                self._migrate()
        except BaseException:
            self._conn.close()
            raise

    @_locked_operation
    def _migrate(self) -> None:
        """Add columns that may be missing from older databases."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for col, ddl in [
                ("plan_id", "TEXT DEFAULT ''"),
                ("step_id", "INTEGER DEFAULT 0"),
                ("instance_id", "TEXT DEFAULT ''"),
            ]:
                columns = {
                    r["name"]
                    for r in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if col not in columns:
                    self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ddl}")
                    logger.info("Migrated jobs table: added column %s", col)
            # Create index after columns are guaranteed to exist.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_plan_step ON jobs(plan_id, step_id)"
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    @_locked_operation
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
        normalized_specs = normalize_specs(specs)
        serialized_specs = json.dumps(
            normalized_specs,
            allow_nan=False,
        )
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
                serialized_specs,
                results_dir, local_results_dir, base_checkpoint,
                plan_id, step_id,
                now, now,
            ),
        )

    @_locked_operation
    def create_job_if_absent(
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
    ) -> bool:
        """Insert a job without replacing a row created by another process."""
        serialized_specs = json.dumps(normalize_specs(specs), allow_nan=False)
        now = created_at or _now()
        cursor = self._conn.execute(
            """INSERT INTO jobs
               (job_id, network_arch, action, workspace_id, backend_type,
                backend_job_id, status, parent_job_id, image, specs,
                results_dir, local_results_dir, base_checkpoint,
                plan_id, step_id,
                last_epoch, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
               ON CONFLICT(job_id) DO NOTHING""",
            (
                job_id, network_arch, action, workspace_id, backend_type,
                backend_job_id, status, parent_job_id, image,
                serialized_specs,
                results_dir, local_results_dir, base_checkpoint,
                plan_id, step_id,
                now, now,
            ),
        )
        return cursor.rowcount == 1

    @_locked_operation
    def get_job(self, job_id: str) -> dict | None:
        """Fetch a job by ID. Returns dict or None."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    @_locked_operation
    def update_job_status(
        self, job_id: str, status: str, last_epoch: int | None = None
    ) -> bool:
        """Update status monotonically and optionally record the last epoch.

        Once any process has persisted a terminal state, a stale monitor or
        status poll cannot move the row back to an active state or overwrite a
        different terminal outcome.
        """
        terminal_placeholders = ", ".join("?" for _ in _TERMINAL_JOB_STATUSES)
        predicate = (
            f"job_id = ? AND (status NOT IN ({terminal_placeholders}, 'Canceling') "
            "OR status = ?"
        )
        if status in _TERMINAL_JOB_STATUSES:
            predicate += " OR status = 'Canceling'"
        predicate += ")"
        if last_epoch is not None:
            cursor = self._conn.execute(
                f"""UPDATE jobs SET status = ?, last_epoch = ?, updated_at = ?
                    WHERE {predicate}""",
                (
                    status,
                    last_epoch,
                    _now(),
                    job_id,
                    *_TERMINAL_JOB_STATUSES,
                    status,
                ),
            )
        else:
            cursor = self._conn.execute(
                f"""UPDATE jobs SET status = ?, updated_at = ?
                    WHERE {predicate}""",
                (
                    status,
                    _now(),
                    job_id,
                    *_TERMINAL_JOB_STATUSES,
                    status,
                ),
            )
        return cursor.rowcount == 1

    @_locked_operation
    def compare_and_set_job_status(
        self,
        job_id: str,
        expected_statuses: tuple[str, ...],
        status: str,
    ) -> bool:
        """Atomically update a job only while it has an expected status."""
        if not expected_statuses:
            raise ValueError("expected_statuses must not be empty")
        placeholders = ", ".join("?" for _ in expected_statuses)
        cursor = self._conn.execute(
            f"""UPDATE jobs SET status = ?, updated_at = ?
                WHERE job_id = ? AND status IN ({placeholders})""",
            (status, _now(), job_id, *expected_statuses),
        )
        return cursor.rowcount == 1

    @_locked_operation
    def update_job_backend_id(
        self, job_id: str, backend_job_id: str, status: str | None = None,
    ) -> bool:
        """Attach a backend process identifier after a pending launch starts."""
        if status is None:
            cursor = self._conn.execute(
                "UPDATE jobs SET backend_job_id = ?, updated_at = ? WHERE job_id = ?",
                (backend_job_id, _now(), job_id),
            )
        else:
            cursor = self._conn.execute(
                """UPDATE jobs SET backend_job_id = ?, status = ?, updated_at = ?
                   WHERE job_id = ? AND status IN ('Pending', 'Running')""",
                (backend_job_id, status, _now(), job_id),
            )
        return cursor.rowcount == 1

    @_locked_operation
    def update_job_specs(self, job_id: str, specs: dict) -> bool:
        """Atomically replace durable job metadata without changing lifecycle state."""
        normalized_specs = normalize_specs(specs)
        serialized_specs = json.dumps(normalized_specs, allow_nan=False)
        cursor = self._conn.execute(
            "UPDATE jobs SET specs = ?, updated_at = ? WHERE job_id = ?",
            (serialized_specs, _now(), job_id),
        )
        return cursor.rowcount == 1

    @_locked_operation
    def compare_and_set_job_specs(
        self,
        job_id: str,
        expected_specs: dict,
        specs: dict,
        *,
        expected_statuses: tuple[str, ...] | None = None,
    ) -> bool:
        """Replace metadata only if both its generation and lifecycle match."""
        expected = json.dumps(normalize_specs(expected_specs), allow_nan=False)
        replacement = json.dumps(normalize_specs(specs), allow_nan=False)
        query = "UPDATE jobs SET specs = ?, updated_at = ? WHERE job_id = ? AND specs = ?"
        params: list = [replacement, _now(), job_id, expected]
        if expected_statuses is not None:
            if not expected_statuses:
                raise ValueError("expected_statuses must not be empty")
            placeholders = ", ".join("?" for _ in expected_statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(expected_statuses)
        cursor = self._conn.execute(query, params)
        return cursor.rowcount == 1

    @_locked_operation
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

    @_locked_operation
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

    @_locked_operation
    def add_checkpoint(
        self,
        job_id: str,
        path: str,
        epoch: int = 0,
        format: str = "safetensors",  # pylint: disable=redefined-builtin
    ) -> None:
        """Record a checkpoint produced by a job."""
        self._conn.execute(
            """INSERT INTO checkpoints (job_id, path, epoch, format, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, path, epoch, format, _now()),
        )

    @_locked_operation
    def get_checkpoints(self, job_id: str) -> list[dict]:
        """Get all checkpoints for a job, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY epoch DESC",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    @_locked_operation
    def get_latest_checkpoint(self, job_id: str) -> dict | None:
        """Get the most recent checkpoint for a job."""
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY epoch DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None

    @_locked_operation
    def delete_checkpoints(self, job_id: str) -> None:
        """Remove stale checkpoint index rows after artifact deletion."""
        self._conn.execute("DELETE FROM checkpoints WHERE job_id = ?", (job_id,))

    @_locked_operation
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

    @_locked_operation
    def get_parent_job(self, job_id: str) -> dict | None:
        """Get the parent job of a given job."""
        row = self._conn.execute(
            """SELECT p.* FROM jobs p
               JOIN jobs c ON c.parent_job_id = p.job_id
               WHERE c.job_id = ?""",
            (job_id,),
        ).fetchone()
        return dict(row) if row else None

    @_locked_operation
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

    @_locked_operation
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

    @_locked_operation
    def _infer_output_dir(self, job_id: str) -> str | None:
        """Same as FTMS infer_output_dir — returns the results cloud URI."""
        job = self.get_job(job_id)
        return job["results_dir"] if job else None

    @_locked_operation
    def _infer_resume_model(self, job_id: str) -> str | None:
        """Checkpoint path from the parent job."""
        return self.get_parent_model_path(job_id) or None

    @_locked_operation
    def _infer_resume_model_bool(self, job_id: str) -> str | None:
        """'true' if the parent job has checkpoints."""
        path = self.get_parent_model_path(job_id)
        return "true" if path else "false"

    @_locked_operation
    def _infer_parent_model_folder(self, job_id: str) -> str | None:
        """Results dir of the parent job."""
        return self.get_parent_results_dir(job_id) or None

    @_locked_operation
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

    @_locked_operation
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

    @_locked_operation
    def import_from_state_file(self, state_file: str) -> int:
        """Import jobs from a tao_session_state.json file.

        Returns the number of jobs imported.
        """
        path = Path(state_file)
        if not path.exists():
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse state file %s", state_file)
            return 0

        count = 0
        for entry in state.get("active_jobs", []):
            job_id = entry.get("job_id", "")
            if not job_id:
                continue
            inserted = self.create_job_if_absent(
                job_id=job_id,
                network_arch=entry.get("network_arch", ""),
                action=entry.get("action", ""),
                workspace_id=entry.get("workspace_id", ""),
                backend_type=entry.get("backend_type", ""),
                backend_job_id=entry.get("backend_job_id", ""),
                status=entry.get("status", "Unknown"),
                created_at=entry.get("created_at", _now()),
            )
            if inserted:
                count += 1
        logger.info("Imported %d jobs from %s", count, state_file)
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @_locked_operation
    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
