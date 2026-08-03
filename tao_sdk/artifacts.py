# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety helpers for deleting one job's persisted result artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import stat
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit


ARTIFACT_METADATA_KEY = "_tao_artifacts"
TERMINAL_JOB_STATUSES = frozenset({
    "complete", "completed", "done", "error", "failed", "canceled",
    "cancelled", "stopped", "timeout", "timedout",
})
REMOTE_DELETE_VERIFY_ATTEMPTS = 5
REMOTE_DELETE_VERIFY_INTERVAL_SECONDS = 0.2
_PROCESS_S3_CREDENTIALS: dict[tuple[str, str], tuple[str, str]] = {}
_PROCESS_S3_CREDENTIALS_LOCK = threading.Lock()


def artifact_specs(
    kind: str,
    root: str | os.PathLike[str],
    *,
    endpoint_url: str = "",
    credential_fingerprint: str = "",
    backend_identity: str = "",
    runtime_identity: dict | None = None,
) -> dict:
    """Build non-secret, durable metadata for one job's artifact backend."""
    metadata = {
        "version": 1,
        "kind": str(kind),
        "root": os.fspath(root),
        "endpoint_url": str(endpoint_url or ""),
        "credential_fingerprint": str(credential_fingerprint or ""),
        "backend_identity": str(backend_identity or ""),
    }
    if runtime_identity is not None:
        metadata["runtime_identity"] = dict(runtime_identity)
    return {ARTIFACT_METADATA_KEY: metadata}


def s3_credential_fingerprint(access_key: str) -> str:
    """Return a durable, non-secret identity for an explicit S3 principal."""
    value = str(access_key or "")
    if not value:
        return ""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"s3-access-key-sha256:{digest}"


def normalize_s3_endpoint(endpoint_url: str) -> str:
    """Normalize an S3-compatible endpoint for credential-cache isolation."""
    value = str(endpoint_url or "").strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = parsed.port
        if (scheme, port) in {("http", 80), ("https", 443)}:
            port = None
        user = parsed.username or ""
        user_prefix = f"{user}@" if user else ""
        netloc = f"{user_prefix}{hostname}{f':{port}' if port is not None else ''}"
        return urlunsplit(
            (scheme, netloc, parsed.path.rstrip("/"), parsed.query, parsed.fragment)
        )
    except ValueError:
        return value.rstrip("/")


def remember_process_s3_credentials(
    access_key: str,
    secret_key: str,
    endpoint_url: str = "",
) -> str:
    """Retain explicit S3 credentials in memory for SDK resume in this process.

    Only the access-key fingerprint is persisted with a job. Raw credentials
    remain process-local and are never serialized to JobStore or logs.
    """
    access = str(access_key or "")
    secret = str(secret_key or "")
    fingerprint = s3_credential_fingerprint(access)
    if not fingerprint or not secret:
        return ""
    cache_key = (fingerprint, normalize_s3_endpoint(endpoint_url))
    with _PROCESS_S3_CREDENTIALS_LOCK:
        _PROCESS_S3_CREDENTIALS[cache_key] = (access, secret)
    return fingerprint


def get_process_s3_credentials(
    credential_fingerprint: str,
    endpoint_url: str = "",
) -> tuple[str, str]:
    """Return a process-local credential matching a durable fingerprint."""
    fingerprint = str(credential_fingerprint or "")
    if not fingerprint:
        return "", ""
    cache_key = (fingerprint, normalize_s3_endpoint(endpoint_url))
    with _PROCESS_S3_CREDENTIALS_LOCK:
        return _PROCESS_S3_CREDENTIALS.get(cache_key, ("", ""))


def require_s3_credential(metadata: dict, access_key: str) -> None:
    """Require cleanup to use the same explicit S3 access-key identity."""
    recorded = str(metadata.get("credential_fingerprint", ""))
    current = s3_credential_fingerprint(access_key)
    if not recorded or not current or not hmac.compare_digest(recorded, current):
        raise ValueError(
            "current S3 credential identity does not match the recorded job owner"
        )


def artifact_metadata(entry: dict | None) -> dict:
    """Read artifact metadata from a JobStore row.

    Cleanup deliberately does not infer a target from mutable environment
    variables or from a path alone. Jobs created before metadata was recorded
    therefore require manual review instead of risking deletion elsewhere.
    """
    if not entry:
        return {}
    specs = entry.get("specs", {})
    if isinstance(specs, str):
        try:
            specs = json.loads(specs)
        except (TypeError, ValueError):
            return {}
    if not isinstance(specs, dict):
        return {}
    metadata = specs.get(ARTIFACT_METADATA_KEY, {})
    return metadata if isinstance(metadata, dict) else {}


def require_terminal_job(entry: dict | None) -> None:
    """Refuse artifact deletion while a job may still have active writers."""
    if not entry:
        raise ValueError("job metadata was not found")
    status = str(entry.get("status", "")).strip().lower().replace("_", "")
    if status not in TERMINAL_JOB_STATUSES:
        raise ValueError(
            f"job is not terminal (recorded status: {entry.get('status', 'Unknown')})"
        )


def validate_job_id(job_id: str) -> str:
    """Return a normalized job ID that is safe as one path component."""
    value = str(job_id or "").strip()
    if not value or value in {".", ".."} or PurePosixPath(value).name != value:
        raise ValueError("job_id must be a non-empty path component")
    if "\\" in value:
        raise ValueError("job_id must not contain path separators")
    return value


def job_scoped_remote_path(results_dir: str, job_id: str, scheme: str) -> str:
    """Validate and strip the URI scheme from ``.../results/<job_id>``."""
    safe_job_id = validate_job_id(job_id)
    prefix = f"{scheme}://"
    if not str(results_dir).startswith(prefix):
        raise ValueError(f"results directory must use {scheme}://")
    remote_path = str(results_dir)[len(prefix):].rstrip("/")
    parts = PurePosixPath(remote_path).parts
    if ".." in parts:
        raise ValueError("results directory must not contain parent traversal")
    if len(parts) < 3 or parts[-2:] != ("results", safe_job_id):
        raise ValueError("refusing to delete a non-job-scoped results directory")
    return remote_path


def delete_local_job_results(
    results_dir: str | os.PathLike[str],
    job_id: str,
    *,
    expected_path: str | os.PathLike[str],
) -> bool:
    """Delete a validated local result path without following directory symlinks.

    The caller must derive ``expected_path`` from a durable, trusted root.
    Missing paths are treated as an already-completed, idempotent deletion.
    """
    validate_job_id(job_id)
    candidate = Path(results_dir).expanduser().absolute()
    expected = Path(expected_path).expanduser().absolute()
    if candidate != expected:
        raise ValueError("stored results path does not match the expected job path")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError(
            "this platform cannot safely delete a directory relative to an "
            "open parent descriptor"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        parent_fd = os.open(expected.parent, flags)
    except FileNotFoundError:
        return not os.path.lexists(candidate)
    try:
        # Anchor deletion to the parent descriptor and then prove that the
        # currently named parent is the same non-symlink directory. This closes
        # the resolve-then-rmtree window even if an intermediate path component
        # is exchanged for a symlink while cleanup is starting.
        if expected.parent.resolve(strict=True) != expected.parent:
            raise ValueError("expected job parent contains a symlink or traversal")
        opened_parent = os.fstat(parent_fd)
        named_parent = os.stat(expected.parent, follow_symlinks=False)
        if (
            opened_parent.st_dev != named_parent.st_dev
            or opened_parent.st_ino != named_parent.st_ino
        ):
            raise ValueError("expected job parent changed during deletion")

        def named_path_is_absent_from_opened_parent() -> bool:
            """Verify both the anchored and currently named paths after deletion."""
            try:
                if expected.parent.resolve(strict=True) != expected.parent:
                    return False
                current_parent = os.stat(expected.parent, follow_symlinks=False)
            except OSError:
                return False
            if (
                opened_parent.st_dev != current_parent.st_dev
                or opened_parent.st_ino != current_parent.st_ino
            ):
                return False
            try:
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                return False
            except FileNotFoundError:
                # The fd-relative leaf is absent; also verify that the named
                # path did not move to a replacement parent containing a new
                # job directory during deletion.
                return not os.path.lexists(candidate)
            except OSError:
                return False

        leaf = expected.name
        try:
            leaf_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return named_path_is_absent_from_opened_parent()
        if stat.S_ISDIR(leaf_stat.st_mode):
            # CPython's fd-relative implementation opens every descendant with
            # O_NOFOLLOW and verifies directory identity before recursing.
            shutil.rmtree(leaf, dir_fd=parent_fd)
        else:
            os.unlink(leaf, dir_fd=parent_fd)

        return named_path_is_absent_from_opened_parent()
    finally:
        os.close(parent_fd)


def delete_remote_job_results(
    fs,
    results_dir: str,
    job_id: str,
    scheme: str,
    *,
    verify_attempts: int = REMOTE_DELETE_VERIFY_ATTEMPTS,
    verify_interval: float = REMOTE_DELETE_VERIFY_INTERVAL_SECONDS,
) -> bool:
    """Delete and verify one validated job prefix.

    Object-store deletion can return without removing every object immediately.
    Only report success after the exact job prefix is observed absent so callers
    do not persist a deletion tombstone for a silent or partial failure.
    """
    remote_path = job_scoped_remote_path(results_dir, job_id, scheme)
    if not fs.exists(remote_path):
        return True
    fs.rm(remote_path, recursive=True)
    attempts = max(int(verify_attempts), 1)
    interval = max(float(verify_interval), 0.0)
    for attempt in range(attempts):
        if not fs.exists(remote_path):
            return True
        if attempt + 1 < attempts and interval:
            time.sleep(interval)
    return False
