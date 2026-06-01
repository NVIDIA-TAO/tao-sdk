# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for timestamped TAO action workflow folders with status sync."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"Complete", "Error", "Canceled", "Unknown"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(value).items()}
    return value


class ActionWorkflow:
    """Persist action-run metadata and backend status events to a run folder."""

    def __init__(self, root_dir: str | Path, run_name: str, timestamp: str):
        self.root_dir = Path(root_dir)
        self.run_name = run_name
        self.timestamp = timestamp
        self.workspace = self.root_dir / run_name / timestamp
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.workspace / "metadata.json"
        self.status_path = self.workspace / "status.json"
        self.events_path = self.workspace / "status_events.jsonl"
        self.active_jobs_path = self.workspace / "active_jobs.json"
        self.logs_path = self.workspace / "latest_logs.txt"
        self.failure_analysis_path = self.workspace / "failure_analysis.json"

    def write_metadata(self, **metadata: Any) -> None:
        payload = {"updated_at": _utcnow(), **_to_jsonable(metadata)}
        self.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def write_submission(self, job: Any, specs: dict, script_runner: dict) -> None:
        payload = {
            "updated_at": _utcnow(),
            "job": _to_jsonable(job),
            "specs": _to_jsonable(specs),
            "script_runner": _to_jsonable(script_runner),
        }
        self.status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        active_jobs = [{
            "job_id": getattr(job, "id", ""),
            "backend_job_id": getattr(job, "backend_job_id", ""),
            "network_arch": getattr(job, "network_arch", ""),
            "action": getattr(job, "action", ""),
            "workspace_id": getattr(job, "workspace_id", ""),
            "status": getattr(job, "status", "Pending"),
            "created_at": getattr(job, "created_at", ""),
            "updated_at": _utcnow(),
        }]
        self.active_jobs_path.write_text(json.dumps(active_jobs, indent=2, sort_keys=True), encoding="utf-8")
        self._append_event({
            "event": "submitted",
            "job_id": getattr(job, "id", ""),
            "backend_job_id": getattr(job, "backend_job_id", ""),
            "status": getattr(job, "status", "Pending"),
        })

    def sync_status(self, status: Any, message: str = "", logs: str = "", failure_analysis: Any = None) -> None:
        current = self._read_json(self.status_path)
        job_payload = dict(current.get("job", {}))
        job_payload["status"] = getattr(status, "status", job_payload.get("status", ""))
        payload = {
            **current,
            "job": job_payload,
            "updated_at": _utcnow(),
            "job_status": {
                "job_id": getattr(status, "job_id", ""),
                "status": getattr(status, "status", ""),
                "message": message or getattr(status, "message", ""),
                "updated_at": getattr(status, "updated_at", ""),
            },
        }
        self.status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if logs:
            self.logs_path.write_text(logs, encoding="utf-8")
        if failure_analysis:
            self.failure_analysis_path.write_text(
                json.dumps(_to_jsonable(failure_analysis), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        job_status = getattr(status, "status", "")
        if job_status in TERMINAL_STATUSES:
            self.active_jobs_path.write_text("[]\n", encoding="utf-8")
        self._append_event({
            "event": "status",
            "job_id": getattr(status, "job_id", ""),
            "status": job_status,
            "message": message or getattr(status, "message", ""),
        })

    def sync_from_sdk(self, sdk: Any, job_id: str, tail: int = 200) -> Any:
        """Refresh this workflow folder from a TAO SDK job status query."""
        status = sdk.get_job_status(job_id)
        logs = sdk.get_job_logs(job_id, tail=tail)
        failure_analysis = sdk.get_failure_analysis(job_id)
        self.sync_status(status=status, logs=logs, failure_analysis=failure_analysis)
        return status

    @classmethod
    def from_workspace(cls, workspace: str | Path) -> "ActionWorkflow":
        """Re-open an existing workflow folder."""
        workspace = Path(workspace)
        obj = cls.__new__(cls)
        obj.workspace = workspace
        obj.root_dir = workspace.parent.parent
        obj.run_name = workspace.parent.name
        obj.timestamp = workspace.name
        obj.metadata_path = workspace / "metadata.json"
        obj.status_path = workspace / "status.json"
        obj.events_path = workspace / "status_events.jsonl"
        obj.active_jobs_path = workspace / "active_jobs.json"
        obj.logs_path = workspace / "latest_logs.txt"
        obj.failure_analysis_path = workspace / "failure_analysis.json"
        return obj

    def _append_event(self, event: dict) -> None:
        payload = {"timestamp": _utcnow(), **_to_jsonable(event)}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
