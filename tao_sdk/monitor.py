# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background job monitor with hook dispatch and state file persistence."""

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tao_sdk.models import Job

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"Complete", "Error", "Canceled"}


@dataclass
class _WatchEntry:
    """Internal tracking state for a monitored job."""
    job: Job
    hooks: dict
    handler: Any
    last_status: str = "Pending"
    last_epoch: int = 0


class JobMonitor:
    """Daemon thread that polls job status via backend handlers and fires hooks."""

    def __init__(self, poll_interval: int = 30, epoch_milestone_interval: int = 10,
                 state_file: str = "tao_session_state.json", store=None):
        self._poll_interval = poll_interval
        self._epoch_milestone_interval = epoch_milestone_interval
        self._state_file = state_file
        self._store = store
        self._watched: dict[str, _WatchEntry] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def watch(self, job: Job, hooks: dict, handler: Any) -> None:
        """Register a job for monitoring."""
        if self._store:
            # Keep JobMonitor usable as a standalone public component while
            # ensuring its first status transition has a durable row on which
            # monotonic terminal-state protection can operate. The insert is
            # atomic so a concurrent terminal write can never be replaced.
            self._store.create_job_if_absent(
                job_id=job.id,
                network_arch=job.network_arch,
                action=job.action,
                workspace_id=job.workspace_id,
                backend_type=job.backend_type,
                backend_job_id=job.backend_job_id,
                status=job.status,
                parent_job_id=job.parent_job_id or None,
                results_dir=job.results_dir,
                local_results_dir=job.local_results_dir,
                base_checkpoint=job.base_checkpoint,
            )
            durable = self._store.get_job(job.id)
            if durable:
                durable_status = durable.get("status", job.status)
                job.status = durable_status
                if durable_status in TERMINAL_STATUSES:
                    # Another process already finalized this job. Registering
                    # a stale Job object must never revive backend polling.
                    with self._lock:
                        self._watched.pop(job.id, None)
                    self._write_state_file()
                    return
        with self._lock:
            self._watched[job.id] = _WatchEntry(
                job=job,
                hooks=hooks,
                handler=handler,
                last_status=job.status,
            )
        self._write_state_file()

    def unwatch(self, job_id: str) -> None:
        """Stop monitoring a job."""
        with self._lock:
            self._watched.pop(job_id, None)

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 5)

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            with self._lock:
                entries = list(self._watched.items())
            for job_id, entry in entries:
                if self._stop_event.is_set():
                    break
                try:
                    self._check_one(job_id, entry)
                except Exception:
                    logger.exception("Error polling job %s", job_id)
            self._stop_event.wait(self._poll_interval)

    def _check_one(self, job_id: str, entry: _WatchEntry) -> None:
        """Poll one job and fire hooks on state transitions."""
        if self._store:
            durable = self._store.get_job(job_id)
            if not durable:
                logger.warning(
                    "Job %s disappeared from the durable store before polling",
                    job_id,
                )
                return
            durable_status = durable.get("status", entry.last_status)
            if durable_status in TERMINAL_STATUSES:
                entry.last_status = durable_status
                entry.job.status = durable_status
                self.unwatch(job_id)
                self._write_state_file()
                return
        try:
            raw_status = entry.handler.get_tao_job_status(job_id)
            status = entry.handler.normalize_status(raw_status)
        except Exception:
            logger.exception("Failed to get status for job %s", job_id)
            return

        if self._store:
            if not self._store.update_job_status(job_id, status):
                durable = self._store.get_job(job_id)
                if not durable:
                    logger.warning(
                        "Job %s disappeared from the durable store while polling",
                        job_id,
                    )
                    return
                status = durable.get("status", status)

        if status == entry.last_status:
            # No transition — but if Running, check for epoch milestones
            if status == "Running":
                self._check_epoch_milestone(job_id, entry)
            return

        # Status transition detected
        old_status = entry.last_status
        entry.last_status = status
        entry.job.status = status
        logger.info("Job %s: %s -> %s", job_id, old_status, status)

        self._write_state_file()

        if status == "Complete":
            self._fire_hook(entry, "on_complete", entry.job)
        elif status == "Error":
            self._fire_hook(entry, "on_error", entry.job)

        if status in TERMINAL_STATUSES:
            self.unwatch(job_id)

    def _check_epoch_milestone(self, job_id: str, entry: _WatchEntry) -> None:
        """Check logs for epoch progress and fire milestone hook."""
        if "on_epoch_milestone" not in entry.hooks:
            return
        try:
            logs = entry.handler.get_job_logs(job_id, tail_lines=20)
            if not logs:
                return
            # Look for epoch patterns in TAO training logs
            matches = re.findall(r'[Ee]poch\s+(\d+)(?:/(\d+))?', logs)
            if not matches:
                return
            current_epoch_str, total_epoch_str = matches[-1]
            current_epoch = int(current_epoch_str)
            total_epochs = int(total_epoch_str) if total_epoch_str else 0

            if (current_epoch > entry.last_epoch and
                    current_epoch % self._epoch_milestone_interval == 0):
                entry.last_epoch = current_epoch
                if self._store:
                    self._store.update_job_status(
                        job_id, entry.job.status, last_epoch=current_epoch
                    )
                self._fire_hook(entry, "on_epoch_milestone",
                                entry.job, current_epoch, total_epochs)
        except Exception:
            logger.debug("Failed to check epoch milestone for job %s", job_id)

    def _fire_hook(self, entry: _WatchEntry, hook_name: str, *args) -> None:
        """Safely invoke a registered hook callback."""
        callback = entry.hooks.get(hook_name)
        if callback:
            try:
                callback(*args)
            except Exception:
                logger.exception("Hook '%s' raised for job %s", hook_name, entry.job.id)

    def _write_state_file(self) -> None:
        """Persist current job states to disk."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            jobs = []
            for entry in self._watched.values():
                d = entry.job.to_dict()
                d["last_epoch"] = entry.last_epoch
                d["updated_at"] = now
                jobs.append(d)
        state = {"active_jobs": jobs}
        try:
            with open(self._state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            logger.exception("Failed to write state file %s", self._state_file)
