# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev execution handler for TAO SDK.

Runs jobs on Brev GPU instances via the ``brev`` CLI. The handler is
stateless — it executes whatever command is given on the specified (or
newly created) instance. The agent decides the instance lifecycle
(reuse vs ephemeral).

Instance lifecycle:
  - If ``instance_id`` is in backend_details, reuse that instance.
  - If not, create a new instance for the job.
  - The handler never deletes instances on its own — the agent manages that
    via ``delete_instance()``.
"""

from __future__ import annotations

import logging
import re
import uuid

from tao_sdk.platforms.base import BaseHandler
from tao_sdk.platforms.brev.cli import BrevCLI, BrevCLIError

logger = logging.getLogger(__name__)

# Job log file path on the remote instance (per job_id)
_LOG_PATH_TEMPLATE = "/tmp/tao-job-{job_id}.log"

# Cold-start first-exec timeout: covers SSH bring-up + container pull.
# The previous default (30s) truncated valid startup as `exec failed`.
_FIRST_EXEC_TIMEOUT = 600


class BrevHandler(BaseHandler):
    """Execution handler for Brev GPU instances.

    Implements the ExecutionHandler protocol by mapping job operations
    to ``brev exec`` commands on remote instances.
    """

    def __init__(self, cli: BrevCLI | None = None, creds: dict | None = None):
        self._cli = cli or BrevCLI()
        self._creds = creds or {}
        # Track job → (instance, container/pid) mapping in memory.
        # The job store is the durable source of truth; this is a runtime cache.
        self._job_instances: dict[str, str] = {}  # job_id → instance_name

    @classmethod
    def from_credentials(cls, creds: dict) -> "BrevHandler":
        """Construct a BrevHandler from a credentials dict.

        If BREV_API_TOKEN is in creds, the CLI auto-authenticates
        via ``brev login --token`` (headless mode). Otherwise, the
        user must have run ``brev login`` manually.
        """
        api_token = creds.get("BREV_API_TOKEN", "")
        cli = BrevCLI(api_token=api_token)
        return cls(cli=cli, creds=creds)

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(
        self,
        image: str,
        network: str,
        action: str,
        cloud_metadata: dict,
        specs: dict,
        job_id: str,
        docker_env_vars: dict,
        resource_shape: str | None,
        dedicated_node_group: str | None,
        num_nodes: int,
        entrypoint: dict | None = None,
        **kwargs,
    ) -> object:
        """Run a job on a Brev instance.

        If ``instance_id`` is in kwargs, reuses that instance. Otherwise
        creates a new one based on resource_shape/gpu type.

        The command to run comes from the entrypoint (built by the SDK in
        ``BrevSDK.create_job``). The handler doesn't care whether
        it's a docker run, a python script, or any other command.
        """
        instance_id = kwargs.get("instance_id", "")
        gpu_type = kwargs.get("gpu_type", "")
        cloud_cred_id = kwargs.get("cloud_cred_id", "")
        workspace_group_id = kwargs.get("workspace_group_id", "")
        first_exec = False

        # Resolve instance: reuse existing or create new
        if instance_id:
            instance = self._cli.get_instance(instance_id)
            if not instance:
                logger.info("Instance '%s' not found, creating...", instance_id)
                self._cli.create_instance(
                    instance_id,
                    gpu_type=gpu_type,
                    cloud_cred_id=cloud_cred_id or None,
                    workspace_group_id=workspace_group_id or None,
                )
                self._wait_for_instance(instance_id)
                first_exec = True
        else:
            # Generate a name from the job_id
            instance_id = f"tao-{job_id[:8]}"
            self._cli.create_instance(
                instance_id,
                gpu_type=gpu_type,
                cloud_cred_id=cloud_cred_id or None,
                workspace_group_id=workspace_group_id or None,
            )
            self._wait_for_instance(instance_id)
            first_exec = True

        self._job_instances[job_id] = instance_id

        # Build the command from entrypoint
        if entrypoint and entrypoint.get("command"):
            command = entrypoint["command"]
        else:
            raise ValueError(
                "Brev handler requires an entrypoint with 'command'. "
                "Ensure the skill has a script_runner config."
            )

        # Set environment variables on the remote instance
        env_exports = ""
        all_env = {**docker_env_vars}
        # Add cloud storage creds for S3 access
        for key in ("ACCESS_KEY", "SECRET_KEY", "S3_ENDPOINT_URL",
                     "S3_BUCKET_NAME", "NGC_KEY", "HF_TOKEN"):
            val = self._creds.get(key, "")
            if val:
                all_env[key] = val
        if all_env:
            env_exports = " ".join(
                f"{k}={_shell_quote(v)}" for k, v in all_env.items()
            ) + " "

        # Run the command in background on the instance
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        full_cmd = (
            f"export {env_exports.strip()} && "
            f"({command}) > {log_path} 2>&1 & echo $!"
        ) if env_exports.strip() else (
            f"({command}) > {log_path} 2>&1 & echo $!"
        )

        # Cold-start path: allow time for sshd + container pull on the
        # newly created instance. Reused instances use the normal timeout.
        exec_timeout = _FIRST_EXEC_TIMEOUT if first_exec else 30
        result = self._cli.exec(instance_id, full_cmd, timeout=exec_timeout)
        pid = result.stdout.strip()

        logger.info(
            "Job %s started on instance '%s' (PID=%s, log=%s)",
            job_id, instance_id, pid, log_path,
        )

        return {"job_id": job_id, "instance_id": instance_id, "pid": pid}

    def get_job_name(self, job_id: str) -> str:
        """Return the job ID as-is (no backend-specific naming)."""
        return job_id

    def get_tao_job_status(self, job_id: str) -> str | None:
        """Check if the job process is still running on the instance.

        Returns 'Running', 'Error', or None (complete).
        """
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return "Error"

        # Check if process is still running by looking for the log file
        # and checking for completion markers
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        result = self._cli.exec(
            instance_id,
            f"if [ -f {log_path} ]; then "
            f"  tail -20 {log_path} | grep -c 'Execution status:' || "
            f"  pgrep -f 'tao-job-{job_id[:8]}' > /dev/null 2>&1 && echo RUNNING || echo DONE; "
            f"else echo MISSING; fi",
            timeout=15,
        )
        output = result.stdout.strip()

        if "RUNNING" in output:
            return "Running"
        if "MISSING" in output:
            return "Pending"
        # Process finished — check logs for pass/fail
        return None  # Signals "complete" — caller checks logs for success

    def normalize_status(self, raw_status: str | None) -> str:
        """Map raw status to SDK status string."""
        if raw_status is None:
            return "Complete"
        return raw_status

    def get_job_message(self, job_id: str) -> str:
        """Get last status line from job log."""
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return ""
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        result = self._cli.exec(
            instance_id, f"tail -1 {log_path} 2>/dev/null || echo ''",
            timeout=10,
        )
        return result.stdout.strip()

    def get_job_logs(self, job_id: str, tail_lines: int | None = None) -> str | None:
        """Fetch job logs from the remote instance."""
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return None
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        if tail_lines:
            cmd = f"tail -{tail_lines} {log_path} 2>/dev/null"
        else:
            cmd = f"cat {log_path} 2>/dev/null"
        result = self._cli.exec(instance_id, cmd, timeout=30)
        return result.stdout if result.returncode == 0 else None

    def delete_job(self, job_id: str) -> bool:
        """Kill the job process on the instance. Does NOT delete the instance."""
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return False
        # Kill any process associated with this job
        self._cli.exec(
            instance_id,
            f"pkill -f 'tao-job-{job_id[:8]}' 2>/dev/null; true",
            timeout=10,
        )
        self._job_instances.pop(job_id, None)
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        return self.delete_job(job_id)

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    def get_available_instances(self, **kwargs) -> dict:
        """List available GPU instance types from Brev search."""
        try:
            results = self._cli.search()
            available = {}
            for i, item in enumerate(results):
                platform_id = f"brev-{i}"
                available[platform_id] = {
                    "cluster": "brev",
                    "gpu_type": item.get("gpu", ""),
                    "instance_type": item.get("type", ""),
                    "gpu_count": item.get("gpuCount", 1),
                    "backend_type": "brev",
                }
            return available
        except Exception as e:
            logger.warning("Failed to query Brev instances: %s", e)
            return {}

    def resolve_resource_shape(
        self, num_gpus: int, preferred_node_group: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Map GPU count to a Brev instance type."""
        instances = self.get_available_instances()
        for pid, info in instances.items():
            if info.get("gpu_count", 0) >= num_gpus:
                return info["gpu_type"], "brev"
        return None, None

    # ------------------------------------------------------------------
    # Instance management (called by agent, not by SDK)
    # ------------------------------------------------------------------

    def delete_instance(self, instance_id: str) -> bool:
        """Delete a Brev instance. Called by agent when done with it."""
        return self._cli.delete_instance(instance_id)

    def list_instances(self) -> list[dict]:
        """List running Brev instances."""
        try:
            return self._cli.list_instances()
        except BrevCLIError:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_instance(self, instance_id: str, timeout: int = 300):
        """Wait for a Brev instance to reach RUNNING state AND be exec-ready.

        Two-stage wait:
          1. Poll status until RUNNING/READY.
          2. Probe `brev exec ... -- true` until it succeeds — status flips
             well before sshd/hostname resolution is ready, so a status-only
             wait surfaces as `hostname not resolvable` / `Connection refused`
             on the first real exec.
        """
        import time
        deadline = time.time() + timeout
        status_ok = False
        while time.time() < deadline:
            inst = self._cli.get_instance(instance_id)
            if inst and inst.get("status", "").lower() in ("running", "ready"):
                logger.info("Instance '%s' reached RUNNING", instance_id)
                status_ok = True
                break
            time.sleep(10)
        if not status_ok:
            raise BrevCLIError(
                f"Instance '{instance_id}' did not reach RUNNING state "
                f"within {timeout}s"
            )
        # Bound the exec-readiness probe by whatever wall-clock budget remains
        # (with a 60s floor for very tight timeouts).
        remaining = max(int(deadline - time.time()), 60)
        self._cli.wait_for_exec_ready(instance_id, timeout=remaining)


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\"'\"'") + "'"
