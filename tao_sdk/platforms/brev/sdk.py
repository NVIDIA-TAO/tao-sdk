# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev platform SDK for TAO.

Submits and monitors GPU training jobs on Brev GPU instances.
Credentials come from environment variables only.

Usage::

    from tao_sdk.platforms.brev import BrevSDK
    sdk = BrevSDK()  # reads BREV_API_TOKEN from env (optional)
    job = sdk.create_job(
        image='nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt',
        command='visual_changenet train -e /tmp/spec.yaml',
        gpu_count=1,
        inputs={'/data/train.csv': 's3://bucket/data/train.csv'},
        outputs=['/results/'],
    )
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from tao_sdk.job_store import JobStore
from tao_sdk.models import CredentialError, Job, JobStatus
from tao_sdk.monitor import JobMonitor

logger = logging.getLogger(__name__)


class BrevSDK:
    """SDK for running jobs on Brev GPU instances.

    Reads credentials from environment variables:
        - BREV_API_TOKEN (optional — falls back to manual brev login)
        - S3_BUCKET_NAME, ACCESS_KEY, SECRET_KEY, S3_ENDPOINT_URL (for S3 I/O)
        - NGC_KEY, HF_TOKEN (passed to containers)

    Requires the ``brev`` CLI to be installed.
    Install: see platform-specific instructions at
    https://github.com/brevdev/brev-cli#installation
    Also recommended: ``brev agent-skill install`` to expose CLI primitives
    (search, create, exec, delete) to the agent.
    """

    def __init__(
        self,
        poll_interval: int = 30,
        state_file: str = "tao_session_state.json",
    ):
        # Brev doesn't strictly require a token (manual login works)
        api_token = os.environ.get("BREV_API_TOKEN", "")

        from tao_sdk.platforms.brev.handler import BrevHandler
        from tao_sdk.platforms.brev.cli import BrevCLI
        self._handler = BrevHandler(cli=BrevCLI(api_token=api_token))

        db_path = state_file.replace(".json", ".db") if state_file.endswith(".json") else state_file + ".db"
        self._store = JobStore(db_path=db_path)
        self._monitor = JobMonitor(
            poll_interval=poll_interval,
            state_file=state_file,
            store=self._store,
        )
        self._monitor.start()
        self._jobs: dict[str, Job] = {}

    def _results_root_prefix(self, mounts: list | None = None) -> str | None:
        """Brev shells out to ``docker run``; same rules as the local Docker SDK.
        ``/results`` if a bind mount targets it, else ``None``."""
        for m in mounts or []:
            if isinstance(m, dict) and m.get("container_path", "").rstrip("/") == "/results":
                return "/results"
        return None

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def create_job(
        self,
        image: str,
        command: str,
        gpu_count: int = 1,
        env_vars: dict | None = None,
        # Brev-specific:
        instance_id: str | None = None,
        gpu_type: str | None = None,
        cloud_cred_id: str | None = None,
        workspace_group_id: str | None = None,
    ) -> Job:
        """Submit a job to a Brev GPU instance.

        The agent constructs ``command`` itself — typically by calling
        ``tao_sdk.script_runner.build_entrypoint(...)`` first to inline the
        in-container runtime + S3/HF/NGC download + spec rewriting + output
        upload, then passing the resulting command string here.

        Args:
            image: Container image URI.
            command: Full shell command. Runs as-is inside the container —
                no SDK wrapping. Use ``build_entrypoint`` for S3 I/O.
            gpu_count: Number of GPUs.
            env_vars: Additional container env vars (S3/NGC/HF credentials
                auto-injected from os.environ).
            instance_id: Reuse an existing Brev instance. If None, creates new.
            gpu_type: GPU type for new instance (e.g., 'A100', 'L40S').
            cloud_cred_id: Brev cloud credential ID for instance placement.
                Required on multi-credential accounts. Discover via
                `brev orgs --json`.
            workspace_group_id: Brev workspace group ID for instance placement.
                Required on multi-workspace accounts. Discover via
                `brev ls --json`.
        """
        job_id = str(uuid.uuid4())

        # Results dir for SDK / store metadata only (the agent's
        # build_entrypoint handles any actual S3 I/O inside the container).
        bucket = os.environ.get("S3_BUCKET_NAME", "")
        results_dir = f"s3://{bucket}/results/{job_id}" if bucket else ""

        # Build container env vars + script_runner contract.
        container_env = dict(env_vars or {})
        _CRED_TO_ENV = {
            "ACCESS_KEY": "AWS_ACCESS_KEY_ID",
            "SECRET_KEY": "AWS_SECRET_ACCESS_KEY",
            "S3_ENDPOINT_URL": "AWS_ENDPOINT_URL",
            "S3_BUCKET_NAME": "S3_BUCKET_NAME",     # cloud fallback when no /results bind
            "HF_TOKEN": "HF_TOKEN",
            "NGC_KEY": "NGC_KEY",
        }
        for cred_key, env_key in _CRED_TO_ENV.items():
            val = os.environ.get(cred_key, "")
            if val and env_key not in container_env:
                container_env[env_key] = val

        container_env["TAO_JOB_ID"] = job_id
        # Brev's create_job doesn't yet expose mounts; when added, pass through.
        prefix = self._results_root_prefix()
        if prefix:
            container_env["TAO_RESULTS_ROOT"] = prefix

        # Build the full docker run command for the Brev instance —
        # agent's command runs as-is inside the container.
        env_flags = " ".join(f"-e {k}={v}" for k, v in container_env.items()) if container_env else ""
        docker_command = (
            f"docker run --gpus all --rm {env_flags} "
            f"{image} /bin/bash -c '{command}'"
        )

        # Build entrypoint for handler
        entrypoint = {"command": docker_command, "args_template": ""}

        # Store job
        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="", backend_type="brev",
            results_dir=results_dir,
        )

        # Submit to handler
        self._handler.create_job(
            image=image,
            network="",
            action="",
            cloud_metadata={},
            specs={},
            job_id=job_id,
            docker_env_vars=container_env,
            resource_shape=None,
            dedicated_node_group=None,
            num_nodes=1,
            entrypoint=entrypoint,
            instance_id=instance_id or "",
            gpu_type=gpu_type or "",
            cloud_cred_id=cloud_cred_id or "",
            workspace_group_id=workspace_group_id or "",
        )

        backend_job_id = self._handler.get_job_name(job_id)
        job = Job(
            id=job_id,
            network_arch="",
            action="",
            workspace_id="",
            backend_type="brev",
            backend_job_id=backend_job_id,
            results_dir=results_dir,
        )
        self._jobs[job_id] = job

        logger.info("Job %s submitted to Brev (instance: %s, results: %s)",
                     job_id, instance_id or "new", results_dir)
        return job

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status."""
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return JobStatus(job_id=job_id, status="Unknown", message="Job not found")

        raw = self._handler.get_tao_job_status(job_id)
        status = self._handler.normalize_status(raw)
        message = self._handler.get_job_message(job_id)

        job.status = status
        self._store.update_job_status(job_id, status)

        return JobStatus(job_id=job_id, status=status, message=message)

    def get_job_logs(self, job_id: str, tail: int | None = None) -> str:
        """Fetch job logs."""
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return ""
        logs = self._handler.get_job_logs(job_id, tail_lines=tail)
        return logs or ""

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return False
        success = self._handler.cancel_job(job_id)
        if success:
            job.status = "Canceled"
            self._monitor.unwatch(job_id)
        return success

    def get_failure_analysis(self, job_id: str) -> dict | None:
        """Not supported on Brev — returns None."""
        return None

    def get_job_results_dir(self, job_id: str) -> str:
        """Get the S3 results directory for a job."""
        entry = self._store.get_job(job_id)
        return entry.get("results_dir", "") if entry else ""

    def get_job(self, job_id: str) -> dict | None:
        """Read a job's record from the persistent store."""
        return self._store.get_job(job_id)

    def list_jobs(self, status: str | None = None) -> list[dict]:
        """List all jobs in the persistent store, optionally filtered by status."""
        return self._store.list_jobs(status=status)

    def get_checkpoints(self, job_id: str) -> list[str]:
        """List checkpoint files in the job's results dir, filtered by extension."""
        results = self.get_job_results_dir(job_id)
        if not results:
            return []
        try:
            files = self.list_path(results)
        except Exception:
            return []
        exts = (".pth", ".tlt", ".hdf5", ".pt", ".ckpt", ".engine", ".onnx", ".trt")
        return [f for f in files if f.lower().endswith(exts)]

    # ------------------------------------------------------------------
    # Brev instance management
    # ------------------------------------------------------------------

    def delete_instance(self, instance_id: str) -> bool:
        """Delete a Brev instance. Call when done with it."""
        return self._handler.delete_instance(instance_id)

    def list_instances(self) -> list[dict]:
        """List running Brev instances."""
        return self._handler.list_instances()

    # ------------------------------------------------------------------
    # Dataset utilities
    # ------------------------------------------------------------------

    def check_path(self, remote_path: str) -> bool:
        """Check if a path exists on S3."""
        try:
            fs = self._get_fs()
            path = remote_path.split("://", 1)[1] if "://" in remote_path else remote_path
            return fs.exists(path)
        except Exception as e:
            logger.warning("check_path failed for %s: %s", remote_path, e)
            return False

    def list_path(self, remote_path: str) -> list[str]:
        """List files at a path on S3."""
        try:
            fs = self._get_fs()
            path = remote_path.split("://", 1)[1] if "://" in remote_path else remote_path
            path = path.rstrip("/")
            entries = fs.ls(path)
            return [e.split("/")[-1] for e in entries]
        except Exception as e:
            logger.warning("list_path failed for %s: %s", remote_path, e)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_job_from_store(self, job_id: str):
        entry = self._store.get_job(job_id)
        if not entry:
            return None
        job = Job(
            id=entry["job_id"],
            network_arch="",
            action="",
            workspace_id="",
            backend_type="brev",
            status=entry.get("status", "Unknown"),
            created_at=entry.get("created_at", ""),
            backend_job_id=entry.get("backend_job_id", ""),
            results_dir=entry.get("results_dir", ""),
        )
        self._jobs[job_id] = job
        return job

    def _get_fs(self):
        import fsspec
        storage_opts = {}
        ak = os.environ.get("ACCESS_KEY", "")
        sk = os.environ.get("SECRET_KEY", "")
        ep = os.environ.get("S3_ENDPOINT_URL", "")
        if ak:
            storage_opts["key"] = ak
        if sk:
            storage_opts["secret"] = sk
        if ep:
            storage_opts["client_kwargs"] = {"endpoint_url": ep}
        return fsspec.filesystem("s3", **storage_opts)
