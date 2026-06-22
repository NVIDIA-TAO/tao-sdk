# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM platform SDK for TAO.

Submits and monitors GPU training jobs on a remote SLURM cluster via SSH.
Stages compact specs/env/cloud metadata files on Lustre, runs ``sbatch`` with
Pyxis/Enroot ``srun --container-image`` to launch jobs in containers, and
queries ``squeue``/``sacct`` for status.

Usage::

    from tao_sdk.platforms.slurm import SlurmSDK
    sdk = SlurmSDK()  # reads SLURM_USER, SLURM_HOSTNAME, etc. from env
    job = sdk.create_job(
        image='nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt',
        command='visual_changenet train -e /tmp/spec.yaml',
        gpu_count=8,
        num_nodes=1,
        inputs={'/data/train.csv': 'lustre:///lustre/fsw/portfolios/.../train.csv'},
        outputs=['/results/'],
    )
"""

from __future__ import annotations

import logging
import os
import uuid

from tao_sdk.platforms.slurm.handler import SlurmHandler
from tao_sdk.job_store import JobStore
from tao_sdk.models import CredentialError, Job, JobStatus
from tao_sdk.monitor import JobMonitor

logger = logging.getLogger(__name__)


class SlurmSDK:
    """SDK for running jobs on a remote SLURM cluster via SSH.

    Reads credentials from environment variables:
        - SLURM_USER (or SSH_USER) — required
        - SLURM_HOSTNAME (or SSH_HOST) — required (comma-separated for failover)
        - SSH_KEY_PATH — optional (defaults to ssh-agent / ~/.ssh/id_*)
        - SLURM_BASE_RESULTS_DIR — Lustre directory for stage + outputs
        - SLURM_PARTITION, SLURM_ACCOUNT — optional sbatch params
        - SLURM_CONTAINER_MOUNTS — extra Pyxis container mounts
        - NGC_KEY (for nvcr.io image pulls), HF_TOKEN (passed to containers)
        - S3_BUCKET_NAME, ACCESS_KEY, SECRET_KEY, S3_ENDPOINT_URL (for S3 I/O)
    """

    def __init__(
        self,
        poll_interval: int = 30,
        epoch_milestone_interval: int = 10,
        state_file: str = "tao_session_state.json",
    ):
        self._handler = SlurmHandler.from_credentials(self._get_creds_from_env())

        db_path = state_file.replace(".json", ".db") if state_file.endswith(".json") else state_file + ".db"
        self._store = JobStore(db_path=db_path)
        self._monitor = JobMonitor(
            poll_interval=poll_interval,
            epoch_milestone_interval=epoch_milestone_interval,
            state_file=state_file,
            store=self._store,
        )
        self._monitor.start()
        self._jobs: dict[str, Job] = {}

    @staticmethod
    def _get_creds_from_env() -> dict:
        keys = (
            "SLURM_USER", "SSH_USER", "SLURM_HOSTNAME", "SSH_HOST",
            "SSH_KEY_PATH", "SLURM_BASE_RESULTS_DIR",
            "SLURM_PARTITION", "SLURM_ACCOUNT", "SLURM_CONTAINER_MOUNTS",
            "NGC_KEY", "ACCESS_KEY", "SECRET_KEY",
            "S3_ENDPOINT_URL", "S3_BUCKET_NAME", "CLOUD_REGION",
            "HF_TOKEN",
        )
        return {k: os.environ[k] for k in keys if k in os.environ}

    def _results_root_prefix(self) -> str:
        """Lustre prefix injected as ``TAO_RESULTS_ROOT`` in the container.
        Handler auto-fills ``base_results_dir``, so this always returns a
        path. No S3 upload — avoids GPU-idle scheduler kill."""
        return f"{self._handler.base_results_dir.rstrip('/')}/results"

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def create_job(
        self,
        image: str,
        command: str,
        gpu_count: int = 1,
        env_vars: dict | None = None,
        hooks: dict | None = None,
        # SLURM-specific:
        num_nodes: int = 1,
        partition: str | None = None,
        account: str | None = None,
        mounts: list | None = None,
    ) -> Job:
        """Submit a job to SLURM.

        The agent constructs ``command`` itself — typically by calling
        ``tao_sdk.script_runner.build_entrypoint(...)`` first to inline the
        in-container runtime + S3/HF/NGC download + spec rewriting + output
        upload.

        Args:
            image: Container image URI (Pyxis pulls and converts to SQSH).
            command: Full shell command. Runs as-is inside the container —
                no SDK wrapping. Use ``build_entrypoint`` for S3 I/O.
            gpu_count: Number of GPUs **per node**.
            env_vars: Additional container env vars (S3/NGC/HF credentials
                auto-injected from os.environ).
            hooks: Callbacks: on_complete, on_error, on_epoch_milestone.
            num_nodes: Multi-node distributed training. Default 1.
            partition: SLURM partition (queue) override.
            account: SLURM accounting key override.
            mounts: Extra Pyxis container-mount strings.
        """
        job_id = str(uuid.uuid4())

        bucket = os.environ.get("S3_BUCKET_NAME", "")
        results_dir = (
            f"lustre://{self._handler.base_results_dir}/results/{job_id}"
            if self._handler.base_results_dir
            else (f"s3://{bucket}/results/{job_id}" if bucket else "")
        )

        # Build container env — inject S3 creds (for s3:// inputs via fsspec)
        # plus NGC/HF creds. No S3_BUCKET_NAME — outputs always go to Lustre
        # via TAO_RESULTS_ROOT, never S3.
        container_env = dict(env_vars or {})
        _CRED_TO_ENV = {
            "ACCESS_KEY": "AWS_ACCESS_KEY_ID",
            "SECRET_KEY": "AWS_SECRET_ACCESS_KEY",
            "S3_ENDPOINT_URL": "AWS_ENDPOINT_URL",
            "HF_TOKEN": "HF_TOKEN",
            "NGC_KEY": "NGC_KEY",
        }
        for cred_key, env_key in _CRED_TO_ENV.items():
            val = os.environ.get(cred_key, "")
            if val and env_key not in container_env:
                container_env[env_key] = val

        container_env["TAO_JOB_ID"] = job_id
        container_env["TAO_RESULTS_ROOT"] = self._results_root_prefix()

        cloud_metadata = self._build_cloud_metadata()

        # Agent's command runs as-is.
        entrypoint = {"command": command, "args_template": ""}

        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="slurm", backend_type="slurm",
            results_dir=results_dir,
        )

        resource_shape, _ = self._handler.resolve_resource_shape(gpu_count)

        self._handler.create_job(
            image=image,
            network="",
            action="",
            cloud_metadata=cloud_metadata,
            specs={},
            job_id=job_id,
            docker_env_vars=container_env,
            resource_shape=resource_shape,
            num_nodes=num_nodes,
            entrypoint=entrypoint,
            mounts=mounts,
            partition=partition,
            account=account,
        )

        backend_job_id = self._handler.get_job_name(job_id)
        job = Job(
            id=job_id,
            network_arch="",
            action="",
            workspace_id="slurm",
            backend_type="slurm",
            backend_job_id=backend_job_id,
            results_dir=results_dir,
        )
        self._jobs[job_id] = job

        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="slurm", backend_type="slurm",
            backend_job_id=backend_job_id, image=image,
            results_dir=results_dir,
        )

        if hooks:
            self._monitor.watch(job, hooks, self._handler)

        logger.info("Job %s submitted (slurm: %s, results: %s)",
                    job_id, backend_job_id, results_dir)
        return job

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def get_job_status(self, job_id: str) -> JobStatus:
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
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return ""
        logs = self._handler.get_job_logs(job_id, tail_lines=tail)
        return logs or ""

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return False
        success = self._handler.cancel_job(job_id)
        if success:
            job.status = "Canceled"
            self._monitor.unwatch(job_id)
        return success

    def get_failure_analysis(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return None
        if hasattr(self._handler, "get_failure_analysis"):
            return self._handler.get_failure_analysis(job_id)
        return None

    def get_job_results_dir(self, job_id: str) -> str:
        entry = self._store.get_job(job_id)
        return entry.get("results_dir", "") if entry else ""

    def get_job(self, job_id: str) -> dict | None:
        """Read a job's record from the persistent store."""
        return self._store.get_job(job_id)

    def list_jobs(self, status: str | None = None) -> list[dict]:
        """List all jobs in the persistent store, optionally filtered by status."""
        return self._store.list_jobs(status=status)

    def get_checkpoints(self, job_id: str) -> list[str]:
        """List checkpoint files in the job's results dir, filtered by extension.

        For ``lustre://`` results dirs, listing is currently not implemented —
        returns an empty list. S3-backed results dirs work via ``list_path``.
        """
        results = self.get_job_results_dir(job_id)
        if not results or results.startswith("lustre://"):
            return []
        try:
            files = self.list_path(results)
        except Exception:
            return []
        exts = (".pth", ".tlt", ".hdf5", ".pt", ".ckpt", ".engine", ".onnx", ".trt")
        return [f for f in files if f.lower().endswith(exts)]

    # ------------------------------------------------------------------
    # Dataset utilities
    # ------------------------------------------------------------------

    def check_path(self, remote_path: str) -> bool:
        """Check if a path exists. For lustre:// paths, queries via SSH."""
        if remote_path.startswith("lustre://"):
            return self._handler.check_path_exists(remote_path)
        try:
            fs = self._get_fs()
            path = remote_path.split("://", 1)[1] if "://" in remote_path else remote_path
            return fs.exists(path)
        except Exception as e:
            logger.warning("check_path failed for %s: %s", remote_path, e)
            return False

    def list_path(self, remote_path: str) -> list[str]:
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
            network_arch=entry.get("network_arch", ""),
            action=entry.get("action", ""),
            workspace_id=entry.get("workspace_id", "slurm"),
            backend_type="slurm",
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

    def _build_cloud_metadata(self) -> dict:
        bucket = os.environ.get("S3_BUCKET_NAME", "")
        return {
            "aws": {
                bucket: {
                    "cloud_region": os.environ.get("CLOUD_REGION", ""),
                    "access_key": os.environ.get("ACCESS_KEY", ""),
                    "secret_key": os.environ.get("SECRET_KEY", ""),
                    "endpoint_url": os.environ.get("S3_ENDPOINT_URL", ""),
                }
            }
        }
