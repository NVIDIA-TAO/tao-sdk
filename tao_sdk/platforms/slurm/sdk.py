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

import json
import logging
import os
import uuid
from pathlib import PurePosixPath

from tao_sdk.artifacts import (
    artifact_metadata,
    artifact_specs,
    job_scoped_remote_path,
    require_terminal_job,
)
from tao_sdk.platforms.slurm.handler import SlurmHandler
from tao_sdk.job_store import JobStore, resolve_sdk_state_paths
from tao_sdk.models import Job, JobStatus
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
        - TAO_SDK_STATE_DIR — optional stable per-user state directory
        - SLURM_CONTAINER_MOUNTS — extra Pyxis container mounts
        - NGC_KEY (for nvcr.io image pulls), HF_TOKEN (passed to containers)
        - S3_BUCKET_NAME, ACCESS_KEY, SECRET_KEY, S3_ENDPOINT_URL (for S3 I/O)
    """

    def __init__(
        self,
        poll_interval: int = 30,
        epoch_milestone_interval: int = 10,
        state_file: str | os.PathLike[str] | None = None,
    ):
        self._handler = SlurmHandler.from_credentials(self._get_creds_from_env())

        state_file, db_path = resolve_sdk_state_paths(state_file)
        self._store = JobStore(db_path=db_path)
        self._handler.set_runtime_state_callback(self._persist_slurm_runtime)
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
        canonical_results_root = self._handler.get_canonical_results_root()
        results_dir = f"lustre://{canonical_results_root}/{job_id}"
        backend_identity = self._handler.get_backend_identity()
        if not backend_identity:
            raise RuntimeError(
                "SLURM cluster did not provide a stable identity; refusing to "
                "launch an unrecoverable job"
            )
        stored_specs = artifact_specs(
            "lustre",
            results_dir,
            backend_identity=backend_identity,
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
        container_env["TAO_RESULTS_ROOT"] = canonical_results_root

        cloud_metadata = self._build_cloud_metadata()

        # Agent's command runs as-is.
        entrypoint = {"command": command, "args_template": ""}
        backend_job_id = self._handler.get_job_name(job_id)
        runtime_identity = self._handler.get_job_runtime_identity(job_id)
        if not isinstance(runtime_identity, dict):
            runtime_identity = {
                "job_name": backend_job_id,
                "slurm_job_id": "",
                "submission_attempted": False,
                "launch_uncertain": False,
            }
        stored_specs["_slurm_runtime"] = runtime_identity

        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="slurm", backend_type="slurm",
            backend_job_id=backend_job_id, image=image,
            results_dir=results_dir,
            specs=stored_specs,
        )
        if self._handler.restore_job_runtime(job_id, runtime_identity) is False:
            self._store.update_job_status(job_id, "Error")
            raise RuntimeError("Could not initialize durable SLURM runtime identity")

        resource_shape, _ = self._handler.resolve_resource_shape(gpu_count)

        try:
            self._handler.create_job(
                image=image,
                network="",
                action="",
                cloud_metadata=cloud_metadata,
                specs=stored_specs,
                job_id=job_id,
                docker_env_vars=container_env,
                resource_shape=resource_shape,
                num_nodes=num_nodes,
                entrypoint=entrypoint,
                mounts=mounts,
                partition=partition,
                account=account,
            )
            self._persist_current_runtime(job_id, required=True)
        except Exception as exc:
            durable_status = self._claim_canceling(job_id)
            quiescent = durable_status in {"Complete", "Error", "Canceled"}
            quiescent = quiescent or bool(
                getattr(exc, "writer_quiescent", False)
            )
            if not quiescent:
                try:
                    quiescent = self._handler.cancel_job(job_id)
                except Exception:
                    quiescent = False
            if quiescent:
                terminal_status = (
                    durable_status
                    if durable_status in {"Complete", "Error", "Canceled"}
                    else self._observed_terminal_status(job_id, "Error")
                )
                try:
                    self._store.update_job_status(job_id, terminal_status)
                except Exception:
                    logger.exception("Could not persist failed SLURM launch status")
                raise
            logger.critical(
                "SLURM launch for job %s failed ambiguously and cancellation "
                "could not be confirmed; returning its durable identity for recovery: %s",
                job_id,
                exc,
            )
        except BaseException as exc:
            durable_status = self._claim_canceling(job_id)
            quiescent = durable_status in {"Complete", "Error", "Canceled"}
            if not quiescent:
                try:
                    quiescent = self._handler.cancel_job(job_id)
                except BaseException:
                    logger.exception(
                        "Could not cancel interrupted SLURM submission %s",
                        job_id,
                    )
                    quiescent = False
            exc.tao_job_id = job_id
            exc.tao_launch_uncertain = not quiescent
            if quiescent:
                terminal_status = (
                    durable_status
                    if durable_status in {"Complete", "Error", "Canceled"}
                    else self._observed_terminal_status(job_id, "Canceled")
                )
                try:
                    self._store.update_job_status(job_id, terminal_status)
                except Exception:
                    logger.exception("Could not persist interrupted SLURM launch status")
            raise

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
        entry = self._store.get_job(job_id)
        if not entry or entry.get("backend_type") != "slurm":
            return JobStatus(job_id=job_id, status="Unknown", message="Job not found")
        durable_status = entry.get("status", "Unknown")
        if durable_status in {"Complete", "Error", "Canceled"}:
            job.status = durable_status
            return JobStatus(
                job_id=job_id,
                status=durable_status,
                message=f"Terminal status {durable_status} was confirmed",
            )
        if durable_status == "Canceling":
            job.status = durable_status
            return JobStatus(
                job_id=job_id,
                status=durable_status,
                message="Cancellation is durably in progress",
            )
        if not self._backend_identity_matches(job_id, entry=entry):
            return JobStatus(
                job_id=job_id,
                status="Unknown",
                message="Job belongs to a different or unverified SLURM cluster",
            )
        raw = self._handler.get_tao_job_status(job_id)
        status = self._handler.normalize_status(raw)
        message = self._handler.get_job_message(job_id)
        if not self._store.update_job_status(job_id, status):
            durable = self._store.get_job(job_id)
            status = durable.get("status", "Unknown") if durable else "Unknown"
        job.status = status
        return JobStatus(job_id=job_id, status=status, message=message)

    def get_job_logs(self, job_id: str, tail: int | None = None) -> str:
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job or not self._backend_identity_matches(job_id):
            return ""
        logs = self._handler.get_job_logs(job_id, tail_lines=tail)
        return logs or ""

    def cancel_job(self, job_id: str) -> bool:
        entry = self._store.get_job(job_id)
        if not entry or entry.get("backend_type") != "slurm":
            return False
        if entry.get("status") in {"Complete", "Error", "Canceled"}:
            job = self._jobs.get(job_id)
            if job:
                job.status = entry["status"]
            self._monitor.unwatch(job_id)
            return True
        if not self._backend_identity_matches(job_id, entry=entry):
            return False
        # Persist cancellation intent before restoring handler state or waiting
        # on its in-process lifecycle lock. A concurrent submit must observe
        # this fence at its mandatory pre-sbatch CAS revalidation.
        durable_status = self._claim_canceling(job_id)
        if durable_status in {"Complete", "Error", "Canceled"}:
            job = self._jobs.get(job_id)
            if job:
                job.status = durable_status
            self._monitor.unwatch(job_id)
            return True
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return False
        try:
            observed = self._handler.normalize_status(
                self._handler.get_tao_job_status(job_id, allow_retry=False)
            )
        except Exception:
            observed = None
        if observed in {"Complete", "Error", "Canceled"}:
            self._store.update_job_status(job_id, observed)
            job.status = observed
            self._monitor.unwatch(job_id)
            return True
        success = self._handler.cancel_job(job_id)
        if success:
            try:
                observed = self._handler.normalize_status(
                    self._handler.get_tao_job_status(job_id, allow_retry=False)
                )
            except Exception:
                observed = None
            desired = (
                observed
                if observed in {"Complete", "Error", "Canceled"}
                else "Canceled"
            )
            self._store.update_job_status(job_id, desired)
            durable = self._store.get_job(job_id)
            job.status = durable.get("status", desired) if durable else desired
            self._monitor.unwatch(job_id)
        return success

    def get_failure_analysis(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job or not self._backend_identity_matches(job_id):
            return None
        if hasattr(self._handler, "get_failure_analysis"):
            return self._handler.get_failure_analysis(job_id)
        return None

    def get_job_results_dir(self, job_id: str) -> str:
        entry = self._store.get_job(job_id)
        return entry.get("results_dir", "") if entry else ""

    def delete_job_artifacts(self, job_id: str) -> bool:
        """Delete the exact Lustre result directory owned by ``job_id``."""
        entry = self._store.get_job(job_id)
        if not entry:
            return False
        try:
            require_terminal_job(entry)
            metadata = artifact_metadata(entry)
            if not self._backend_identity_matches(job_id, entry=entry):
                raise ValueError(
                    "job belongs to a different or unverified SLURM cluster"
                )
            results_dir = entry.get("results_dir", "")
            if metadata.get("kind") != "lustre" or metadata.get("root") != results_dir:
                raise ValueError("recorded Lustre artifact identity is missing or mismatched")
            remote_path = job_scoped_remote_path(
                results_dir, job_id, "lustre"
            )
            recorded = PurePosixPath("/" + remote_path.lstrip("/"))
            expected_root = recorded.parent
            if recorded.name != job_id or expected_root.name != "results":
                raise ValueError("stored path does not match a SLURM job directory")
            deleted = self._handler.delete_job_artifacts(
                job_id, expected_root=str(expected_root)
            )
            if deleted:
                self._store.delete_checkpoints(job_id)
            return deleted
        except Exception as exc:
            logger.warning("Could not delete artifacts for job %s: %s", job_id, exc)
            return False

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

    @staticmethod
    def _runtime_from_entry(entry: dict | None) -> dict | None:
        """Extract validated-shape SLURM runtime metadata from a store row."""
        if not entry:
            return None
        specs = entry.get("specs", {})
        if isinstance(specs, str):
            try:
                specs = json.loads(specs)
            except (TypeError, ValueError):
                return None
        if not isinstance(specs, dict):
            return None
        runtime = specs.get("_slurm_runtime")
        return dict(runtime) if isinstance(runtime, dict) else None

    @staticmethod
    def _validated_runtime_transition(
        current: dict | None,
        incoming: dict,
    ) -> dict:
        """Validate one scheduler-generation transition before its CAS write."""
        candidate = dict(incoming)
        if current is None:
            if candidate.get("launch_uncertain"):
                raise RuntimeError("Cannot fence a SLURM launch without base identity")
            candidate["revision"] = 1
            return candidate

        current = dict(current)
        immutable_keys = (
            "job_name", "job_dir", "log_dir", "remote_script",
            "remote_entrypoint", "remote_specs", "remote_env", "remote_meta",
        )
        if any(
            str(candidate.get(key, "")) != str(current.get(key, ""))
            for key in immutable_keys
        ):
            raise RuntimeError("SLURM runtime recovery coordinates changed")

        current_id = str(current.get("slurm_job_id", "") or "")
        incoming_id = str(candidate.get("slurm_job_id", "") or "")
        current_retry = int(current.get("retry_count", 0))
        incoming_retry = int(candidate.get("retry_count", 0))
        current_token = str(current.get("launch_token", "") or "")
        incoming_token = str(candidate.get("launch_token", "") or "")
        current_fenced = bool(current.get("launch_uncertain"))
        incoming_fenced = bool(candidate.get("launch_uncertain"))
        current_failed = {
            str(value) for value in current.get("failed_slurm_job_ids", [])
        }
        incoming_failed = {
            str(value) for value in candidate.get("failed_slurm_job_ids", [])
        }
        if not current_failed.issubset(incoming_failed):
            raise RuntimeError("SLURM failed scheduler IDs cannot regress")

        if current_fenced:
            if not current_token or incoming_token != current_token:
                raise RuntimeError("SLURM launch fence is owned by another process")
            pre_launch_id = str(
                current.get("pre_launch_slurm_job_id", "") or ""
            )
            if incoming_fenced:
                if incoming_id != current_id or incoming_retry != current_retry:
                    raise RuntimeError("A fenced SLURM generation cannot change identity")
            elif incoming_id == current_id:
                if incoming_retry != current_retry:
                    raise RuntimeError("Rejected SLURM launch changed retry generation")
            else:
                if not incoming_id or (pre_launch_id and incoming_id == pre_launch_id):
                    raise RuntimeError("SLURM launch resolved to its pre-attempt ID")
                expected_retry = current_retry + (1 if pre_launch_id else 0)
                if incoming_retry != expected_retry:
                    raise RuntimeError("Accepted SLURM launch has the wrong generation")
        elif incoming_fenced:
            if not incoming_token:
                raise RuntimeError("SLURM launch fence token is missing")
            if incoming_id != current_id or incoming_retry != current_retry:
                raise RuntimeError("SLURM launch fence was based on stale identity")
            if str(candidate.get("pre_launch_slurm_job_id", "") or "") != current_id:
                raise RuntimeError("SLURM launch fence has the wrong pre-attempt ID")
        else:
            if current_token and incoming_token != current_token:
                raise RuntimeError("Stale SLURM runtime generation")
            if current_id and incoming_id != current_id:
                raise RuntimeError("SLURM scheduler identity cannot regress")
            if incoming_retry != current_retry:
                raise RuntimeError("SLURM retry generation changed without a fence")

        candidate["revision"] = int(current.get("revision", 0)) + 1
        return candidate

    def _persist_slurm_runtime(self, job_id: str, runtime: dict) -> dict:
        """Persist scheduler identity through the authoritative JobStore row."""
        if not isinstance(runtime, dict):
            raise TypeError("SLURM runtime identity must be a dictionary")
        for _attempt in range(16):
            entry = self._store.get_job(job_id)
            if not entry:
                raise RuntimeError(f"Durable SLURM job row {job_id} is missing")
            if entry.get("backend_type") != "slurm":
                raise RuntimeError(f"Durable row {job_id} is not a SLURM job")
            if entry.get("status") not in {"Pending", "Running"}:
                raise RuntimeError(
                    f"SLURM runtime for {job_id} is fenced by durable status "
                    f"{entry.get('status')}"
                )
            specs = entry.get("specs", {})
            if isinstance(specs, str):
                specs = json.loads(specs)
            if not isinstance(specs, dict):
                raise RuntimeError(f"Durable SLURM specs for {job_id} are invalid")
            specs = dict(specs)
            current = specs.get("_slurm_runtime")
            current = dict(current) if isinstance(current, dict) else None
            candidate = self._validated_runtime_transition(current, runtime)
            replacement = dict(specs)
            replacement["_slurm_runtime"] = candidate
            if self._store.compare_and_set_job_specs(
                job_id,
                specs,
                replacement,
                expected_statuses=("Pending", "Running"),
            ):
                return candidate
        raise RuntimeError(f"SLURM runtime for {job_id} changed too frequently")

    def _persist_current_runtime(self, job_id: str, *, required: bool) -> bool:
        """Copy the handler's current scheduler identity into JobStore."""
        runtime = self._handler.get_job_runtime_identity(job_id)
        if not isinstance(runtime, dict):
            if required:
                raise RuntimeError(f"SLURM runtime identity for {job_id} is missing")
            return False
        try:
            persisted = self._persist_slurm_runtime(job_id, runtime)
            if isinstance(persisted, dict):
                self._handler.restore_job_runtime(job_id, persisted)
        except Exception:
            if required:
                raise
            logger.exception("Could not persist SLURM runtime for %s", job_id)
            return False
        return True

    def _observed_terminal_status(self, job_id: str, fallback: str) -> str:
        """Preserve a backend-observed terminal outcome after quiescence."""
        try:
            raw_status = self._handler.get_tao_job_status(
                job_id,
                allow_retry=False,
            )
            status = self._handler.normalize_status(raw_status)
        except Exception:
            return fallback
        return status if status in {"Complete", "Error", "Canceled"} else fallback

    def _claim_canceling(self, job_id: str) -> str:
        """Fence retry side effects before cancellation or launch cleanup."""
        if self._store.compare_and_set_job_status(
            job_id,
            ("Pending", "Running", "Paused"),
            "Canceling",
        ):
            return "Canceling"
        entry = self._store.get_job(job_id)
        return entry.get("status", "Unknown") if entry else "Unknown"

    def _backend_identity_matches(
        self,
        job_id: str,
        *,
        entry: dict | None = None,
    ) -> bool:
        """Require durable proof that this SDK is using the job's cluster."""
        entry = entry or self._store.get_job(job_id)
        if not entry or entry.get("backend_type") != "slurm":
            return False
        recorded = str(artifact_metadata(entry).get("backend_identity", ""))
        current = self._handler.get_backend_identity()
        if not recorded and self._runtime_from_entry(entry) is None and current:
            # Pre-durable-identity SLURM rows were tied to the configured
            # cluster only by backend type and job name. Adopt that identity
            # once, atomically, so scheduler-name recovery remains reachable.
            for _attempt in range(8):
                latest = self._store.get_job(job_id)
                if not latest or latest.get("backend_type") != "slurm":
                    break
                specs = latest.get("specs", {})
                if isinstance(specs, str):
                    try:
                        specs = json.loads(specs)
                    except (TypeError, ValueError):
                        break
                if not isinstance(specs, dict):
                    break
                latest_recorded = str(
                    artifact_metadata(latest).get("backend_identity", "")
                )
                if latest_recorded:
                    recorded = latest_recorded
                    break
                replacement = dict(specs)
                replacement.update(
                    artifact_specs(
                        "lustre",
                        latest.get("results_dir", ""),
                        backend_identity=current,
                    )
                )
                if self._store.compare_and_set_job_specs(
                    job_id,
                    specs,
                    replacement,
                    expected_statuses=("Pending", "Running", "Paused"),
                ):
                    recorded = current
                    logger.warning(
                        "Adopted current cluster identity for legacy SLURM job %s",
                        job_id,
                    )
                    break
        matches = bool(recorded) and recorded == current
        if not matches:
            logger.warning(
                "Refusing SLURM operation for job %s: backend identity is "
                "missing or does not match the current cluster",
                job_id,
            )
        return matches

    def _load_job_from_store(self, job_id: str):
        # Read and restore one snapshot under the same lifecycle lock used by
        # retry/cancel. Otherwise a cold loader can fetch revision N, wait for
        # a retry to persist N+1, then overwrite the handler with N.
        with self._handler._job_lifecycle_lock(job_id):  # noqa: SLF001
            entry = self._store.get_job(job_id)
            if not entry:
                return None
            if entry.get("backend_type") != "slurm":
                logger.warning(
                    "Refusing to load non-SLURM durable row %s as a SLURM job",
                    job_id,
                )
                return None
            runtime = self._runtime_from_entry(entry)
            if runtime is not None and not self._handler._restore_job_runtime_locked(  # noqa: SLF001
                job_id, runtime
            ):
                logger.error(
                    "Stored SLURM runtime identity is invalid for job %s",
                    job_id,
                )
                return None
            if runtime is None:
                logger.warning(
                    "SLURM job %s predates durable runtime metadata; falling back "
                    "to scheduler-name discovery",
                    job_id,
                )
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
