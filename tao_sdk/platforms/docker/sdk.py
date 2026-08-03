# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker (local) platform SDK for TAO.

Submits and monitors GPU training jobs as containers on the local Docker
daemon. Credentials and Docker connection settings come from environment
variables.

Usage::

    from tao_sdk.platforms.docker import DockerSDK
    sdk = DockerSDK()  # talks to local Docker (or $DOCKER_HOST)
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
from pathlib import Path

from tao_sdk.artifacts import (
    artifact_metadata,
    artifact_specs,
    delete_local_job_results,
    delete_remote_job_results,
    get_process_s3_credentials,
    remember_process_s3_credentials,
    require_s3_credential,
    require_terminal_job,
    s3_credential_fingerprint,
)
from tao_sdk.platforms.docker.handler import DockerHandler
from tao_sdk.job_store import JobStore, resolve_sdk_state_paths
from tao_sdk.models import Job, JobStatus
from tao_sdk.monitor import JobMonitor

logger = logging.getLogger(__name__)

_CREDENTIAL_ENV_MAP = {
    "ACCESS_KEY": "AWS_ACCESS_KEY_ID",
    "SECRET_KEY": "AWS_SECRET_ACCESS_KEY",
    "S3_ENDPOINT_URL": "AWS_ENDPOINT_URL",
    "S3_BUCKET_NAME": "S3_BUCKET_NAME",
    "HF_TOKEN": "HF_TOKEN",
    "NGC_KEY": "NGC_KEY",
}


class DockerSDK:
    """SDK for running jobs on a local Docker daemon.

    Reads connection settings + credentials from environment variables:
        - DOCKER_HOST (optional) — Docker socket / TCP endpoint
        - DOCKER_NETWORK (optional) — network name (default ``tao_default``)
        - DOCKER_PULL_POLICY (optional) — ``always``/``missing``/``never``
        - TAO_SDK_STATE_DIR (optional) — stable per-user state directory
        - NGC_KEY (for nvcr.io image pulls)
        - HF_TOKEN (passed to containers)
        - S3_BUCKET_NAME, ACCESS_KEY, SECRET_KEY, S3_ENDPOINT_URL (for S3 I/O)

    Local Docker is **single-node only** — ``num_nodes`` must be 1.
    """

    def __init__(
        self,
        poll_interval: int = 30,
        epoch_milestone_interval: int = 10,
        state_file: str | os.PathLike[str] | None = None,
    ):
        self._handler = DockerHandler.from_credentials(self._get_creds_from_env())

        state_file, db_path = resolve_sdk_state_paths(state_file)
        self._store = JobStore(db_path=db_path)
        self._monitor = JobMonitor(
            poll_interval=poll_interval,
            epoch_milestone_interval=epoch_milestone_interval,
            state_file=state_file,
            store=self._store,
        )
        self._monitor.start()
        self._jobs: dict[str, Job] = {}
        # Per-call S3 credentials are intentionally process-local. Persist only
        # their one-way principal fingerprint in JobStore.
        self._artifact_credentials: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _get_creds_from_env() -> dict:
        keys = (
            "DOCKER_HOST", "DOCKER_NETWORK", "DOCKER_USERNAME",
            "DOCKER_PULL_POLICY", "DOCKER_AUTO_REMOVE",
            "NGC_KEY", "ACCESS_KEY", "SECRET_KEY",
            "S3_ENDPOINT_URL", "S3_BUCKET_NAME", "CLOUD_REGION",
            "HF_TOKEN",
        )
        return {k: os.environ[k] for k in keys if k in os.environ}

    def _results_root_prefix(self, mounts: list | None = None) -> str | None:
        """``/results`` if a bind mount targets it, else ``None`` (script_runner
        falls back to ``s3://$S3_BUCKET_NAME`` or container-ephemeral)."""
        volumes = self._handler._build_volumes(mounts)  # noqa: SLF001
        for config in (volumes or {}).values():
            if (
                str(config.get("bind", "")).rstrip("/") == "/results"
                and self._handler._mount_mode_is_writable(  # noqa: SLF001
                    str(config.get("mode", "rw"))
                )
            ):
                return "/results"
        return None

    def _artifact_route(
        self,
        job_id: str,
        mounts: list | None = None,
        *,
        bucket: str | None = None,
    ) -> tuple[str, str]:
        """Return the durable artifact kind and authoritative job location."""
        volumes = self._handler._build_volumes(mounts)  # noqa: SLF001
        for source, config in (volumes or {}).items():
            target = str(config.get("bind", "")).rstrip("/")
            mode = str(config.get("mode", "rw"))
            if target != "/results" or not self._handler._mount_mode_is_writable(mode):  # noqa: SLF001
                continue
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                return (
                    "named_volume",
                    f"docker-volume://{source}/{job_id}",
                )
            if self._handler._docker_host_is_remote():  # noqa: SLF001
                return "remote_bind", f"docker-bind://{source_path}/{job_id}"
            root = source_path.resolve(strict=False)
            return "local_bind", str(root / job_id)

        if bucket is None:
            bucket = os.environ.get("S3_BUCKET_NAME", "")
        if bucket:
            return "s3", f"s3://{bucket}/results/{job_id}"
        return "ephemeral", ""

    def _local_results_dir(self, job_id: str, mounts: list | None = None) -> str:
        """Return the host path matching ``/results/<job_id>`` for local binds."""
        volumes = self._handler._build_volumes(mounts)  # noqa: SLF001
        if not self._handler._has_local_writable_results_bind(volumes):  # noqa: SLF001
            return ""
        for source, config in (volumes or {}).items():
            if str(config.get("bind", "")).rstrip("/") == "/results":
                return str(Path(source).expanduser().resolve(strict=False) / job_id)
        return ""

    def validate_artifact_retention(
        self,
        *,
        mounts: list | None = None,
        env_vars: dict | None = None,
        run_as_user: bool | None = None,
        container_user: str | int | None = None,
        **_kwargs,
    ) -> bool:
        """Fail before launch when automatic artifact deletion is unsafe."""
        if run_as_user is False:
            raise ValueError(
                "automatic checkpoint retention requires Docker jobs to run as "
                "the invoking host UID:GID; run_as_user=False is unsafe"
            )
        container_env = self._effective_container_env(env_vars)
        kind, _ = self._artifact_route(
            "retention-preflight",
            mounts,
            bucket=str(container_env.get("S3_BUCKET_NAME", "")),
        )
        if kind in {"named_volume", "remote_bind"}:
            raise ValueError(
                f"automatic checkpoint retention cannot safely delete Docker "
                f"{kind.replace('_', ' ')} results; use a local /results bind "
                "or explicitly authenticated S3 output"
            )
        if kind == "ephemeral":
            raise ValueError(
                "automatic checkpoint retention requires a durable reclaimable "
                "output route; configure a local /results bind or explicitly "
                "authenticated S3 output"
            )
        if kind == "local_bind":
            invoking_user = self._handler._invoking_user()  # noqa: SLF001
            if not invoking_user or invoking_user.split(":", 1)[0] == "0":
                raise RuntimeError(
                    "automatic checkpoint retention requires a non-root invoking "
                    "UID:GID for a local Docker /results bind"
                )
            if container_user is not None and str(container_user) != invoking_user:
                raise ValueError(
                    "container_user must exactly match the invoking host UID:GID "
                    f"({invoking_user}) for automatic checkpoint retention"
                )
            volumes = self._handler._build_volumes(mounts)  # noqa: SLF001
            resolved_user = self._handler._resolve_container_user(  # noqa: SLF001
                volumes=volumes,
                run_as_user=run_as_user,
                container_user=container_user,
            )
            if resolved_user != invoking_user:
                raise ValueError(
                    "Docker /results ownership is not bound to the invoking UID:GID"
                )
        if kind == "s3":
            self._require_explicit_s3_credentials(container_env)
        return True

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
        # Docker-specific:
        mounts: list | None = None,
        shm_size: int | str | None = None,
        run_as_user: bool | None = None,
        container_user: str | int | None = None,
    ) -> Job:
        """Submit a job to the local Docker daemon.

        The agent constructs ``command`` itself — typically by calling
        ``tao_sdk.script_runner.build_entrypoint(...)`` first to inline the
        in-container runtime + S3/HF/NGC download + spec rewriting + output
        upload.

        Args:
            image: Container image URI.
            command: Full shell command. Runs as-is inside the container —
                no SDK wrapping. Use ``build_entrypoint`` for S3 I/O.
            gpu_count: Number of GPUs (passed via ``--gpus``). Use ``-1`` for all.
            env_vars: Additional container env vars (S3/NGC/HF credentials
                auto-injected from os.environ).
            hooks: Callbacks: on_complete, on_error, on_epoch_milestone.
            mounts: Extra bind mounts (list of {host_path, container_path}).
            shm_size: ``--shm-size`` value (e.g. ``"8g"`` or bytes as int). When
                unset, Docker's default tmpfs on ``/dev/shm`` is used, which is
                too small for most TAO PyTorch DataLoader workloads.
            run_as_user: User-mapping policy. ``None`` (default) automatically
                runs local jobs with a writable ``/results`` bind mount as the
                invoking UID/GID. ``True`` requests invoking-user mapping for
                other mount layouts or jobs without a writable results bind.
                ``False`` preserves the image's configured user for images that
                require root. Remote Docker auto mode never assumes the SDK
                client host's IDs and rejects a writable ``/results`` bind
                until ``container_user`` or an explicit root opt-out is given.
            container_user: Explicit Docker user override, such as
                ``"1000:1000"``. This takes precedence over ``run_as_user`` and
                is required when a remote Docker host should use a non-root
                identity.
        """
        job_id = str(uuid.uuid4())

        # Build container env — inject S3/NGC/HF creds + script_runner contract.
        container_env = self._effective_container_env(env_vars)

        artifact_kind, results_dir = self._artifact_route(
            job_id,
            mounts,
            bucket=str(container_env.get("S3_BUCKET_NAME", "")),
        )
        local_results_dir = results_dir if artifact_kind == "local_bind" else ""
        artifact_root = (
            str(Path(results_dir).parent)
            if artifact_kind == "local_bind"
            else results_dir
        )
        backend_identity = self._handler.get_backend_identity()
        if not backend_identity:
            raise RuntimeError(
                "Docker daemon did not provide a stable identity; refusing to "
                "launch an unrecoverable job"
            )
        backend_job_id = self._handler.get_job_name(job_id)
        runtime_identity = {
            "container_name": backend_job_id,
            "container_id": "",
            "launch_uncertain": True,
        }
        stored_specs = artifact_specs(
            artifact_kind,
            artifact_root,
            endpoint_url=str(container_env.get("AWS_ENDPOINT_URL", "")),
            credential_fingerprint=(
                s3_credential_fingerprint(
                    str(container_env.get("AWS_ACCESS_KEY_ID", ""))
                )
                if artifact_kind == "s3" else ""
            ),
            backend_identity=backend_identity,
            runtime_identity=runtime_identity,
        )
        if artifact_kind == "s3":
            self._remember_s3_credentials(job_id, container_env)

        container_env["TAO_JOB_ID"] = job_id
        prefix = self._results_root_prefix(mounts)
        if prefix:
            container_env["TAO_RESULTS_ROOT"] = prefix

        cloud_metadata = self._build_cloud_metadata()

        # Agent's command runs as-is.
        entrypoint = {"command": command, "args_template": ""}

        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="local", backend_type="docker",
            backend_job_id=backend_job_id, image=image,
            results_dir=results_dir,
            local_results_dir=local_results_dir,
            specs=stored_specs,
        )

        launch_confirmed = False
        try:
            self._handler.create_job(
                image=image,
                network="",
                action="",
                cloud_metadata=cloud_metadata,
                specs=stored_specs,
                job_id=job_id,
                docker_env_vars=container_env,
                num_gpus=gpu_count,
                entrypoint=entrypoint,
                mounts=mounts,
                shm_size=shm_size,
                run_as_user=run_as_user,
                container_user=container_user,
            )
            launch_confirmed = True
        except Exception as exc:
            terminal_before_cleanup = self._observed_terminal_status(job_id)
            quiescent = bool(getattr(exc, "writer_quiescent", False))
            if not quiescent:
                try:
                    quiescent = self._handler.cancel_job(job_id)
                except Exception:
                    quiescent = False
            if quiescent:
                self._persist_runtime_identity(job_id)
                try:
                    terminal_status = (
                        terminal_before_cleanup
                        or self._terminal_status_after_quiescence(
                            job_id, fallback="Error"
                        )
                    )
                    self._store.update_job_status(job_id, terminal_status)
                except Exception:
                    logger.exception("Could not persist failed Docker launch status")
                raise
            logger.critical(
                "Docker launch for job %s failed ambiguously and cancellation "
                "could not be confirmed; returning its durable identity for recovery: %s",
                job_id,
                exc,
            )
        except BaseException as exc:
            terminal_before_cleanup = self._observed_terminal_status(job_id)
            quiescent = False
            try:
                quiescent = self._handler.cancel_job(job_id)
            except BaseException:
                logger.exception("Could not cancel interrupted Docker submission %s", job_id)
            if quiescent:
                self._persist_runtime_identity(job_id)
                try:
                    terminal_status = (
                        terminal_before_cleanup
                        or self._terminal_status_after_quiescence(
                            job_id, fallback="Canceled"
                        )
                    )
                    self._store.update_job_status(job_id, terminal_status)
                except Exception:
                    logger.exception("Could not persist interrupted Docker launch status")
            else:
                self._attach_ambiguous_launch_identity(
                    exc,
                    job_id=job_id,
                    backend_job_id=backend_job_id,
                )
            exc.tao_job_id = job_id
            exc.tao_launch_uncertain = not quiescent
            raise

        if launch_confirmed:
            self._persist_runtime_identity(job_id)

        job = Job(
            id=job_id,
            network_arch="",
            action="",
            workspace_id="local",
            backend_type="docker",
            backend_job_id=backend_job_id,
            results_dir=local_results_dir or results_dir,
            local_results_dir=local_results_dir,
        )
        self._jobs[job_id] = job

        if hooks:
            self._monitor.watch(job, hooks, self._handler)

        logger.info("Job %s submitted (container: %s, results: %s)",
                    job_id, backend_job_id, local_results_dir or results_dir)
        return job

    # ------------------------------------------------------------------
    # Job monitoring
    # ------------------------------------------------------------------

    def get_job_status(self, job_id: str) -> JobStatus:
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job:
            return JobStatus(job_id=job_id, status="Unknown", message="Job not found")
        if not self._backend_identity_matches(job_id):
            return JobStatus(
                job_id=job_id,
                status="Unknown",
                message="Job belongs to a different or unverified Docker daemon",
            )
        entry = self._store.get_job(job_id)
        if (
            job.status in {"Complete", "Error", "Canceled"}
            and entry
            and entry.get("status") == job.status
        ):
            return JobStatus(
                job_id=job_id,
                status=job.status,
                message=f"Terminal status {job.status} was confirmed",
            )
        raw = self._handler.get_tao_job_status(job_id)
        self._persist_runtime_identity(job_id)
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
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job or not self._backend_identity_matches(job_id):
            return False
        current = self.get_job_status(job_id)
        if current.status in {"Complete", "Error", "Canceled"}:
            self._monitor.unwatch(job_id)
            return True
        success = self._handler.cancel_job(job_id)
        self._persist_runtime_identity(job_id)
        if success:
            desired = self._terminal_status_after_quiescence(
                job_id, fallback="Canceled"
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
        if not entry:
            return ""
        return entry.get("local_results_dir") or entry.get("results_dir", "")

    def delete_job_artifacts(self, job_id: str) -> bool:
        """Delete the exact local bind or S3 result prefix owned by ``job_id``."""
        entry = self._store.get_job(job_id)
        if not entry:
            return False
        try:
            require_terminal_job(entry)
            metadata = artifact_metadata(entry)
            if not self._backend_identity_matches(job_id, entry=entry):
                raise ValueError(
                    "job belongs to a different or unverified Docker daemon"
                )
            kind = metadata.get("kind", "")
            root = metadata.get("root", "")
            local_results_dir = entry.get("local_results_dir", "")
            deleted = False
            if kind == "local_bind" and local_results_dir:
                local_root = Path(root).expanduser().absolute()
                expected = local_root / job_id
                runtime_root = local_root / ".tao-runtime"
                runtime_job_dir = runtime_root / job_id
                self._remove_terminal_container_for_artifact_cleanup(
                    job_id, entry, metadata
                )
                try:
                    results_deleted = delete_local_job_results(
                        local_results_dir, job_id, expected_path=expected
                    )
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "Could not delete local results for job %s: %s", job_id, exc
                    )
                    results_deleted = False
                try:
                    runtime_deleted = delete_local_job_results(
                        runtime_job_dir,
                        job_id,
                        expected_path=runtime_job_dir,
                    )
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "Could not delete local runtime cache for job %s: %s",
                        job_id,
                        exc,
                    )
                    runtime_deleted = False
                deleted = results_deleted and runtime_deleted
                if runtime_deleted:
                    try:
                        runtime_root.rmdir()
                    except OSError:
                        # Other jobs (or legacy shared cache data) still use it.
                        pass
            results_dir = entry.get("results_dir", "")
            if kind == "s3" and results_dir == root:
                access_key, secret_key = self._s3_credentials_for_job(
                    job_id, metadata
                )
                require_s3_credential(metadata, access_key)
                fs = self._get_fs(
                    metadata.get("endpoint_url", ""),
                    access_key=access_key,
                    secret_key=secret_key,
                )
                self._remove_terminal_container_for_artifact_cleanup(
                    job_id, entry, metadata
                )
                deleted = delete_remote_job_results(
                    fs,
                    results_dir,
                    job_id,
                    "s3",
                )
            if deleted:
                self._forget_s3_credentials(job_id)
                self._store.delete_checkpoints(job_id)
                return True
            if kind in {"remote_bind", "named_volume"}:
                raise ValueError(
                    f"{kind.replace('_', ' ')} artifacts cannot be safely "
                    "deleted from the SDK host"
                )
        except Exception as exc:
            logger.warning("Could not delete artifacts for job %s: %s", job_id, exc)
            return False
        logger.warning("No persistent artifact directory is recorded for job %s", job_id)
        return False

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
        if Path(results).is_absolute():
            root = Path(results)
            if not root.is_dir():
                return []
            exts = (".pth", ".tlt", ".hdf5", ".pt", ".ckpt", ".engine", ".onnx", ".trt")
            return [
                str(path)
                for path in root.rglob("*")
                if path.is_file() and path.name.lower().endswith(exts)
            ]
        if results.startswith(("docker-bind://", "docker-volume://")):
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

    def _terminal_status_after_quiescence(
        self,
        job_id: str,
        *,
        fallback: str,
    ) -> str:
        """Preserve a backend-observed terminal outcome after cleanup."""
        return self._observed_terminal_status(job_id) or fallback

    def _observed_terminal_status(self, job_id: str) -> str | None:
        """Return a backend-observed terminal outcome without inventing one."""
        try:
            observed = self._handler.normalize_status(
                self._handler.get_tao_job_status(job_id)
            )
        except Exception as exc:
            logger.warning(
                "Could not observe terminal Docker status for job %s during "
                "launch cleanup: %s",
                job_id,
                exc,
            )
            return None
        return (
            observed
            if observed in {"Complete", "Error", "Canceled"}
            else None
        )

    def _persist_runtime_identity(self, job_id: str) -> None:
        """Sync positively reconciled Docker runtime state into JobStore."""
        try:
            runtime = self._handler.get_job_runtime_identity(job_id)
            entry = self._store.get_job(job_id)
            if not runtime or not entry:
                return
            specs = entry.get("specs", {})
            if isinstance(specs, str):
                specs = json.loads(specs)
            if not isinstance(specs, dict):
                return
            specs = dict(specs)
            metadata = specs.get("_tao_artifacts")
            if not isinstance(metadata, dict):
                return
            metadata = dict(metadata)
            if metadata.get("runtime_identity") == runtime:
                return
            metadata["runtime_identity"] = runtime
            specs["_tao_artifacts"] = metadata
            if not self._store.update_job_specs(job_id, specs):
                logger.warning("Docker job %s disappeared during runtime sync", job_id)
        except Exception:
            logger.exception("Could not persist reconciled Docker runtime for %s", job_id)

    @staticmethod
    def _attach_ambiguous_launch_identity(
        exc: BaseException,
        *,
        job_id: str,
        backend_job_id: str,
    ) -> None:
        """Attach durable reconciliation coordinates to an interrupted launch."""
        runtime_identity = {
            "container_name": backend_job_id,
            "container_id": "",
            "launch_uncertain": True,
        }
        recovery_identity = {
            "job_id": job_id,
            "backend_job_id": backend_job_id,
            **runtime_identity,
        }
        for name, value in (
            ("tao_job_id", job_id),
            ("job_id", job_id),
            ("backend_job_id", backend_job_id),
            ("launch_uncertain", True),
            ("runtime_identity", runtime_identity),
            ("durable_job_identity", recovery_identity),
        ):
            try:
                setattr(exc, name, value)
            except Exception:
                logger.error(
                    "Could not attach %s to interrupted Docker launch exception",
                    name,
                )

    def _backend_identity_matches(
        self,
        job_id: str,
        *,
        entry: dict | None = None,
    ) -> bool:
        """Require durable proof that this SDK is using the job's daemon."""
        entry = entry or self._store.get_job(job_id)
        recorded = str(artifact_metadata(entry).get("backend_identity", ""))
        current = self._handler.get_backend_identity()
        matches = bool(recorded) and recorded == current
        if not matches:
            logger.warning(
                "Refusing Docker operation for job %s: backend identity is "
                "missing or does not match the current daemon",
                job_id,
            )
        return matches

    def _remove_terminal_container_for_artifact_cleanup(
        self,
        job_id: str,
        entry: dict,
        metadata: dict,
    ) -> None:
        """Remove the exact stopped container before reporting cleanup success."""
        runtime = metadata.get("runtime_identity")
        if not self._handler.restore_job_runtime(
            job_id,
            runtime,
            status=entry.get("status", "Unknown"),
        ):
            raise ValueError(
                "recorded Docker container identity is missing or invalid"
            )
        observed = self._handler.normalize_status(
            self._handler.get_tao_job_status(job_id)
        )
        if observed not in {"Complete", "Error", "Canceled"}:
            raise ValueError(
                f"Docker container is not observably terminal (status: {observed})"
            )
        if not self._handler.delete_job(job_id):
            raise RuntimeError(
                "the exact terminal Docker container could not be removed"
            )

    def _load_job_from_store(self, job_id: str):
        entry = self._store.get_job(job_id)
        if not entry:
            return None
        if entry.get("backend_type") != "docker":
            logger.warning(
                "Refusing to load non-Docker durable row %s as a Docker job",
                job_id,
            )
            return None
        runtime = artifact_metadata(entry).get("runtime_identity")
        if not self._handler.restore_job_runtime(
            job_id,
            runtime,
            status=entry.get("status", "Unknown"),
        ):
            logger.warning("Docker runtime identity is missing for job %s", job_id)
        job = Job(
            id=entry["job_id"],
            network_arch=entry.get("network_arch", ""),
            action=entry.get("action", ""),
            workspace_id=entry.get("workspace_id", "local"),
            backend_type="docker",
            status=entry.get("status", "Unknown"),
            created_at=entry.get("created_at", ""),
            backend_job_id=entry.get("backend_job_id", ""),
            results_dir=(
                entry.get("local_results_dir") or entry.get("results_dir", "")
            ),
            local_results_dir=entry.get("local_results_dir", ""),
        )
        self._jobs[job_id] = job
        return job

    @staticmethod
    def _current_s3_credentials() -> tuple[str, str]:
        return (
            os.environ.get("ACCESS_KEY")
            or os.environ.get("AWS_ACCESS_KEY_ID", ""),
            os.environ.get("SECRET_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )

    @staticmethod
    def _effective_container_env(env_vars: dict | None) -> dict:
        container_env = dict(env_vars or {})
        for credential_key, environment_key in _CREDENTIAL_ENV_MAP.items():
            value = os.environ.get(credential_key, "")
            if value and environment_key not in container_env:
                container_env[environment_key] = value
        return container_env

    @staticmethod
    def _require_explicit_s3_credentials(container_env: dict) -> None:
        if not str(container_env.get("AWS_ACCESS_KEY_ID", "")):
            raise ValueError(
                "automatic S3 checkpoint retention requires an explicit "
                "AWS_ACCESS_KEY_ID; ambient IAM/provider-chain credentials cannot "
                "be bound to the job for safe cleanup"
            )
        if not str(container_env.get("AWS_SECRET_ACCESS_KEY", "")):
            raise ValueError(
                "automatic S3 checkpoint retention requires an explicit "
                "AWS_SECRET_ACCESS_KEY for the recorded job principal"
            )
        remember_process_s3_credentials(
            str(container_env.get("AWS_ACCESS_KEY_ID", "")),
            str(container_env.get("AWS_SECRET_ACCESS_KEY", "")),
            str(container_env.get("AWS_ENDPOINT_URL", "")),
        )

    def _remember_s3_credentials(self, job_id: str, container_env: dict) -> None:
        access_key = str(container_env.get("AWS_ACCESS_KEY_ID", ""))
        secret_key = str(container_env.get("AWS_SECRET_ACCESS_KEY", ""))
        credentials = getattr(self, "_artifact_credentials", None)
        if credentials is None:
            credentials = self._artifact_credentials = {}
        credentials[job_id] = (access_key, secret_key)
        remember_process_s3_credentials(
            access_key,
            secret_key,
            str(container_env.get("AWS_ENDPOINT_URL", "")),
        )

    def _s3_credentials_for_job(
        self, job_id: str, metadata: dict
    ) -> tuple[str, str]:
        credentials = getattr(self, "_artifact_credentials", {})
        recorded = str(metadata.get("credential_fingerprint", ""))
        current = self._current_s3_credentials()
        process_local = get_process_s3_credentials(
            recorded, str(metadata.get("endpoint_url", ""))
        )
        for access_key, secret_key in (
            credentials.get(job_id, ("", "")),
            current,
            process_local,
        ):
            if s3_credential_fingerprint(access_key) == recorded and secret_key:
                return access_key, secret_key
        return current

    def _forget_s3_credentials(self, job_id: str) -> None:
        credentials = getattr(self, "_artifact_credentials", None)
        if credentials is not None:
            credentials.pop(job_id, None)

    def _get_fs(
        self,
        endpoint_url: str | None = None,
        *,
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        import fsspec
        storage_opts = {}
        ambient_access_key, ambient_secret_key = self._current_s3_credentials()
        ak = ambient_access_key if access_key is None else access_key
        sk = ambient_secret_key if secret_key is None else secret_key
        ep = endpoint_url if endpoint_url is not None else os.environ.get("S3_ENDPOINT_URL", "")
        if ak:
            storage_opts["key"] = ak
        if sk:
            storage_opts["secret"] = sk
        if endpoint_url is not None:
            storage_opts["config_kwargs"] = {
                "ignore_configured_endpoint_urls": True
            }
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
