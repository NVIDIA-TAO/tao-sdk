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
import shlex
import uuid

from tao_sdk.artifacts import (
    artifact_metadata,
    artifact_specs,
    delete_remote_job_results,
    get_process_s3_credentials,
    remember_process_s3_credentials,
    require_s3_credential,
    require_terminal_job,
    s3_credential_fingerprint,
)
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


class BrevSDK:
    """SDK for running jobs on Brev GPU instances.

    Reads credentials from environment variables:
        - BREV_API_TOKEN (optional — falls back to manual brev login)
        - TAO_SDK_STATE_DIR (optional) — stable per-user state directory
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
        state_file: str | os.PathLike[str] | None = None,
    ):
        # Brev doesn't strictly require a token (manual login works)
        api_token = os.environ.get("BREV_API_TOKEN", "")

        from tao_sdk.platforms.brev.handler import BrevHandler
        from tao_sdk.platforms.brev.cli import BrevCLI
        self._handler = BrevHandler(cli=BrevCLI(api_token=api_token))

        state_file, db_path = resolve_sdk_state_paths(state_file)
        self._store = JobStore(db_path=db_path)
        self._monitor = JobMonitor(
            poll_interval=poll_interval,
            state_file=state_file,
            store=self._store,
        )
        self._monitor.start()
        self._jobs: dict[str, Job] = {}
        self._artifact_credentials: dict[str, tuple[str, str]] = {}

    def _results_root_prefix(self, mounts: list | None = None) -> str | None:
        """Brev shells out to ``docker run``; same rules as the local Docker SDK.
        ``/results`` if a bind mount targets it, else ``None``."""
        for m in mounts or []:
            if isinstance(m, dict) and m.get("container_path", "").rstrip("/") == "/results":
                return "/results"
        return None

    def validate_artifact_retention(
        self,
        *,
        env_vars: dict | None = None,
        mounts: list | None = None,
        **_kwargs,
    ) -> bool:
        """Require a reproducible S3 principal for automatic retention."""
        if self._results_root_prefix(mounts):
            raise ValueError(
                "automatic checkpoint retention cannot safely delete a Brev "
                "/results bind; use explicitly authenticated S3 output"
            )
        container_env = self._effective_container_env(env_vars)
        if not str(container_env.get("S3_BUCKET_NAME", "")):
            raise ValueError(
                "automatic checkpoint retention requires durable reclaimable "
                "output; configure explicitly authenticated S3 output"
            )
        self._require_explicit_s3_credentials(container_env)
        return True

    def _persist_quiescent_launch_status(
        self, job_id: str, fallback: str, context: str
    ) -> None:
        """Persist a natural terminal result observed during launch cleanup."""
        desired = fallback
        try:
            observed = self._handler.normalize_status(
                self._handler.get_tao_job_status(job_id)
            )
            if observed in {"Complete", "Error", "Canceled"}:
                desired = observed
        except Exception as exc:
            logger.warning(
                "Could not inspect quiescent Brev job %s: %s; using %s",
                job_id,
                exc,
                fallback,
            )
        try:
            self._store.update_job_status(job_id, desired)
        except Exception:
            logger.exception("Could not persist %s Brev launch status", context)

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

        # Build container env vars + script_runner contract.
        container_env = self._effective_container_env(env_vars)

        # Persist the effective per-job output route, including supported
        # caller overrides, rather than re-reading mutable host environment at
        # cleanup time.
        bucket = str(container_env.get("S3_BUCKET_NAME", ""))
        results_dir = f"s3://{bucket}/results/{job_id}" if bucket else ""
        stored_specs = artifact_specs(
            "s3" if results_dir else "ephemeral",
            results_dir,
            endpoint_url=str(container_env.get("AWS_ENDPOINT_URL", "")),
            credential_fingerprint=(
                s3_credential_fingerprint(
                    str(container_env.get("AWS_ACCESS_KEY_ID", ""))
                )
                if results_dir else ""
            ),
        )
        if results_dir:
            self._remember_s3_credentials(job_id, container_env)

        container_env["TAO_JOB_ID"] = job_id
        # Brev's create_job doesn't yet expose mounts; when added, pass through.
        prefix = self._results_root_prefix()
        if prefix:
            container_env["TAO_RESULTS_ROOT"] = prefix

        # Build the full docker run command for the Brev instance —
        # agent's command runs as-is inside the container.
        env_flags = " ".join(
            f"-e {shlex.quote(f'{key}={value}')}"
            for key, value in container_env.items()
        )
        container_name = self._handler.get_container_name(job_id)
        cid_path = self._handler.get_container_id_path(job_id)
        if not isinstance(cid_path, str) or not cid_path:
            cid_path = f"/tmp/tao-job-{job_id}.cid"
        docker_invocation = (
            f"docker run --gpus all --rm --name {shlex.quote(container_name)} "
            f"--cidfile {shlex.quote(cid_path)} "
            f"--label {shlex.quote(f'tao-sdk-job-id={job_id}')} "
            "--label tao-sdk-backend=brev "
            f"{env_flags} {shlex.quote(image)} /bin/bash -c {shlex.quote(command)}"
        )
        exit_path = self._handler.get_exit_status_path(job_id)
        if not isinstance(exit_path, str) or not exit_path:
            exit_path = f"/tmp/tao-job-{job_id}.exit"
        quoted_exit_path = shlex.quote(exit_path)
        docker_command = (
            "set +e; "
            f"{docker_invocation}; "
            "tao_exit_code=$?; "
            f"tao_exit_tmp={quoted_exit_path}.tmp.$$; "
            "printf '%s\\n' \"$tao_exit_code\" > \"$tao_exit_tmp\"; "
            f"mv -f -- \"$tao_exit_tmp\" {quoted_exit_path}; "
            "exit \"$tao_exit_code\""
        )

        # Build entrypoint for handler
        entrypoint = {"command": docker_command, "args_template": ""}

        # Store job
        resolved_instance_id = instance_id or f"tao-{job_id[:8]}"
        backend_job_id = self._handler.get_job_name(job_id)
        stored_specs["_brev_runtime"] = {
            "instance_id": resolved_instance_id,
            "pid": "",
            "launch_uncertain": True,
        }
        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="", backend_type="brev",
            backend_job_id=backend_job_id, image=image,
            results_dir=results_dir,
            specs=stored_specs,
        )

        # Submit to handler
        try:
            backend_details = self._handler.create_job(
                image=image,
                network="",
                action="",
                cloud_metadata={},
                specs=stored_specs,
                job_id=job_id,
                docker_env_vars=container_env,
                resource_shape=None,
                dedicated_node_group=None,
                num_nodes=1,
                entrypoint=entrypoint,
                instance_id=resolved_instance_id,
                gpu_type=gpu_type or "",
                cloud_cred_id=cloud_cred_id or "",
                workspace_group_id=workspace_group_id or "",
            )
        except Exception as exc:
            self._handler._job_instances[job_id] = resolved_instance_id  # noqa: SLF001
            quiescent = bool(getattr(exc, "writer_quiescent", False))
            if not quiescent:
                try:
                    quiescent = self._handler.cancel_job(job_id)
                except Exception:
                    quiescent = False
            if quiescent:
                self._persist_quiescent_launch_status(job_id, "Error", "failed")
                raise
            logger.critical(
                "Brev launch for job %s failed ambiguously and cancellation could "
                "not be confirmed; returning its durable identity for recovery: %s",
                job_id,
                exc,
            )
            backend_details = {
                "instance_id": resolved_instance_id,
                "pid": "",
                "launch_uncertain": True,
            }
        except BaseException as exc:
            self._handler._job_instances[job_id] = resolved_instance_id  # noqa: SLF001
            try:
                quiescent = self._handler.cancel_job(job_id)
            except BaseException:
                logger.exception("Could not cancel interrupted Brev submission %s", job_id)
                quiescent = False
            exc.tao_job_id = job_id
            exc.tao_launch_uncertain = not quiescent
            if quiescent:
                self._persist_quiescent_launch_status(
                    job_id, "Canceled", "interrupted"
                )
            raise

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

        runtime = {
            "instance_id": str(backend_details.get("instance_id", "")),
            "pid": str(backend_details.get("pid", "")),
            "launch_uncertain": bool(
                backend_details.get("launch_uncertain", False)
            ),
        }
        container_id = str(backend_details.get("container_id", "")).lower()
        if len(container_id) == 64 and all(
            character in "0123456789abcdef" for character in container_id
        ):
            runtime["container_id"] = container_id
        stored_specs["_brev_runtime"] = runtime
        try:
            self._store.update_job_specs(job_id, stored_specs)
        except Exception as exc:
            logger.error(
                "Could not enrich durable Brev PID for %s; the pre-registered "
                "instance/container identity remains recoverable: %s",
                job_id,
                exc,
            )

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
        current = self.get_job_status(job_id)
        if current.status in {"Complete", "Error", "Canceled"}:
            self._monitor.unwatch(job_id)
            return True
        success = self._handler.cancel_job(job_id)
        if success:
            self._persist_runtime_identity(job_id)
            try:
                observed = self._handler.normalize_status(
                    self._handler.get_tao_job_status(job_id)
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
        """Not supported on Brev — returns None."""
        return None

    def get_job_results_dir(self, job_id: str) -> str:
        """Get the S3 results directory for a job."""
        entry = self._store.get_job(job_id)
        return entry.get("results_dir", "") if entry else ""

    def delete_job_artifacts(self, job_id: str) -> bool:
        """Delete the exact S3 result prefix owned by ``job_id``."""
        entry = self._store.get_job(job_id)
        if not entry:
            return False
        try:
            require_terminal_job(entry)
            metadata = artifact_metadata(entry)
            results_dir = entry.get("results_dir", "")
            if metadata.get("kind") != "s3" or metadata.get("root") != results_dir:
                raise ValueError("recorded S3 artifact identity is missing or mismatched")
            access_key, secret_key = self._s3_credentials_for_job(job_id, metadata)
            require_s3_credential(metadata, access_key)
            deleted = delete_remote_job_results(
                self._get_fs(
                    metadata.get("endpoint_url", ""),
                    access_key=access_key,
                    secret_key=secret_key,
                ),
                results_dir,
                job_id,
                "s3",
            )
            if deleted:
                self._forget_s3_credentials(job_id)
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
        if entry.get("backend_type") != "brev":
            logger.warning(
                "Refusing to load non-Brev durable row %s as a Brev job",
                job_id,
            )
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
        try:
            specs = json.loads(entry.get("specs", "{}"))
        except (TypeError, ValueError):
            specs = {}
        runtime = specs.get("_brev_runtime", {}) if isinstance(specs, dict) else {}
        if not self._handler.restore_job_runtime(job_id, runtime):
            logger.warning("Brev runtime identity is missing for job %s", job_id)
        self._jobs[job_id] = job
        return job

    def _persist_runtime_identity(self, job_id: str) -> None:
        runtime = self._handler.get_job_runtime_identity(job_id)
        entry = self._store.get_job(job_id)
        if not runtime or not entry:
            return
        try:
            specs = entry.get("specs", {})
            if isinstance(specs, str):
                specs = json.loads(specs)
            if not isinstance(specs, dict) or specs.get("_brev_runtime") == runtime:
                return
            specs["_brev_runtime"] = runtime
            if not self._store.update_job_specs(job_id, specs):
                logger.warning("Brev job %s disappeared during runtime sync", job_id)
        except Exception:
            logger.exception("Could not persist reconciled Brev runtime for %s", job_id)

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
                "AWS_ACCESS_KEY_ID; ambient Brev instance credentials cannot be "
                "bound to the job for safe cleanup"
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
        credentials = getattr(self, "_artifact_credentials", None)
        if credentials is None:
            credentials = self._artifact_credentials = {}
        credentials[job_id] = (
            str(container_env.get("AWS_ACCESS_KEY_ID", "")),
            str(container_env.get("AWS_SECRET_ACCESS_KEY", "")),
        )
        remember_process_s3_credentials(
            *credentials[job_id],
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
