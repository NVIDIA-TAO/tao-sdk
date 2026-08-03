# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes platform SDK for TAO.

Submits and monitors GPU training jobs as Kubernetes ``Jobs`` (single-pod) on
any reachable cluster. Authenticates via kubeconfig (laptop) or in-cluster
service account (when running inside a pod).

Usage::

    from tao_sdk.platforms.kubernetes import KubernetesSDK
    sdk = KubernetesSDK()  # uses ~/.kube/config or in-cluster auth
    job = sdk.create_job(
        image='nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt',
        command='visual_changenet train -e /tmp/spec.yaml',
        gpu_count=1,
        inputs={'/data/train.csv': 's3://bucket/data/train.csv'},
        outputs=['/results/'],
    )

Requires the NVIDIA GPU Operator on the cluster for ``nvidia.com/gpu``
scheduling — install per
https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/.
"""

from __future__ import annotations

import json
import logging
import os
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
from tao_sdk.platforms.kubernetes.handler import KubernetesHandler
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


class KubernetesSDK:
    """SDK for running jobs as Kubernetes Jobs.

    Reads optional config from environment variables:
        - TAO_K8S_NAMESPACE — default ``default``
        - TAO_K8S_CONTEXT — kubeconfig context name
        - KUBECONFIG — alternate kubeconfig path (else ``~/.kube/config``)
        - TAO_SDK_STATE_DIR — optional stable per-user state directory
        - NGC_KEY (for nvcr.io image pulls), HF_TOKEN (passed to containers)
        - S3_BUCKET_NAME, ACCESS_KEY, SECRET_KEY, S3_ENDPOINT_URL (for S3 I/O)

    Single-pod jobs only. For multi-node, extend with Volcano / MPI Operator.
    """

    def __init__(
        self,
        poll_interval: int = 30,
        epoch_milestone_interval: int = 10,
        state_file: str | os.PathLike[str] | None = None,
    ):
        self._handler = KubernetesHandler.from_credentials(self._get_creds_from_env())

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
        self._artifact_credentials: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _get_creds_from_env() -> dict:
        keys = (
            "TAO_K8S_NAMESPACE", "TAO_K8S_CONTEXT", "KUBECONFIG",
            "NGC_KEY", "ACCESS_KEY", "SECRET_KEY",
            "S3_ENDPOINT_URL", "S3_BUCKET_NAME", "CLOUD_REGION",
            "HF_TOKEN",
        )
        return {k: os.environ[k] for k in keys if k in os.environ}

    def _results_root_prefix(
        self,
        volume_mounts: list | None = None,
    ) -> str | None:
        """``/results`` if a volume mount targets it (PVC/hostPath/etc.),
        else ``None`` (script_runner falls back to S3 or container-ephemeral)."""
        for vm in volume_mounts or []:
            mount_path = (vm.get("mountPath") if isinstance(vm, dict) else getattr(vm, "mount_path", None)) or ""
            if mount_path.rstrip("/") == "/results":
                return "/results"
        return None

    def validate_artifact_retention(
        self,
        *,
        env_vars: dict | None = None,
        volume_mounts: list | None = None,
        mounts: list | None = None,
        **_kwargs,
    ) -> bool:
        """Fail before launch when Kubernetes outputs cannot be reclaimed."""
        mounted_results = volume_mounts if volume_mounts is not None else mounts
        if self._results_root_prefix(mounted_results):
            raise ValueError(
                "automatic checkpoint retention cannot safely identify and "
                "delete one job's Kubernetes /results volume; use explicitly "
                "authenticated S3 output"
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
                "Could not inspect quiescent Kubernetes job %s: %s; using %s",
                job_id,
                exc,
                fallback,
            )
        try:
            self._store.update_job_status(job_id, desired)
        except Exception:
            logger.exception("Could not persist %s Kubernetes launch status", context)

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
        # Kubernetes-specific:
        num_nodes: int = 1,
        namespace: str | None = None,
        node_selector: dict | None = None,
        tolerations: list | None = None,
        image_pull_secret: str | None = None,
    ) -> Job:
        """Submit a job to Kubernetes.

        Single-pod by default. For ``num_nodes > 1``, submits an Indexed Job
        + headless Service for PyTorch-distributed rendezvous (requires
        k8s 1.28+ for stable pod hostnames).

        The agent constructs ``command`` itself — typically by calling
        ``tao_sdk.script_runner.build_entrypoint(...)`` first to inline the
        in-container runtime + S3/HF/NGC download + spec rewriting + output
        upload.

        Args:
            image: Container image URI.
            command: Full shell command. Runs as-is inside the container —
                no SDK wrapping. Use ``build_entrypoint`` for S3 I/O.
            gpu_count: Number of GPUs **per node** (mapped to
                ``nvidia.com/gpu`` resource). The handler raises
                ``RuntimeError`` if the cluster has fewer allocatable GPUs
                (typically because the NVIDIA GPU Operator isn't installed).
            env_vars: Additional container env vars (S3/NGC/HF credentials
                auto-injected from os.environ). For multi-node, the SDK
                also injects ``NODE_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``,
                ``MASTER_PORT`` so PyTorch-distributed init works without
                user setup.
            hooks: Callbacks: on_complete, on_error, on_epoch_milestone.
            num_nodes: Multi-node distributed training. Default 1. When
                ``> 1``, submits an Indexed Job + headless Service.
            namespace: Override the default namespace.
            node_selector: Pod nodeSelector for targeting GPU nodes.
            tolerations: Pod tolerations (list of V1Toleration-shaped dicts).
            image_pull_secret: Name of an existing image-pull secret in the
                namespace (for nvcr.io/private registries).
        """
        job_id = str(uuid.uuid4())

        # Build container env — inject S3/NGC/HF creds + script_runner contract.
        container_env = self._effective_container_env(env_vars)

        bucket = str(container_env.get("S3_BUCKET_NAME", ""))
        results_dir = f"s3://{bucket}/results/{job_id}" if bucket else ""
        backend_identity = self._handler.get_backend_identity()
        if not backend_identity:
            raise RuntimeError(
                "Kubernetes cluster did not provide a stable identity; refusing "
                "to launch an unrecoverable job"
            )
        runtime_identity = self._handler.get_runtime_identity(
            job_id,
            namespace=namespace,
            num_nodes=num_nodes,
            launch_uncertain=True,
        )
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
            backend_identity=backend_identity,
            runtime_identity=runtime_identity,
        )
        if results_dir:
            self._remember_s3_credentials(job_id, container_env)

        container_env["TAO_JOB_ID"] = job_id
        # K8s create_job doesn't yet expose volume_mounts; when added, pass
        # through to _results_root_prefix(...) here.
        prefix = self._results_root_prefix()
        if prefix:
            container_env["TAO_RESULTS_ROOT"] = prefix

        cloud_metadata = self._build_cloud_metadata()

        # Agent's command runs as-is.
        entrypoint = {"command": command, "args_template": ""}
        backend_job_id = self._handler.get_job_name(job_id)

        self._store.create_job(
            job_id=job_id, network_arch="", action="",
            workspace_id="kubernetes", backend_type="kubernetes",
            backend_job_id=backend_job_id, image=image,
            results_dir=results_dir,
            specs=stored_specs,
        )

        launch_confirmed = False
        try:
            backend_response = self._handler.create_job(
                image=image,
                network="",
                action="",
                cloud_metadata=cloud_metadata,
                specs=stored_specs,
                job_id=job_id,
                docker_env_vars=container_env,
                num_gpus=gpu_count,
                num_nodes=num_nodes,
                entrypoint=entrypoint,
                namespace=namespace,
                node_selector=node_selector,
                tolerations=tolerations,
                image_pull_secret=image_pull_secret,
            )
            launch_confirmed = True
        except Exception as exc:
            self._handler.restore_job_runtime(job_id, runtime_identity)
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
                "Kubernetes launch for job %s failed ambiguously and cancellation "
                "could not be confirmed; returning its durable identity for recovery: %s",
                job_id,
                exc,
            )
            backend_response = {}
        except BaseException as exc:
            self._handler.restore_job_runtime(job_id, runtime_identity)
            try:
                quiescent = self._handler.cancel_job(job_id)
            except BaseException:
                logger.exception(
                    "Could not cancel interrupted Kubernetes submission %s", job_id
                )
                quiescent = False
            exc.tao_job_id = job_id
            exc.tao_launch_uncertain = not quiescent
            if quiescent:
                self._persist_quiescent_launch_status(
                    job_id, "Canceled", "interrupted"
                )
            raise

        runtime_identity["job_uid"] = str(backend_response.get("uid", "") or "")
        runtime_identity["service_uid"] = str(
            backend_response.get("service_uid", "") or ""
        )
        runtime_identity["launch_uncertain"] = not launch_confirmed
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
            backend_identity=backend_identity,
            runtime_identity=runtime_identity,
        )

        job = Job(
            id=job_id,
            network_arch="",
            action="",
            workspace_id="kubernetes",
            backend_type="kubernetes",
            backend_job_id=backend_job_id,
            results_dir=results_dir,
        )
        self._jobs[job_id] = job

        try:
            self._store.update_job_specs(job_id, stored_specs)
        except Exception as exc:
            logger.error(
                "Could not enrich durable Kubernetes runtime identity for %s; "
                "the pre-registered name/namespace remain recoverable: %s",
                job_id,
                exc,
            )

        if hooks:
            self._monitor.watch(job, hooks, self._handler)

        logger.info("Job %s submitted (k8s: %s, results: %s)",
                    job_id, backend_job_id, results_dir)
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
                message="Job belongs to a different or unverified Kubernetes cluster",
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
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job or not self._backend_identity_matches(job_id):
            return None
        return self._handler.get_failure_analysis(job_id)

    def get_job_replicas(self, job_id: str) -> list[dict]:
        job = self._jobs.get(job_id) or self._load_job_from_store(job_id)
        if not job or not self._backend_identity_matches(job_id):
            return []
        return self._handler.get_job_replicas(job_id)

    def get_job_results_dir(self, job_id: str) -> str:
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
            if not self._backend_identity_matches(job_id, entry=entry):
                raise ValueError(
                    "job belongs to a different or unverified Kubernetes cluster"
                )
            results_dir = entry.get("results_dir", "")
            if metadata.get("kind") != "s3" or metadata.get("root") != results_dir:
                raise ValueError("recorded S3 artifact identity is missing or mismatched")
            access_key, secret_key = self._s3_credentials_for_job(job_id, metadata)
            require_s3_credential(metadata, access_key)
            fs = self._get_fs(
                metadata.get("endpoint_url", ""),
                access_key=access_key,
                secret_key=secret_key,
            )
            self._remove_terminal_job_for_artifact_cleanup(
                job_id, metadata
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

    def _backend_identity_matches(
        self,
        job_id: str,
        *,
        entry: dict | None = None,
    ) -> bool:
        """Require durable proof that this SDK is using the job's cluster."""
        entry = entry or self._store.get_job(job_id)
        recorded = str(artifact_metadata(entry).get("backend_identity", ""))
        current = self._handler.get_backend_identity()
        matches = bool(recorded) and recorded == current
        if not matches:
            logger.warning(
                "Refusing Kubernetes operation for job %s: backend identity is "
                "missing or does not match the current cluster",
                job_id,
            )
        return matches

    def _remove_terminal_job_for_artifact_cleanup(
        self,
        job_id: str,
        metadata: dict,
    ) -> None:
        """Remove the exact terminal Job, pods, and Service before S3 data."""
        runtime = metadata.get("runtime_identity")
        if (
            not isinstance(runtime, dict)
            or not str(runtime.get("job_uid", "") or "")
            or not self._handler.restore_job_runtime(job_id, runtime)
        ):
            raise ValueError(
                "recorded Kubernetes Job UID and runtime identity are missing "
                "or invalid"
            )
        if not self._handler.delete_terminal_job(job_id):
            raise RuntimeError(
                "the exact terminal Kubernetes Job, pods, and Service could "
                "not be removed"
            )

    def _persist_runtime_identity(self, job_id: str) -> None:
        """Persist a positively reconciled UID/launch state without replacing the row."""
        runtime = self._handler.get_job_runtime_identity(job_id)
        entry = self._store.get_job(job_id)
        if not runtime or not entry:
            return
        try:
            specs = entry.get("specs", {})
            if isinstance(specs, str):
                specs = json.loads(specs)
            if not isinstance(specs, dict):
                return
            metadata = specs.get("_tao_artifacts")
            if not isinstance(metadata, dict):
                return
            if metadata.get("runtime_identity") == runtime:
                return
            metadata["runtime_identity"] = runtime
            if not self._store.update_job_specs(job_id, specs):
                logger.warning("Kubernetes job %s disappeared during runtime sync", job_id)
        except Exception:
            logger.exception(
                "Could not persist reconciled Kubernetes runtime for %s", job_id
            )

    def _load_job_from_store(self, job_id: str):
        entry = self._store.get_job(job_id)
        if not entry:
            return None
        if entry.get("backend_type") != "kubernetes":
            logger.warning(
                "Refusing to load non-Kubernetes durable row %s as a Kubernetes job",
                job_id,
            )
            return None
        runtime = artifact_metadata(entry).get("runtime_identity")
        if not self._handler.restore_job_runtime(job_id, runtime):
            logger.warning("Kubernetes runtime identity is missing for job %s", job_id)
        job = Job(
            id=entry["job_id"],
            network_arch=entry.get("network_arch", ""),
            action=entry.get("action", ""),
            workspace_id=entry.get("workspace_id", "kubernetes"),
            backend_type="kubernetes",
            status=entry.get("status", "Unknown"),
            created_at=entry.get("created_at", ""),
            backend_job_id=entry.get("backend_job_id", ""),
            results_dir=entry.get("results_dir", ""),
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
                "AWS_ACCESS_KEY_ID; ambient Kubernetes workload credentials "
                "cannot be bound to the job for safe cleanup"
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
