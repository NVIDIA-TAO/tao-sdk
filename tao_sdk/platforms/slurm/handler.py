# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core-aligned SLURM execution handler for TAO SDK.

The SDK does not run the TAO Core microservice container handler or MongoDB
metadata layer, but the SLURM mechanics here intentionally mirror TAO Core's
``microservices/handlers/execution_handlers/slurm_handler.py``:

* SSH to one of the configured login hosts with failover.
* Stage compact specs/env/cloud metadata files on Lustre.
* Stage an SDK entrypoint shell script and sbatch file remotely.
* Optionally convert Docker images to cached SQSH files with enroot.
* Launch with Pyxis ``srun --container-image`` and Core-like sbatch flags.
* Query ``squeue``/``sacct``, stream remote logs, cancel with ``scancel``.
* Retry failures that look like infrastructure/node/GPU issues.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import pwd
import re
import shlex
import subprocess
import tempfile
import threading
import time
import traceback
import fcntl
from uuid import uuid4 as _uuid4
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from tao_sdk.artifacts import validate_job_id
from tao_sdk.platforms.base import BaseHandler

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 5 * 60
MAX_JOB_RETRIES = 10
STATE_PATH = Path(os.path.expanduser("~/.tao_sdk_slurm_jobs.json"))
DEFAULT_JOB_DIR_TEMPLATE = "/lustre/fsw/portfolios/edgeai/users/{user}"
DEFAULT_PARTITION = "polar,polar3,polar4,grizzly"
DEFAULT_ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"
TERMINAL_SLURM_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "REVOKED",
    "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL",
}

RETRIABLE_ERROR_PATTERNS = [
    r"NVIDIA driver.*too old",
    r"CUDA driver version is insufficient",
    r"no CUDA-capable device",
    r"CUDA initialization.*error",
    r"NCCL.*error",
    r"GPU.*not found",
    r"RuntimeError.*CUDA",
    r"cuInit.*failed",
    r"torch\.cuda\.is_available.*False",
    r"Unable to determine the device handle for GPU",
    r"GPU.*fallen off the bus",
    r"Xid.*error",
    r"ECC.*error",
    r"GPU.*temperature",
    r"InfiniBand.*error",
    r"RDMA.*error",
]


class SlurmHandler(BaseHandler):
    """Execution handler for remote SLURM clusters accessed over SSH."""

    def __init__(
        self,
        slurm_user: str,
        slurm_hostnames: list[str],
        creds: dict[str, Any] | None = None,
        ssh_key_path: str | None = None,
        base_results_dir: str | None = None,
        partition: str | None = None,
        account: str | None = None,
        container_mounts: str | None = None,
    ):
        if not slurm_user:
            raise ValueError("SLURM_USER or SSH_USER is required for the slurm backend")
        if not slurm_hostnames:
            raise ValueError("SLURM_HOSTNAME or SSH_HOST is required for the slurm backend")

        self.login_user = slurm_user
        self.login_hostnames = slurm_hostnames
        self.current_hostname_index = 0
        self.login_hostname = self.login_hostnames[self.current_hostname_index]
        self._backend_identity_cache = ""
        self._backend_identity_cache_hostname_index: int | None = None

        self._creds = creds or {}
        self.ssh_key_path = ssh_key_path or self._find_ssh_key()
        self.ssh_options = self._get_ssh_options()

        self.base_results_dir = (
            base_results_dir
            or DEFAULT_JOB_DIR_TEMPLATE.format(user=self.login_user)
        ).rstrip("/")
        base_path = PurePosixPath(self.base_results_dir)
        if (
            not base_path.is_absolute()
            or str(base_path) == "/"
            or ".." in base_path.parts
        ):
            raise ValueError(
                "SLURM_BASE_RESULTS_DIR must be an absolute, scoped path "
                "without parent traversal"
            )
        self.partition = partition or DEFAULT_PARTITION
        self.account = account or ""
        self.container_mounts = container_mounts or "/lustre"

        self.time_hours = self._float_config("SLURM_TIME_HOURS", 4.0)
        self.timeout_hours = self._float_config(
            "SLURM_TIMEOUT_HOURS",
            max(self.time_hours - 0.2, 0.1),
        )
        self.cpus_per_task = self._int_config("SLURM_CPUS_PER_TASK", 16)
        self.max_num_gpus = self._int_config("SLURM_MAX_GPUS_PER_NODE", 8)
        self.use_timeout = self._bool_config("SLURM_USE_TIMEOUT", True)
        self.use_requeue = self._bool_config("SLURM_USE_REQUEUE", True)
        self.use_srun_logs = self._bool_config("SLURM_USE_SRUN_LOGS", False)
        self.use_sqsh = self._bool_config("SLURM_USE_SQSH", True)
        self.sqsh_cache_dir = (
            self._config_value("SLURM_SQSH_CACHE_DIR")
            or self.base_results_dir
        ).rstrip("/")
        self.conversion_partition = (
            self._config_value("SLURM_CONVERSION_PARTITION") or "cpu"
        )
        self.conversion_timeout_minutes = self._int_config(
            "SLURM_CONVERSION_TIMEOUT_MINUTES",
            30,
        )
        self.conversion_memory_gb = self._int_config("SLURM_CONVERSION_MEMORY_GB", 32)
        self.force_reconvert_latest = self._bool_config(
            "SLURM_FORCE_RECONVERT_LATEST",
            False,
        )
        self.mail_user = self._config_value("SLURM_MAIL_USER")

        self._job_messages: dict[str, str] = {}
        self._runtime_state_callback = None
        self._lifecycle_locks: dict[str, threading.RLock] = {}
        self._lifecycle_locks_guard = threading.Lock()
        self._jobs = self._load_state()

        logger.info(
            "SlurmHandler initialized for %s@%s with %d hostname(s)",
            self.login_user,
            self.login_hostname,
            len(self.login_hostnames),
        )

    @classmethod
    def from_credentials(cls, creds: dict) -> "SlurmHandler":
        """Construct a handler from SDK credentials and environment."""
        creds = creds or {}
        user = (
            creds.get("SLURM_USER")
            or creds.get("SSH_USER")
            or os.environ.get("SLURM_USER")
            or os.environ.get("SSH_USER")
        )
        host_value = (
            creds.get("SLURM_HOSTNAME")
            or creds.get("SSH_HOST")
            or os.environ.get("SLURM_HOSTNAME")
            or os.environ.get("SSH_HOST")
        )
        if isinstance(host_value, str):
            hostnames = [h.strip() for h in host_value.split(",") if h.strip()]
        else:
            hostnames = list(host_value or [])

        missing = []
        if not user:
            missing.append("SLURM_USER")
        if not hostnames:
            missing.append("SLURM_HOSTNAME")
        if missing:
            from tao_sdk.models import CredentialError
            raise CredentialError(
                f"Missing required SLURM credentials: {', '.join(missing)}\n"
                "Set these as environment variables (e.g., in ~/.config/tao/.env)."
            )

        return cls(
            slurm_user=user,
            slurm_hostnames=hostnames,
            creds=creds,
            ssh_key_path=creds.get("SSH_KEY_PATH") or os.environ.get("SSH_KEY_PATH"),
            base_results_dir=(
                creds.get("SLURM_BASE_RESULTS_DIR")
                or os.environ.get("SLURM_BASE_RESULTS_DIR")
            ),
            partition=creds.get("SLURM_PARTITION") or os.environ.get("SLURM_PARTITION"),
            account=creds.get("SLURM_ACCOUNT") or os.environ.get("SLURM_ACCOUNT"),
            container_mounts=(
                creds.get("SLURM_CONTAINER_MOUNTS")
                or os.environ.get("SLURM_CONTAINER_MOUNTS")
            ),
        )

    def get_job_name(self, job_id: str) -> str:
        """Generate the SLURM job name."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", job_id)
        return f"tao-job-{safe}"[:120]

    def set_runtime_state_callback(self, callback) -> None:
        """Set the SDK callback used to durably mirror scheduler identity."""
        self._runtime_state_callback = callback

    def _job_lifecycle_lock(self, job_id: str) -> threading.RLock:
        """Return the in-process lock serializing one job's side effects."""
        guard = getattr(self, "_lifecycle_locks_guard", None)
        if guard is None:
            guard = threading.Lock()
            self._lifecycle_locks_guard = guard
        with guard:
            locks = getattr(self, "_lifecycle_locks", None)
            if locks is None:
                locks = {}
                self._lifecycle_locks = locks
            return locks.setdefault(job_id, threading.RLock())

    def _default_runtime_identity(self, job_id: str) -> dict[str, Any]:
        """Build deterministic recovery coordinates before ``sbatch``."""
        job_id = validate_job_id(job_id)
        job_dir = self.base_results_dir
        return {
            "slurm_job_id": "",
            "failed_slurm_job_ids": [],
            "retry_count": 0,
            "job_name": self.get_job_name(job_id),
            "job_dir": job_dir,
            "log_dir": f"{job_dir}/slurm-logs",
            "remote_script": f"{job_dir}/sbatch/job_{job_id}.sbatch",
            "remote_entrypoint": f"{job_dir}/entrypoints/job_{job_id}.sh",
            "remote_specs": f"{job_dir}/specs/{job_id}.json",
            "remote_env": f"{job_dir}/env/{job_id}.json",
            "remote_meta": f"{job_dir}/meta/{job_id}.json",
            "partition": self.partition,
            "account": self.account,
            "hostname": self.login_hostname,
            "submission_attempted": False,
            "launch_uncertain": False,
            "launch_token": "",
            "pre_launch_slurm_job_id": "",
            "revision": 0,
            "message": "Preparing SLURM execution",
        }

    def get_job_runtime_identity(self, job_id: str) -> dict[str, Any]:
        """Return scheduler coordinates suitable for durable JobStore specs."""
        metadata = self._jobs.get(job_id) or self._default_runtime_identity(job_id)
        return copy.deepcopy(metadata)

    def restore_job_runtime(self, job_id: str, runtime: dict | None) -> bool:
        """Restore runtime while serializing it with scheduler side effects."""
        with self._job_lifecycle_lock(job_id):
            return self._restore_job_runtime_locked(job_id, runtime)

    def _restore_job_runtime_locked(self, job_id: str, runtime: dict | None) -> bool:
        """Restore validated scheduler coordinates from the durable JobStore."""
        if not isinstance(runtime, dict):
            return False
        defaults = self._default_runtime_identity(job_id)
        immutable_keys = (
            "job_name",
            "job_dir",
            "log_dir",
            "remote_script",
            "remote_entrypoint",
            "remote_specs",
            "remote_env",
            "remote_meta",
        )
        if any(
            key in runtime and str(runtime[key]) != str(defaults[key])
            for key in immutable_keys
        ):
            logger.error("Refusing invalid durable SLURM runtime for %s", job_id)
            return False

        slurm_job_id = str(runtime.get("slurm_job_id", "") or "")
        if slurm_job_id and not slurm_job_id.isdigit():
            return False
        failed_ids = runtime.get("failed_slurm_job_ids", [])
        if not isinstance(failed_ids, list) or any(
            not str(item).isdigit() for item in failed_ids
        ):
            return False
        try:
            retry_count = int(runtime.get("retry_count", 0))
        except (TypeError, ValueError):
            return False
        if retry_count < 0:
            return False

        launch_token = str(runtime.get("launch_token", "") or "")
        pre_launch_slurm_job_id = str(
            runtime.get("pre_launch_slurm_job_id", "") or ""
        )
        if pre_launch_slurm_job_id and not pre_launch_slurm_job_id.isdigit():
            return False
        if bool(runtime.get("launch_uncertain", False)) and not launch_token:
            return False
        try:
            revision = int(runtime.get("revision", 0))
        except (TypeError, ValueError):
            return False
        if revision < 0:
            return False

        hostname = str(runtime.get("hostname") or defaults["hostname"])
        configured_hosts = {
            self._normalize_hostname(host) for host in self.login_hostnames
        }
        if self._normalize_hostname(hostname) not in configured_hosts:
            # Login nodes are replaceable transport coordinates. Cluster
            # identity is independently verified from ClusterName + user.
            hostname = self.login_hostname

        restored = {**defaults, **copy.deepcopy(runtime)}
        restored["slurm_job_id"] = slurm_job_id
        restored["failed_slurm_job_ids"] = [str(item) for item in failed_ids]
        restored["retry_count"] = retry_count
        restored["launch_token"] = launch_token
        restored["pre_launch_slurm_job_id"] = pre_launch_slurm_job_id
        restored["revision"] = revision
        restored["hostname"] = hostname
        restored["submission_attempted"] = bool(
            restored.get("submission_attempted", False)
        )
        restored["launch_uncertain"] = bool(
            restored.get("launch_uncertain", False)
        )

        current = self._jobs.get(job_id)
        if isinstance(current, dict):
            try:
                current_revision = int(current.get("revision", 0))
            except (TypeError, ValueError):
                current_revision = -1
            if current_revision > revision:
                logger.info(
                    "Ignoring stale SLURM runtime revision %d for %s; "
                    "in-memory revision is %d",
                    revision,
                    job_id,
                    current_revision,
                )
                return True
            if current_revision == revision:
                generation_keys = (
                    "slurm_job_id",
                    "failed_slurm_job_ids",
                    "retry_count",
                    "submission_attempted",
                    "launch_uncertain",
                    "launch_token",
                    "pre_launch_slurm_job_id",
                )
                if any(
                    current.get(key) != restored.get(key)
                    for key in generation_keys
                ):
                    logger.warning(
                        "Ignoring conflicting SLURM runtime at revision %d for %s",
                        revision,
                        job_id,
                    )
                    return True
        self._jobs[job_id] = restored
        return True

    def _persist_runtime_state(self, job_id: str, *, required: bool = False) -> None:
        """Mirror one runtime to JobStore while retaining the legacy sidecar."""
        self._save_state()
        callback = getattr(self, "_runtime_state_callback", None)
        if callback is None:
            return
        try:
            persisted = callback(job_id, self.get_job_runtime_identity(job_id))
            if isinstance(persisted, dict):
                self._jobs[job_id] = copy.deepcopy(persisted)
                self._save_state()
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"Could not durably persist SLURM runtime for {job_id}"
                ) from exc
            logger.warning("Could not mirror SLURM runtime for %s: %s", job_id, exc)

    def _begin_launch_fence(self, job_id: str) -> dict[str, Any]:
        """Atomically claim one scheduler submission generation."""
        previous = self.get_job_runtime_identity(job_id)
        candidate = copy.deepcopy(previous)
        candidate["submission_attempted"] = True
        candidate["launch_uncertain"] = True
        candidate["launch_token"] = _uuid4().hex
        candidate["pre_launch_slurm_job_id"] = str(
            previous.get("slurm_job_id", "") or ""
        )
        self._jobs[job_id] = candidate
        try:
            self._persist_runtime_state(job_id, required=True)
        except Exception:
            self._jobs[job_id] = previous
            raise
        return self._jobs[job_id]

    def _confirm_launch_fence(
        self,
        job_id: str,
        launch_token: str,
    ) -> dict[str, Any]:
        """Revalidate durable ownership immediately before scheduler submit."""
        runtime = self.get_job_runtime_identity(job_id)
        if (
            not runtime.get("launch_uncertain")
            or not launch_token
            or str(runtime.get("launch_token", "")) != launch_token
        ):
            raise RuntimeError("SLURM launch fence ownership changed before sbatch")
        self._persist_runtime_state(job_id, required=True)
        persisted = self.get_job_runtime_identity(job_id)
        if (
            not persisted.get("launch_uncertain")
            or str(persisted.get("launch_token", "")) != launch_token
        ):
            raise RuntimeError("SLURM launch fence ownership was lost before sbatch")
        return persisted

    def get_backend_identity(self) -> str:
        """Return a stable non-secret identity for the configured cluster.

        The identity (ClusterName + login user) is stable for the life of a
        handler on a given login host, but this is called on every monitored
        operation — an SSH round-trip per poll, and on a flaky network every
        poll degrades into a 30s timeout plus a refused operation. Cache after
        the first successful fetch; the cache is keyed on the login-hostname
        index so failover never serves an identity fetched from a different
        host, and an empty cache still fails safe to "".
        """
        # getattr: recovery paths and tests build handlers via __new__,
        # skipping __init__.
        cached = getattr(self, "_backend_identity_cache", "")
        cached_index = getattr(self, "_backend_identity_cache_hostname_index", None)
        current_index = getattr(self, "current_hostname_index", 0)
        if cached and cached_index == current_index:
            return cached
        try:
            result = self._ssh("scontrol show config", check=False, timeout=30)
        except Exception as exc:
            logger.warning("Could not verify SLURM cluster identity: %s", exc)
            return ""
        match = re.search(
            r"(?m)^\s*ClusterName\s*=\s*(\S+)\s*$",
            result.stdout or "",
        )
        if result.returncode != 0 or not match:
            logger.warning("SLURM did not report a stable ClusterName")
            return ""
        target = json.dumps(
            {
                "cluster_name": match.group(1),
                "user": self.login_user,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        identity = f"slurm-sha256:{digest}"
        self._backend_identity_cache = identity
        self._backend_identity_cache_hostname_index = current_index
        return identity

    def resolve_resource_shape(
        self,
        num_gpus: int,
        preferred_node_group: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Carry the requested GPU count through the SDK resource-shape slot."""
        return f"gpu.{max(int(num_gpus or 1), 0)}", preferred_node_group

    def resolve_cpu_shape(
        self,
        preferred_node_group: str | None = None,
        preference: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Represent CPU-only requests without forcing cloud-shape semantics."""
        return "gpu.0", preferred_node_group

    def get_mount_objects(
        self,
        volumes: list[dict] | None = None,
        path_prefix: str = "/",
    ) -> tuple[str | None, list | None]:
        """SDK SLURM jobs use direct S3 IO unless explicit mounts are provided."""
        return None, None

    def create_job(
        self,
        image: str,
        network: str,
        action: str,
        cloud_metadata: dict | None = None,
        specs: dict | None = None,
        job_id: str = "",
        docker_env_vars: dict | None = None,
        resource_shape: str | None = None,
        dedicated_node_group: str | None = None,
        num_nodes: int = 1,
        entrypoint: dict | None = None,
        mounts: list | str | None = None,
        **kwargs,
    ) -> str:
        """Stage and submit a job while serializing its lifecycle side effects."""
        if not job_id:
            raise ValueError("job_id is required")
        with self._job_lifecycle_lock(job_id):
            return self._create_job_locked(
                image=image,
                network=network,
                action=action,
                cloud_metadata=cloud_metadata,
                specs=specs,
                job_id=job_id,
                docker_env_vars=docker_env_vars,
                resource_shape=resource_shape,
                dedicated_node_group=dedicated_node_group,
                num_nodes=num_nodes,
                entrypoint=entrypoint,
                mounts=mounts,
                **kwargs,
            )

    def _create_job_locked(
        self,
        image: str,
        network: str,
        action: str,
        cloud_metadata: dict | None = None,
        specs: dict | None = None,
        job_id: str = "",
        docker_env_vars: dict | None = None,
        resource_shape: str | None = None,
        dedicated_node_group: str | None = None,
        num_nodes: int = 1,
        entrypoint: dict | None = None,
        mounts: list | str | None = None,
        **kwargs,
    ) -> str:
        """Stage and submit a Pyxis/Enroot-backed SLURM job."""
        if not entrypoint or "command" not in entrypoint:
            raise ValueError("entrypoint with 'command' key is required")
        if not job_id:
            raise ValueError("job_id is required")

        specs = dict(specs or {})
        docker_env_vars = dict(docker_env_vars or {})
        cloud_metadata = dict(cloud_metadata or {})

        self._convert_slurm_urls(specs)
        self._convert_slurm_urls(docker_env_vars)

        job_name = self.get_job_name(job_id)
        job_dir = self.base_results_dir
        log_dir = f"{job_dir}/slurm-logs"
        account = self.account or self._detect_default_account() or DEFAULT_ACCOUNT
        partition = kwargs.get("partition") or dedicated_node_group or self.partition
        allowed_nodes = kwargs.get("allowed_nodes")

        num_gpus = self._num_gpus_from_shape(resource_shape)
        num_nodes = max(int(num_nodes or 1), 1)
        exclusive = False
        if num_gpus >= self.max_num_gpus:
            computed_nodes = math.ceil(num_gpus / self.max_num_gpus)
            if computed_nodes > num_nodes:
                num_nodes = computed_nodes
            num_gpus = self.max_num_gpus
            exclusive = True

        if not specs.get("results_dir"):
            specs["results_dir"] = f"{job_dir}/results/{job_id}"
        results_base_dir = (
            specs["results_dir"].rsplit("/", 1)[0]
            if "/" in specs["results_dir"]
            else specs["results_dir"]
        )
        docker_env_vars["TAO_API_RESULTS_DIR"] = results_base_dir
        docker_env_vars.setdefault("HF_HOME", f"{job_dir}/.cache/huggingface")
        docker_env_vars["PYTHONNOUSERSITE"] = "1"
        docker_env_vars["PYTHONDONTWRITEBYTECODE"] = "1"
        if cloud_metadata:
            docker_env_vars["CLOUD_METADATA"] = self._to_compact_json(cloud_metadata)
        if self._creds.get("CLOUD_REGION") and "AWS_DEFAULT_REGION" not in docker_env_vars:
            docker_env_vars["AWS_DEFAULT_REGION"] = self._creds["CLOUD_REGION"]

        if network == "cosmos-rl":
            docker_env_vars.setdefault("COSMOS_CACHE", f"{job_dir}/.cache/cosmos")
            docker_env_vars.setdefault("CUDA_OVERRIDE_VERSION", "12.8")

        if mounts is not None:
            kwargs = {**kwargs, "mounts": mounts}
        container_mounts = self._container_mounts_from_kwargs(kwargs)
        slurm_image = image
        remote_specs = f"{job_dir}/specs/{job_id}.json"
        remote_env = f"{job_dir}/env/{job_id}.json"
        remote_meta = f"{job_dir}/meta/{job_id}.json"
        remote_entrypoint = f"{job_dir}/entrypoints/job_{job_id}.sh"
        remote_script = f"{job_dir}/sbatch/job_{job_id}.sbatch"

        runtime_identity = self._default_runtime_identity(job_id)
        runtime_identity.update({
            "partition": partition,
            "allowed_nodes": allowed_nodes,
            "account": account,
            "hostname": self.login_hostname,
            "submission_attempted": False,
            "launch_uncertain": False,
            "message": "Preparing container image for SLURM execution",
        })
        self._jobs[job_id] = runtime_identity
        self._persist_runtime_state(job_id, required=True)

        if self.use_sqsh:
            self._set_job_message(
                job_id,
                "Preparing container image for SLURM execution",
            )
            slurm_image = self._prepare_container_image(
                image=image,
                sqsh_cache_dir=self.sqsh_cache_dir,
                conversion_partition=self.conversion_partition,
                conversion_account=account,
                conversion_timeout_minutes=self.conversion_timeout_minutes,
                conversion_memory_gb=self.conversion_memory_gb,
                force_reconvert_latest=self.force_reconvert_latest,
                job_id=job_id,
            )

        specs_json = self._to_compact_json(specs)
        env_json = self._to_compact_json(docker_env_vars)
        meta_json = self._to_compact_json(cloud_metadata)

        entrypoint_script = self._build_entrypoint_script(entrypoint["command"])
        self._scp_text(specs_json, remote_specs)
        self._scp_text(env_json, remote_env)
        self._scp_text(meta_json, remote_meta)
        self._scp_text(entrypoint_script, remote_entrypoint)

        sdk_command = f"/bin/bash {shlex.quote(remote_entrypoint)}"
        slurm_script = self._build_slurm_script(
            job_name=job_name,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            cpus_per_task=self.cpus_per_task,
            time_hours=self.time_hours,
            account=account,
            partition=partition,
            log_dir=log_dir,
            mail_user=self.mail_user,
            image=slurm_image,
            container_mounts=container_mounts,
            use_timeout=self.use_timeout,
            timeout_hours=self.timeout_hours,
            use_requeue=self.use_requeue,
            use_srun=self.use_srun_logs,
            command=sdk_command,
            docker_env_vars=docker_env_vars,
            exclusive=exclusive,
            network=network,
            allowed_nodes=allowed_nodes,
        )
        self._scp_text(slurm_script, remote_script)

        try:
            fenced_runtime = self._begin_launch_fence(job_id)
            launch_token = str(fenced_runtime.get("launch_token", ""))
            fenced_runtime = self._confirm_launch_fence(job_id, launch_token)
        except Exception as exc:
            error = RuntimeError(
                "Could not confirm the durable SLURM launch fence before sbatch"
            )
            error.writer_quiescent = True
            raise error from exc
        result = self._ssh(
            f"sbatch --export=ALL {shlex.quote(remote_script)}",
            check=False,
            timeout=TIMEOUT_SECONDS,
            allow_failover=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            self._set_job_message(job_id, f"SLURM sbatch failed: {detail}")
            error = RuntimeError(f"sbatch failed for {remote_script}: {detail}")
            if not self._is_connection_failure(result):
                # A scheduler-side rejection (bad account/partition/spec) did
                # not create a writer. Transport failures remain ambiguous.
                self._jobs[job_id]["submission_attempted"] = False
                self._jobs[job_id]["launch_uncertain"] = False
                try:
                    self._persist_runtime_state(job_id, required=True)
                except Exception:
                    logger.exception(
                        "Could not persist definitive SLURM scheduler rejection"
                    )
                error.writer_quiescent = True
            raise error

        sbatch_output = result.stdout.strip()
        slurm_job_id = self.get_slurm_job_id_from_output(sbatch_output)
        if not slurm_job_id:
            raise RuntimeError(
                f"Could not parse SLURM job id from sbatch output: {sbatch_output!r}"
            )

        submitted_runtime = copy.deepcopy(fenced_runtime)
        submitted_runtime.update({
            "slurm_job_id": slurm_job_id,
            "job_name": job_name,
            "job_dir": job_dir,
            "log_dir": log_dir,
            "remote_script": remote_script,
            "remote_entrypoint": remote_entrypoint,
            "remote_specs": remote_specs,
            "remote_env": remote_env,
            "remote_meta": remote_meta,
            "image": slurm_image,
            "partition": partition,
            "allowed_nodes": allowed_nodes,
            "account": account,
            "hostname": self.login_hostname,
            "submission_attempted": True,
            "launch_uncertain": False,
            "message": (
                f"Job submitted to SLURM cluster "
                f"(Job ID: {slurm_job_id}). Waiting for resources..."
            ),
        })
        self._jobs[job_id] = submitted_runtime
        try:
            self._persist_runtime_state(job_id, required=True)
        except Exception as exc:
            quiescent = False
            try:
                quiescent = self.cancel_job(job_id)
            except Exception:
                logger.exception(
                    "Could not cancel SLURM job %s after runtime persistence failed",
                    slurm_job_id,
                )
            error = RuntimeError(
                f"SLURM accepted job {slurm_job_id}, but its scheduler identity "
                "could not be persisted"
            )
            error.writer_quiescent = quiescent
            raise error from exc
        self._set_job_message(job_id, self._jobs[job_id]["message"])
        logger.info("Submitted SLURM job %s for TAO job %s", slurm_job_id, job_id)
        return sbatch_output

    def get_tao_job_status(
        self,
        job_id: str,
        *,
        allow_retry: bool = True,
    ) -> str | None:
        """Get status while serializing retries and reconciliation per job."""
        with self._job_lifecycle_lock(job_id):
            return self._get_tao_job_status_locked(
                job_id,
                allow_retry=allow_retry,
            )

    def _get_tao_job_status_locked(
        self,
        job_id: str,
        *,
        allow_retry: bool = True,
    ) -> str | None:
        """Get TAO SDK status by checking SLURM state."""
        metadata = self._get_job_metadata(job_id)
        if not metadata:
            self._set_job_message(
                job_id,
                "SLURM job metadata not found; submission may still be in progress",
            )
            return "Pending"
        if not self._select_recorded_hostname(metadata, job_id):
            return "Pending"

        slurm_job_id = metadata.get("slurm_job_id")
        if metadata.get("launch_uncertain"):
            candidates = self._launch_reconciliation_candidates(metadata, job_id)
            if len(candidates) != 1:
                detail = (
                    "not visible yet"
                    if not candidates
                    else (
                        "ambiguous across scheduler ids "
                        f"{[item[0] for item in candidates]}"
                    )
                )
                self._set_job_message(
                    job_id,
                    f"SLURM launch is fenced and {detail}",
                )
                return "Pending"
            slurm_job_id, _ = candidates[0]
            if allow_retry:
                metadata["slurm_job_id"] = slurm_job_id
                metadata["launch_uncertain"] = False
                if metadata.get("pre_launch_slurm_job_id"):
                    failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
                    pre_launch_id = str(metadata["pre_launch_slurm_job_id"])
                    if pre_launch_id not in failed_ids:
                        failed_ids.append(pre_launch_id)
                    metadata["retry_count"] = (
                        int(metadata.get("retry_count", 0)) + 1
                    )
                self._jobs[job_id] = metadata
                self._persist_runtime_state(job_id, required=True)
        if not slurm_job_id:
            discovered = self._find_slurm_job_by_name(
                metadata.get("job_name") or self.get_job_name(job_id)
            )
            if not discovered:
                self._set_job_message(job_id, "SLURM job id not recorded yet")
                return "Pending"
            slurm_job_id, _ = discovered
            if allow_retry:
                metadata["slurm_job_id"] = slurm_job_id
                self._jobs[job_id] = metadata
                self._persist_runtime_state(job_id, required=True)

        slurm_status = self.get_slurm_job_status(slurm_job_id)
        self._set_job_message(job_id, f"SLURM job {slurm_job_id}: {slurm_status}")

        if slurm_status in ("PENDING", "CONFIGURING", "RESIZING"):
            return "Pending"
        if slurm_status in ("RUNNING", "COMPLETING"):
            return "Running"
        if slurm_status == "COMPLETED":
            return None
        if slurm_status in ("CANCELLED", "PREEMPTED", "REVOKED"):
            return "Canceled"
        if slurm_status in ("SUSPENDED", "STOPPED"):
            return "Paused"
        if slurm_status in ("FAILED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "NODE_FAIL"):
            retry_count = int(metadata.get("retry_count", 0))
            if (
                allow_retry
                and retry_count < MAX_JOB_RETRIES
                and self._is_retriable_failure(slurm_status, job_id)
            ):
                success, new_slurm_job_id, error_msg = self._resubmit_failed_job(
                    job_id,
                    failed_slurm_job_id=slurm_job_id,
                    retry_count=retry_count,
                )
                if success:
                    self._set_job_message(
                        job_id,
                        (
                            f"Job resubmitted after infrastructure failure "
                            f"(attempt {retry_count + 1}/{MAX_JOB_RETRIES}); "
                            f"new SLURM Job ID: {new_slurm_job_id}"
                        ),
                    )
                    return "Pending"
                self._set_job_message(
                    job_id,
                    f"SLURM retry failed after {slurm_status}: {error_msg}",
                )
            elif allow_retry:
                failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
                if slurm_job_id not in failed_ids:
                    failed_ids.append(slurm_job_id)
                    self._persist_runtime_state(job_id, required=True)
            return "Error"
        if slurm_status == "TIMEOUT":
            self._set_job_message(job_id, f"SLURM job {slurm_job_id} hit time limit")
            return "Error"
        if slurm_status == "ERROR":
            self._set_job_message(
                job_id,
                "Unable to check SLURM job status; will retry on next poll",
            )
            return "Pending"
        if slurm_status == "NOT_FOUND":
            self._set_job_message(
                job_id,
                f"SLURM job {slurm_job_id} not found in queue/history yet",
            )
            return "Pending"

        logger.warning("Unknown SLURM status %s for job %s", slurm_status, job_id)
        return "Pending"

    def normalize_status(self, raw_status: str | None) -> str:
        """Map raw backend status to SDK status strings."""
        return "Complete" if raw_status is None else raw_status

    def get_job_message(self, job_id: str) -> str:
        metadata = self._get_job_metadata(job_id)
        return self._job_messages.get(job_id) or (metadata or {}).get("message", "")

    def get_job_logs(self, job_id: str, tail_lines: int | None = None) -> str | None:
        """Read main stdout/stderr files from the remote SLURM log directory."""
        metadata = self._get_job_metadata(job_id)
        if not metadata:
            return None
        if not self._select_recorded_hostname(metadata, job_id):
            return None

        slurm_job_id = metadata.get("slurm_job_id")
        job_name = metadata.get("job_name") or self.get_job_name(job_id)
        log_dir = metadata.get("log_dir") or f"{self.base_results_dir}/slurm-logs"
        if not slurm_job_id:
            return None

        log_path = f"{log_dir}/{job_name}-{slurm_job_id}"
        logs: list[str] = []
        for filename in ("main.out", "main.err"):
            content = self._read_remote_log_file(f"{log_path}/{filename}", tail_lines)
            if content:
                logs.append(content)
        return "\n".join(logs) if logs else None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job while serializing scheduler lifecycle side effects."""
        with self._job_lifecycle_lock(job_id):
            return self._cancel_job_locked(job_id)

    def _cancel_job_locked(self, job_id: str) -> bool:
        """Cancel every scheduler candidate owned by this durable TAO job."""
        metadata = self._get_job_metadata(job_id)
        if not metadata:
            logger.warning(
                "Cannot confirm cancellation of %s because scheduler metadata is missing",
                job_id,
            )
            return False
        if not self._select_recorded_hostname(metadata, job_id):
            return False

        fenced = bool(metadata.get("launch_uncertain"))
        if metadata.get("launch_uncertain"):
            candidates = self._launch_reconciliation_candidates(metadata, job_id)
            if not candidates:
                logger.warning(
                    "Cannot confirm cancellation of %s while its fenced SLURM "
                    "launch is not visible",
                    job_id,
                )
                return False
        else:
            recorded_id = str(metadata.get("slurm_job_id", "") or "")
            if recorded_id and not metadata.get("job_name"):
                candidates = [(recorded_id, "UNKNOWN")]
            else:
                job_name = metadata.get("job_name") or self.get_job_name(job_id)
                candidates = self._find_slurm_jobs_by_name(job_name)
                if recorded_id and all(
                    item[0] != recorded_id for item in candidates
                ):
                    candidates.append((recorded_id, "UNKNOWN"))

        if not candidates:
            if metadata.get("submission_attempted") is False:
                return True
            logger.warning(
                "Cannot confirm cancellation of %s because its SLURM job ID is missing",
                job_id,
            )
            return False

        all_quiescent = True
        for slurm_job_id, _listed_status in candidates:
            force_known_id = bool(
                metadata.get("submission_attempted")
                and str(metadata.get("slurm_job_id", "") or "") == slurm_job_id
            )
            if force_known_id:
                canceled = self._cancel_slurm_job_id(
                    job_id,
                    slurm_job_id,
                    force_scancel_if_missing=True,
                )
            else:
                canceled = self._cancel_slurm_job_id(job_id, slurm_job_id)
            if not canceled:
                all_quiescent = False
        if not all_quiescent:
            return False

        if fenced:
            selected_id = candidates[0][0]
            metadata["slurm_job_id"] = selected_id
            metadata["launch_uncertain"] = False
            if metadata.get("pre_launch_slurm_job_id"):
                failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
                pre_launch_id = str(metadata["pre_launch_slurm_job_id"])
                if pre_launch_id not in failed_ids:
                    failed_ids.append(pre_launch_id)
                metadata["retry_count"] = int(metadata.get("retry_count", 0)) + 1
            failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
            for candidate_id, _status in candidates[1:]:
                if candidate_id not in failed_ids:
                    failed_ids.append(candidate_id)
            self._jobs[job_id] = metadata
            try:
                self._persist_runtime_state(job_id, required=True)
            except Exception:
                # Every possible writer is already terminal; retaining the
                # durable fence is conservative and safe for later recovery.
                logger.exception(
                    "Could not persist reconciled SLURM cancellation for %s",
                    job_id,
                )
        return True

    def _cancel_slurm_job_id(
        self,
        job_id: str,
        slurm_job_id: str,
        *,
        force_scancel_if_missing: bool = False,
    ) -> bool:
        """Cancel and verify one scheduler ID without performing discovery."""
        slurm_status = self.get_slurm_job_status(slurm_job_id)
        if slurm_status in TERMINAL_SLURM_STATES:
            self._set_job_message(
                job_id,
                f"SLURM job {slurm_job_id} already terminated with status: {slurm_status}",
            )
            return True
        if slurm_status == "NOT_FOUND":
            if not force_scancel_if_missing:
                logger.warning(
                    "SLURM job %s is not visible before cancellation; retaining "
                    "artifacts because scheduler/accounting state is ambiguous",
                    slurm_job_id,
                )
                return False
            logger.warning(
                "Accepted SLURM job %s is not visible yet; issuing scancel by its "
                "authoritative scheduler id",
                slurm_job_id,
            )

        result = self._ssh(
            f"scancel {shlex.quote(str(slurm_job_id))}",
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                slurm_status = self.get_slurm_job_status(slurm_job_id)
                if slurm_status in TERMINAL_SLURM_STATES or slurm_status == "NOT_FOUND":
                    self._set_job_message(
                        job_id,
                        f"SLURM job {slurm_job_id} terminated with status: {slurm_status}",
                    )
                    return True
                time.sleep(0.5)
            logger.warning(
                "Timed out waiting for SLURM job %s to terminate after scancel",
                slurm_job_id,
            )
            return False
        detail = (result.stderr or result.stdout or "").strip()
        if "Invalid job id" in detail or "does not exist" in detail:
            return True
        logger.warning("Failed to cancel SLURM job %s: %s", slurm_job_id, detail)
        self._set_job_message(job_id, f"Failed to cancel SLURM job {slurm_job_id}: {detail}")
        return False

    def delete_job(self, job_id: str) -> bool:
        return self.cancel_job(job_id)

    def get_canonical_results_root(self) -> str:
        """Resolve the configured remote results root for durable ownership."""
        base = PurePosixPath(self.base_results_dir)
        if not base.is_absolute() or str(base) == "/" or ".." in base.parts:
            raise ValueError("SLURM base results directory is not safely scoped")
        result = self._ssh(
            f"realpath -m -- {shlex.quote(str(base / 'results'))}",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        resolved = (result.stdout or "").strip()
        resolved_path = PurePosixPath(resolved)
        if (
            result.returncode != 0
            or not resolved_path.is_absolute()
            or str(resolved_path) == "/"
            or ".." in resolved_path.parts
            or "\n" in resolved
        ):
            detail = (result.stderr or result.stdout or "").strip()
            raise ValueError(f"Could not resolve a safe SLURM results root: {detail}")
        return str(resolved_path)

    def delete_job_artifacts(self, job_id: str, *, expected_root: str) -> bool:
        """Delete only ``<base_results_dir>/results/<job_id>`` over SSH."""
        try:
            safe_job_id = validate_job_id(job_id)
            recorded_root = PurePosixPath(expected_root)
            configured_base = PurePosixPath(self.base_results_dir)
            if (
                not recorded_root.is_absolute()
                or str(recorded_root) == "/"
                or ".." in recorded_root.parts
                or not configured_base.is_absolute()
                or str(configured_base) == "/"
                or ".." in configured_base.parts
            ):
                raise ValueError("recorded SLURM results root is not safely scoped")
        except ValueError as exc:
            logger.warning("Could not validate artifacts for job %s: %s", job_id, exc)
            return False

        quoted_root = shlex.quote(str(recorded_root))
        quoted_configured = shlex.quote(str(configured_base / "results"))
        quoted_job_id = shlex.quote(safe_job_id)
        command = (
            f"root=$(realpath -m -- {quoted_configured}) || exit 1; "
            f"[ \"$root\" = {quoted_root} ] || exit 1; "
            f"cd -P -- {quoted_root} || exit 1; "
            f"[ \"$(pwd -P)\" = {quoted_root} ] || exit 1; "
            f"rm -rf -- ./{quoted_job_id}"
        )
        result = self._ssh(
            command,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return True
        detail = (result.stderr or result.stdout or "").strip()
        logger.warning("Failed to delete artifacts for SLURM job %s: %s", job_id, detail)
        return False

    def delete(self, job_id: str) -> bool:
        return self.cancel_job(job_id)

    def get_available_instances(self, **kwargs) -> dict:
        """Best-effort partition discovery for the SDK resource UI."""
        result = self._ssh(
            "sinfo -h -o '%P|%G|%c|%m|%D|%T'",
            check=False,
            timeout=60,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}

        instances: dict[str, dict[str, Any]] = {}
        for line in result.stdout.splitlines():
            parts = line.split("|")
            if len(parts) != 6:
                continue
            partition, gres, cpus, mem, nodes, state = [p.strip() for p in parts]
            gpu_count = self._gpu_count_from_gres(gres)
            instances[partition.rstrip("*")] = {
                "cluster": partition.rstrip("*"),
                "gpu_type": partition.rstrip("*"),
                "instance_type": gres,
                "gpu_count": gpu_count,
                "cpu_cores": int(cpus) if cpus.isdigit() else cpus,
                "system_memory": f"{mem}MB",
                "backend_type": "slurm",
                "partition": partition.rstrip("*"),
                "gres": gres,
                "cpus": cpus,
                "memory_mb": mem,
                "nodes": nodes,
                "state": state,
            }
        return instances

    def get_job_replicas(self, job_id: str) -> list[dict]:
        metadata = self._get_job_metadata(job_id)
        if not metadata or not metadata.get("slurm_job_id"):
            return []
        slurm_job_id = metadata["slurm_job_id"]
        result = self._ssh(
            f"squeue -h -j {shlex.quote(str(slurm_job_id))} -o '%i|%T|%N|%R'",
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        replicas = []
        for line in result.stdout.splitlines():
            fields = line.split("|")
            if len(fields) == 4:
                replicas.append({
                    "id": fields[0],
                    "status": fields[1],
                    "nodes": fields[2],
                    "reason": fields[3],
                })
        return replicas

    def get_failure_analysis(self, job_id: str) -> dict | None:
        logs = self.get_job_logs(job_id, tail_lines=500)
        if not logs:
            return None
        for pattern in RETRIABLE_ERROR_PATTERNS:
            match = re.search(pattern, logs, re.IGNORECASE)
            if match:
                return {
                    "reason": "infrastructure_failure_pattern",
                    "pattern": pattern,
                    "match": match.group(0),
                    "retriable": True,
                }
        return None

    def _build_slurm_script(
        self,
        job_name: str,
        num_nodes: int,
        num_gpus: int,
        cpus_per_task: int,
        time_hours: float,
        account: str | None,
        partition: str,
        log_dir: str,
        mail_user: str | None,
        image: str,
        container_mounts: str,
        use_timeout: bool,
        timeout_hours: float,
        use_requeue: bool,
        use_srun: bool,
        command: str,
        docker_env_vars: dict[str, Any],
        exclusive: bool,
        network: str | None = None,
        allowed_nodes: list[str] | str | None = None,
    ) -> str:
        """Build a Core-style sbatch script for SDK entrypoint execution."""
        time_str = f"{int(time_hours):02d}:{int((time_hours % 1) * 60):02d}:00"
        is_multi_node = num_nodes > 1
        ntasks_per_node = 1

        if network != "cosmos-rl" and not is_multi_node and num_gpus > 1:
            # The single task carries every local rank (Lightning self-spawn),
            # so the allocation must carry CPUs for every rank's dataloaders:
            # a 4-GPU job with the base 16 CPUs starves workers vs TAO Core's
            # 64 (base × GPUs). Capped so an 8-GPU request stays within a
            # typical GPU node; override with SLURM_MAX_CPUS_PER_TASK.
            max_cpus = int(os.environ.get("SLURM_MAX_CPUS_PER_TASK", "128"))
            cpus_per_task = min(cpus_per_task * num_gpus, max_cpus)

        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --nodes={num_nodes}",
            f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        ]
        if num_gpus > 0:
            script_lines.append(f"#SBATCH --gres=gpu:{num_gpus}")
        if is_multi_node:
            script_lines.append(f"#SBATCH --ntasks={num_nodes}")

        script_lines.extend([
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            "#SBATCH --wait-all-nodes=1",
        ])
        if network == "cosmos-rl":
            script_lines.append("#SBATCH --mem=0")
        if exclusive:
            script_lines.append("#SBATCH --exclusive")

        script_lines.extend([
            f"#SBATCH --time={time_str}",
            f"#SBATCH --job-name={job_name}",
        ])
        if account:
            script_lines.append(f"#SBATCH --account={account}")
        if partition:
            script_lines.append(f"#SBATCH --partition={partition}")
        nodelist = self._format_node_list(allowed_nodes)
        if nodelist:
            script_lines.append(f"#SBATCH --nodelist={nodelist}")

        script_lines.extend([
            f"#SBATCH --output={log_dir}/%x-%j/main.out",
            f"#SBATCH --error={log_dir}/%x-%j/main.err",
            "#SBATCH --open-mode=append",
        ])

        if mail_user:
            script_lines.extend([
                "#SBATCH --mail-type=all",
                f"#SBATCH --mail-user={mail_user}",
            ])
        if use_requeue:
            script_lines.append("#SBATCH --requeue")
        else:
            # Absence of --requeue does not disable a cluster/account default.
            # Emit the explicit inverse required by the platform contract.
            script_lines.append("#SBATCH --no-requeue")

        script_lines.extend([
            "",
            "export NCCL_DEBUG=INFO",
            "export LOGLEVEL=INFO",
            "",
        ])

        if docker_env_vars:
            script_lines.append("{ set +x; } 2>/dev/null  # don't echo credential exports")
        for key, value in docker_env_vars.items():
            safe_value = str(value).replace("'", "'\"'\"'")
            script_lines.append(f"export {key}='{safe_value}'")
        if docker_env_vars:
            script_lines.append("set -x  # re-enable trace after credential exports")
            script_lines.append("")

        if is_multi_node:
            script_lines.extend([
                f"export WORLD_SIZE={num_nodes}",
                f"export NUM_GPU_PER_NODE={num_gpus}",
                "export MASTER_PORT=29500",
                "",
                "NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)",
                "export MASTER_ADDR=$(echo $NODELIST | cut -d' ' -f1)",
                "export NODE_RANK=$SLURM_NODEID",
                "",
            ])

        cuda_override = str(docker_env_vars.get("CUDA_OVERRIDE_VERSION", "")).strip()
        if cuda_override:
            cuda_base_path = f"/usr/local/cuda-{cuda_override}"
            script_lines.extend([
                "# CUDA environment override",
                f"export CUDA_HOME={cuda_base_path}",
                f"export CUDA_PATH={cuda_base_path}",
                f"export LD_LIBRARY_PATH=\"{cuda_base_path}/lib64:{cuda_base_path}/lib:$LD_LIBRARY_PATH\"",
                f"export LIBRARY_PATH=\"{cuda_base_path}/lib64:{cuda_base_path}/lib:${{LIBRARY_PATH:-}}\"",
                f"export CPATH=\"{cuda_base_path}/include:${{CPATH:-}}\"",
                "",
            ])

        script_lines.extend([
            "# Job spec",
            f"OUTFILE=\"{log_dir}/%x-%j/%n.out\"",
            f"ERRFILE=\"{log_dir}/%x-%j/%n.err\"",
            "",
        ])

        container_image = self._to_pyxis_image(image)
        container_env_keys = ",".join(docker_env_vars.keys()) if docker_env_vars else ""
        if is_multi_node:
            multi_node_env_keys = [
                "NODE_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "NUM_GPU_PER_NODE",
            ]
            multi_node_env_str = ",".join(multi_node_env_keys)
            container_env_keys = (
                f"{multi_node_env_str},{container_env_keys}"
                if container_env_keys
                else multi_node_env_str
            )

        srun_base = (
            f"srun --container-image={container_image} "
            f"--container-mounts={container_mounts}"
        )
        if container_env_keys:
            srun_base += f" --container-env={container_env_keys}"
        if use_srun:
            srun_base += " -o $OUTFILE -e $ERRFILE --open-mode=append"
        if network == "cosmos-rl":
            srun_base += " --no-container-mount-home --export=ALL"

        if network != "cosmos-rl":
            # TAO PyTorch launches Lightning networks as plain `python
            # train.py` (torchrun is reserved for LIGHTNING_EXCLUDED_NETWORKS),
            # so multi-GPU depends on Lightning self-spawning its DDP workers —
            # which the container env disables twice over: SLURM detection
            # (older Lightning honors only job name "bash"; every version keys
            # on SLURM_NTASKS), and rank vars injected at container start on
            # the node (LOCAL_RANK=0 RANK=0 WORLD_SIZE=1;
            # creates_processes_externally is `"LOCAL_RANK" in os.environ`).
            # The wrapper runs after that injection, so unsetting yields a
            # clean LightningEnvironment that self-spawns train.num_gpus
            # ranks — identical to Docker/K8s behavior. Single-node drops
            # every distributed var; multi-node keeps the five coordination
            # exports (WORLD_SIZE NUM_GPU_PER_NODE NODE_RANK MASTER_ADDR
            # MASTER_PORT) that the TAO entrypoint override and cross-node
            # rendezvous need, unsetting only rank-level vars — GROUP_RANK
            # included, since LightningEnvironment falls back to it for
            # node_rank while NODE_RANK must stay. cosmos-rl has its own
            # launcher contract (ntasks=1 by design) and is untouched. Scoped
            # to the in-container command so srun step naming and %x log
            # paths keep the real job name. SLURM_NODEID is never unset in
            # any path: cosmos-rl consumes it, and the multi-node wrapper
            # derives NODE_RANK from it.
            unset_vars = (
                "SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
                "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK"
            )
            if is_multi_node:
                # The batch script's `export NODE_RANK=$SLURM_NODEID` runs
                # once, on the batch host, so every node inherits NODE_RANK=0
                # and Lightning assigns duplicate global ranks (both nodes
                # log GLOBAL_RANK: 0..G-1 and rendezvous never completes).
                # slurmstepd sets SLURM_NODEID per task on every node —
                # re-derive here, overriding the baked value; :? fails loudly
                # if SLURM_NODEID is ever absent.
                node_rank_fix = "export NODE_RANK=${SLURM_NODEID:?}; "
            else:
                node_rank_fix = ""
                unset_vars += (
                    " WORLD_SIZE NODE_RANK NUM_GPU_PER_NODE MASTER_ADDR"
                    " MASTER_PORT NNODES NPROC_PER_NODE"
                )
            command = "/bin/bash -c " + shlex.quote(
                f"export SLURM_JOB_NAME=bash; {node_rank_fix}"
                f"unset {unset_vars}; {command}"
            )

        if use_timeout:
            timeout_mins = (
                int(timeout_hours * 60)
                if timeout_hours
                else int(max(time_hours - 0.2, 0.1) * 60)
            )
            script_lines.append(f"timeout {timeout_mins}m {srun_base} -- {command}")
        else:
            script_lines.append(f"{srun_base} -- {command}")

        if use_timeout and use_requeue:
            script_lines.extend([
                "",
                "# Launch self again when timeout exits with code 124",
                "if [[ $? == 124 ]]; then",
                "    scontrol requeue $SLURM_JOB_ID",
                "fi",
            ])

        return "\n".join(script_lines) + "\n"

    def get_slurm_job_status(self, slurm_job_id: str | int | None) -> str:
        """Get raw SLURM status using squeue first, then sacct."""
        if not slurm_job_id:
            return "ERROR"

        quoted_id = shlex.quote(str(slurm_job_id))
        result = self._ssh(
            f"squeue --job={quoted_id} --format='%T' --noheader",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip().upper()

        result = self._ssh(
            f"sacct --job={quoted_id} --format=State --noheader --parsable2",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout.strip():
            states = []
            for line in result.stdout.splitlines():
                state = line.strip().split("|", 1)[0].split("+", 1)[0].upper()
                if state:
                    states.append(state)
            for terminal in (
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "BOOT_FAIL",
                "DEADLINE",
            ):
                if terminal in states:
                    return terminal
            if "COMPLETED" in states:
                return "COMPLETED"
            if states:
                return states[0]

        if result.returncode != 0:
            logger.warning(
                "Failed to query SLURM status for %s: %s",
                slurm_job_id,
                (result.stderr or result.stdout or "").strip(),
            )
            return "ERROR"
        return "NOT_FOUND"

    def get_slurm_job_id_from_output(self, sbatch_output: str) -> str | None:
        match = re.search(r"Submitted batch job\s+(\d+)", sbatch_output or "")
        return match.group(1) if match else None

    def _is_retriable_failure(self, slurm_status: str, job_id: str) -> bool:
        """Return True for infrastructure-looking failures."""
        if slurm_status in {"NODE_FAIL", "BOOT_FAIL"}:
            return True
        if slurm_status != "FAILED":
            return False

        logs = self.get_job_logs(job_id, tail_lines=500)
        if not logs:
            return True

        for pattern in RETRIABLE_ERROR_PATTERNS:
            if re.search(pattern, logs, re.IGNORECASE):
                return True
        return False

    def _resubmit_failed_job(
        self,
        job_id: str,
        failed_slurm_job_id: str,
        retry_count: int,
    ) -> tuple[bool, str | None, str | None]:
        metadata = self._get_job_metadata(job_id)
        if not metadata:
            return False, None, "job metadata not found"

        remote_script = metadata.get("remote_script")
        if not remote_script:
            return False, None, "remote sbatch script path not recorded"

        if not self.check_path_exists(remote_script):
            return False, None, f"sbatch script not found at {remote_script}"

        try:
            fenced_runtime = self._begin_launch_fence(job_id)
            launch_token = str(fenced_runtime.get("launch_token", ""))
            fenced_runtime = self._confirm_launch_fence(job_id, launch_token)
        except Exception as exc:
            return False, None, f"could not confirm retry launch fence: {exc}"

        result = self._ssh(
            f"sbatch --export=ALL {shlex.quote(remote_script)}",
            check=False,
            timeout=TIMEOUT_SECONDS,
            allow_failover=False,
        )
        if result.returncode != 0:
            if self._is_connection_failure(result):
                raise RuntimeError(
                    "SLURM retry submission outcome is ambiguous; durable launch "
                    "fence retained for reconciliation"
                )
            metadata = copy.deepcopy(fenced_runtime)
            metadata["launch_uncertain"] = False
            self._jobs[job_id] = metadata
            self._persist_runtime_state(job_id, required=True)
            return False, None, (result.stderr or result.stdout or "").strip()

        new_slurm_job_id = self.get_slurm_job_id_from_output(result.stdout)
        if not new_slurm_job_id:
            raise RuntimeError(
                f"Could not parse retry sbatch output; launch remains fenced: "
                f"{result.stdout!r}"
            )

        metadata = copy.deepcopy(fenced_runtime)
        failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
        if failed_slurm_job_id and failed_slurm_job_id not in failed_ids:
            failed_ids.append(failed_slurm_job_id)
        metadata["retry_count"] = retry_count + 1
        metadata["slurm_job_id"] = new_slurm_job_id
        metadata["launch_uncertain"] = False
        metadata["message"] = (
            f"Job resubmitted after infrastructure failure "
            f"(attempt {retry_count + 1}/{MAX_JOB_RETRIES}); "
            f"new SLURM Job ID: {new_slurm_job_id}"
        )
        self._jobs[job_id] = metadata
        try:
            self._persist_runtime_state(job_id, required=True)
        except Exception as exc:
            quiescent = False
            try:
                quiescent = self.cancel_job(job_id)
            except Exception:
                logger.exception(
                    "Could not cancel retry SLURM job %s after persistence failed",
                    new_slurm_job_id,
                )
            error = RuntimeError(
                f"Retry SLURM job {new_slurm_job_id} was accepted but its "
                "scheduler identity could not be persisted"
            )
            error.writer_quiescent = quiescent
            raise error from exc
        return True, new_slurm_job_id, None

    def _prepare_container_image(
        self,
        image: str,
        sqsh_cache_dir: str,
        conversion_partition: str = "cpu",
        conversion_account: str | None = None,
        conversion_timeout_minutes: int = 30,
        conversion_memory_gb: int = 32,
        force_reconvert_latest: bool = False,
        job_id: str | None = None,
    ) -> str:
        """Prepare a cached SQSH image, falling back to Docker image on failure."""
        # A caller-supplied SQSH is already the native Pyxis image artifact.
        # Use it directly: it must not be treated as a registry reference and
        # sent through enroot import or copied into the SDK cache.
        if image.startswith("/") and image.endswith(".sqsh"):
            if not self.check_path_exists(image):
                raise FileNotFoundError(f"caller-supplied SQSH does not exist: {image}")
            logger.info("Using caller-supplied SQSH image directly: %s", image)
            if job_id:
                self._set_job_message(job_id, "Using caller-supplied SQSH image directly")
            return image

        cache_dir = sqsh_cache_dir.rstrip("/")
        sqsh_path = f"{cache_dir}/{self._get_sqsh_filename(image)}"
        is_latest_tag = ":latest" in image or image.endswith(":latest")
        should_force_reconvert = is_latest_tag and force_reconvert_latest

        if should_force_reconvert and self._check_sqsh_exists(sqsh_path):
            self._ssh(f"rm -f {shlex.quote(sqsh_path)}", check=False)
        elif self._check_sqsh_exists(sqsh_path):
            logger.info("Using cached SQSH image: %s", sqsh_path)
            if job_id:
                self._set_job_message(job_id, "Using cached SQSH image")
            return sqsh_path

        logger.info(
            "Starting SQSH conversion for %s into %s; timeout=%d minutes",
            image,
            sqsh_path,
            conversion_timeout_minutes,
        )
        if job_id:
            self._set_job_message(
                job_id,
                "Starting SQSH conversion for container image (may take 5-30 minutes)",
            )

        try:
            success = self._convert_to_sqsh_via_srun(
                image=image,
                sqsh_path=sqsh_path,
                sqsh_cache_dir=cache_dir,
                partition=conversion_partition,
                account=conversion_account,
                timeout_minutes=conversion_timeout_minutes,
                memory_gb=conversion_memory_gb,
            )
            if success:
                if job_id:
                    self._set_job_message(
                        job_id,
                        "SQSH conversion complete; submitting job to SLURM",
                    )
                return sqsh_path
        except Exception as exc:
            logger.warning("SQSH conversion raised for image %s: %s", image, exc)
            logger.debug(traceback.format_exc())

        # A concurrent session may have completed the conversion while ours
        # failed; a cache hit here is still a valid (atomically renamed) image.
        if self._check_sqsh_exists(sqsh_path):
            logger.info("Using cached SQSH image from concurrent conversion: %s", sqsh_path)
            return sqsh_path

        if self._bool_config("SLURM_REQUIRE_SQSH", False):
            raise RuntimeError(
                f"SQSH conversion failed for {image} and SLURM_REQUIRE_SQSH is "
                "set; refusing to fall back to direct registry image."
            )

        if job_id:
            self._set_job_message(
                job_id,
                "SQSH conversion failed; falling back to Docker image",
            )
        return image

    def _convert_to_sqsh_via_srun(
        self,
        image: str,
        sqsh_path: str,
        sqsh_cache_dir: str,
        partition: str = "cpu",
        account: str | None = None,
        timeout_minutes: int = 30,
        memory_gb: int = 32,
    ) -> bool:
        """Convert a Docker image to SQSH using remote ``srun enroot import``.

        Concurrency- and crash-safe: the import writes to a job-scoped
        ``.partial`` file and only a magic-verified image is renamed onto the
        final cache path, so readers never observe a half-written file. A
        deterministic conversion job name lets concurrent sessions wait on an
        in-flight conversion instead of starting a second writer, and lets a
        failed session scancel its own orphaned conversion. Enroot layer
        extraction runs in a job-scoped temp dir (``SLURM_ENROOT_TEMP_PATH``
        overridable) instead of the node's shared /tmp, which is too small or
        reaped mid-extraction on some clusters.
        """
        self._ssh(f"mkdir -p {shlex.quote(sqsh_cache_dir)}", check=True)
        docker_image = self._to_enroot_image(image)
        job_name = f"tao-sqsh-{hashlib.sha1(image.encode()).hexdigest()[:8]}"

        if self._sqsh_conversion_in_flight(job_name):
            logger.info(
                "SQSH conversion for %s already in flight (job name %s); waiting",
                image,
                job_name,
            )
            return self._wait_for_sqsh(sqsh_path, timeout_minutes * 60)

        enroot_temp_root = os.environ.get(
            "SLURM_ENROOT_TEMP_PATH", f"{sqsh_cache_dir}/.enroot-tmp"
        )
        convert_script = "; ".join([
            "set -eu",
            f"sqsh={shlex.quote(sqsh_path)}",
            'partial="$sqsh.partial.${SLURM_JOB_ID:-$$}"',
            f'export ENROOT_TEMP_PATH={shlex.quote(enroot_temp_root)}"/${{SLURM_JOB_ID:-$$}}"',
            'mkdir -p "$ENROOT_TEMP_PATH"',
            "trap 'rm -rf \"$ENROOT_TEMP_PATH\" \"$partial\"' EXIT",
            f'enroot import -o "$partial" -- {shlex.quote(docker_image)}',
            '[ "$(head -c 4 "$partial")" = "hsqs" ]',
            'mv -f "$partial" "$sqsh"',
        ])

        srun_parts = [
            "srun",
            "-n1",
            f"-p {shlex.quote(partition)}",
            f"--mem={int(memory_gb)}G",
            f"-t {int(timeout_minutes)}:00",
            f"--job-name={shlex.quote(job_name)}",
        ]
        if account:
            srun_parts.append(f"-A {shlex.quote(account)}")
        srun_parts.extend(["bash", "-c", shlex.quote(convert_script)])

        start = time.time()
        logger.info(
            "Running remote SQSH conversion command on partition %s with %dGB memory",
            partition,
            memory_gb,
        )
        result = self._ssh(
            " ".join(srun_parts),
            check=False,
            # srun -t only bounds run time; leave headroom for queue wait.
            timeout=timeout_minutes * 60 + 600,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            logger.warning(
                "SQSH conversion failed after %.1fs: %s",
                elapsed,
                (result.stderr or result.stdout or "").strip(),
            )
            # If the SSH channel died (not the job), the srun keeps running
            # remotely; reap it so it cannot race a future conversion.
            self._ssh(
                f"scancel --jobname={shlex.quote(job_name)} -u \"$USER\"",
                check=False,
                timeout=TIMEOUT_SECONDS,
            )
            return False
        if self._check_sqsh_exists(sqsh_path):
            logger.info("SQSH conversion completed in %.1fs: %s", elapsed, sqsh_path)
            return True
        logger.warning("SQSH conversion command completed but %s was not created", sqsh_path)
        return False

    def _sqsh_conversion_in_flight(self, job_name: str) -> bool:
        result = self._ssh(
            f"squeue -h -u \"$USER\" --name={shlex.quote(job_name)} -o %T",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    def _wait_for_sqsh(self, sqsh_path: str, timeout_seconds: int) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self._check_sqsh_exists(sqsh_path):
                return True
            time.sleep(15)
        return False

    def _get_sqsh_filename(self, image: str) -> str:
        return image.replace("/", "_").replace(":", "_") + ".sqsh"

    def _check_sqsh_exists(self, sqsh_path: str) -> bool:
        # Require the SquashFS magic, not just existence: a legacy or
        # non-atomic writer may have left a partial file at the cache path.
        quoted = shlex.quote(sqsh_path)
        result = self._ssh(
            f'test -f {quoted} && [ "$(head -c 4 {quoted})" = "hsqs" ] '
            "&& echo exists || echo not_found",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        return result.returncode == 0 and result.stdout.strip() == "exists"

    def check_path_exists(self, remote_path: str) -> bool:
        result = self._ssh(
            f"test -e {shlex.quote(remote_path)} && echo exists || echo not_found",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        return result.returncode == 0 and result.stdout.strip() == "exists"

    def find_checkpoint_paths(self, remote_path: str) -> list[str]:
        """Return checkpoint artifacts below an absolute shared-Lustre path.

        Cosmos-RL promotions resume from the ``policy`` directory containing
        ``cosmos_config`` rather than from the HF safetensors export. Include
        those directories alongside generic checkpoint files.
        """
        path = remote_path.removeprefix("lustre://")
        if not path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError("checkpoint search path must be an absolute scoped path")
        file_names = (
            "*.pth", "*.pth.tar", "*.tlt", "*.hdf5", "*.pt", "*.ckpt",
            "*.engine", "*.onnx", "*.trt",
        )
        name_expr = " -o ".join(
            f"-name {shlex.quote(pattern)}" for pattern in file_names
        )
        command = (
            f"find {shlex.quote(path)} "
            f"\\( -type f \\( {name_expr} \\) -print \\) -o "
            "\\( -type f -name cosmos_config -printf '%h\\n' \\)"
        )
        result = self._ssh(command, check=False, timeout=60)
        if result.returncode != 0:
            return []
        return list(dict.fromkeys(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ))

    def test_ssh_connection(self) -> tuple[bool, str]:
        result = self._ssh(
            "echo 'SSH connection successful'",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, (result.stderr or result.stdout or "").strip()

    def _switch_to_next_hostname(self) -> bool:
        if len(self.login_hostnames) <= 1:
            return False
        previous = self.login_hostname
        self.current_hostname_index = (
            self.current_hostname_index + 1
        ) % len(self.login_hostnames)
        self.login_hostname = self.login_hostnames[self.current_hostname_index]
        if self.login_hostname == previous:
            return False
        self.ssh_options = self._get_ssh_options()
        logger.info(
            "Switching SLURM hostname from %s to %s",
            previous,
            self.login_hostname,
        )
        return True

    @staticmethod
    def _normalize_hostname(hostname: str) -> str:
        return str(hostname or "").strip().rstrip(".").lower()

    def _select_recorded_hostname(self, metadata: dict, job_id: str) -> bool:
        """Prefer the submission host, falling back within the verified cluster."""
        recorded = self._normalize_hostname(metadata.get("hostname", ""))
        configured = {
            self._normalize_hostname(host): (index, host)
            for index, host in enumerate(self.login_hostnames)
        }
        if not recorded or recorded not in configured:
            logger.info(
                "Recorded login host %r for SLURM job %s is unavailable; "
                "using configured host %s",
                metadata.get("hostname"),
                job_id,
                self.login_hostname,
            )
            metadata["hostname"] = self.login_hostname
            return True
        index, hostname = configured[recorded]
        self.current_hostname_index = index
        self.login_hostname = hostname
        return True

    def _find_ssh_key(self) -> str | None:
        ssh_dirs = [os.path.join(os.path.expanduser("~"), ".ssh")]
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            if current_user == "www-data":
                ssh_dirs.append("/home/www-data/.ssh")
        except Exception:
            pass
        if "/root/.ssh" not in ssh_dirs:
            ssh_dirs.append("/root/.ssh")

        for ssh_dir in ssh_dirs:
            if not os.path.isdir(ssh_dir):
                continue
            try:
                os.listdir(ssh_dir)
            except PermissionError:
                continue
            for key_name in ("id_ed25519", "id_ecdsa", "id_rsa", "id_dsa"):
                key_path = os.path.join(ssh_dir, key_name)
                if os.path.exists(key_path):
                    return key_path
        return None

    def _get_ssh_options(self) -> list[str]:
        options = [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "BatchMode=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "PubkeyAuthentication=yes",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            f"ConnectTimeout={TIMEOUT_SECONDS}",
            "-o",
            f"ServerAliveInterval={TIMEOUT_SECONDS}",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "UpdateHostKeys=no",
        ]
        if self.ssh_key_path and os.path.exists(self.ssh_key_path):
            options.extend(["-i", self.ssh_key_path])
            try:
                os.chmod(self.ssh_key_path, 0o600)
            except (PermissionError, OSError):
                pass
        return options

    def _build_ssh_command(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            *self.ssh_options,
            f"{self.login_user}@{self.login_hostname}",
            remote_command,
        ]

    def _ssh(
        self,
        remote_command: str,
        check: bool = True,
        timeout: int = TIMEOUT_SECONDS,
        *,
        allow_failover: bool = True,
    ) -> subprocess.CompletedProcess:
        attempts = max(len(self.login_hostnames), 1) if allow_failover else 1
        last_result: subprocess.CompletedProcess | None = None
        last_exc: BaseException | None = None

        for attempt in range(attempts):
            cmd = self._build_ssh_command(remote_command)
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                last_exc = exc
                if attempt + 1 < attempts and self._switch_to_next_hostname():
                    continue
                raise

            last_result = result
            if self._is_connection_failure(result):
                if attempt + 1 < attempts and self._switch_to_next_hostname():
                    continue

            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    result.stdout,
                    result.stderr,
                )
            return result

        if last_result is not None:
            if check and last_result.returncode != 0:
                raise subprocess.CalledProcessError(
                    last_result.returncode,
                    self._build_ssh_command(remote_command),
                    last_result.stdout,
                    last_result.stderr,
                )
            return last_result
        if last_exc:
            raise last_exc
        raise RuntimeError("SSH command failed without a result")

    def _scp_text(self, content: str, remote_path: str) -> None:
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self._ssh(f"mkdir -p {shlex.quote(remote_dir)}", check=True)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            local_path = tmp.name
        try:
            attempts = max(len(self.login_hostnames), 1)
            for attempt in range(attempts):
                cmd = [
                    "scp",
                    *self.ssh_options,
                    local_path,
                    f"{self.login_user}@{self.login_hostname}:{remote_path}",
                ]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    return
                if self._is_connection_failure(result):
                    if attempt + 1 < attempts and self._switch_to_next_hostname():
                        continue
                raise subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    result.stdout,
                    result.stderr,
                )
        finally:
            try:
                os.unlink(local_path)
            except OSError:
                pass

    def _is_connection_failure(self, result: subprocess.CompletedProcess) -> bool:
        if result.returncode == 255:
            return True
        text = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
        needles = (
            "connection timed out",
            "connection refused",
            "no route to host",
            "could not resolve hostname",
            "connection reset",
            "operation timed out",
            "temporary failure in name resolution",
            "network is unreachable",
        )
        return any(needle in text for needle in needles)

    @staticmethod
    def _to_compact_json(value: Any) -> str:
        def _json_serializer(obj: Any) -> Any:
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            if isinstance(obj, (bytes, bytearray)):
                return obj.decode("utf-8", errors="replace")
            if hasattr(obj, "__fspath__"):
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
                default=_json_serializer,
            )
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return json.dumps(
                    parsed,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=_json_serializer,
                )
            except Exception:
                return value
        return json.dumps(value, separators=(",", ":"), default=_json_serializer)

    def _config_value(self, key: str) -> Any:
        return self._creds.get(key) or os.environ.get(key)

    def _int_config(self, key: str, default: int) -> int:
        value = self._config_value(key)
        try:
            return int(value) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def _float_config(self, key: str, default: float) -> float:
        value = self._config_value(key)
        try:
            return float(value) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def _bool_config(self, key: str, default: bool) -> bool:
        value = self._config_value(key)
        if value in (None, ""):
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _num_gpus_from_shape(self, resource_shape: str | None) -> int:
        if resource_shape:
            match = re.search(r"(\d+)$", resource_shape)
            if match:
                return max(int(match.group(1)), 0)
        return self._int_config("SLURM_NUM_GPUS", 1)

    @staticmethod
    def _format_node_list(nodes: list[str] | str | None) -> str | None:
        if not nodes:
            return None
        if isinstance(nodes, str):
            return nodes
        return ",".join(str(node) for node in nodes if node)

    @staticmethod
    def _gpu_count_from_gres(gres: str) -> int | None:
        if not gres or gres.lower() in {"(null)", "none", "n/a"}:
            return None
        counts = [int(match) for match in re.findall(r"gpu(?::[^:,]+)?:(\d+)", gres)]
        if counts:
            return max(counts)
        if "gpu" in gres.lower():
            return 1
        return None

    def _container_mounts_from_kwargs(self, kwargs: dict[str, Any]) -> str:
        mounts = kwargs.get("mounts")
        if isinstance(mounts, str) and mounts:
            return mounts
        if isinstance(mounts, list) and mounts:
            parts = []
            for mount in mounts:
                if not isinstance(mount, dict):
                    continue
                source = mount.get("from_path") or mount.get("source") or mount.get("src")
                target = (
                    mount.get("mount_path")
                    or mount.get("path")
                    or mount.get("target")
                    or source
                )
                if source and target:
                    parts.append(f"{source}:{target}")
            if parts:
                return ",".join(parts)
        return self.container_mounts

    @staticmethod
    def _build_entrypoint_script(command: str) -> str:
        return "\n".join([
            "#!/bin/bash",
            "set -euo pipefail",
            command,
            "",
        ])

    def _to_pyxis_image(self, image: str) -> str:
        if image.startswith("/") or image.startswith(("docker://", "dockerd://")):
            return image
        if "/" in image and not image.startswith("#"):
            registry, rest = image.split("/", 1)
            if "." in registry or ":" in registry:
                return f"{registry}#{rest}"
        return image

    def _to_enroot_image(self, image: str) -> str:
        if image.startswith(("docker://", "dockerd://")):
            return image
        if "/" in image:
            registry, rest = image.split("/", 1)
            return f"docker://{registry}#{rest}"
        return f"docker://{image}"

    def _convert_slurm_urls(self, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(value, str) and value.startswith("slurm://"):
                    obj[key] = self._slurm_url_to_path(value)
                else:
                    self._convert_slurm_urls(value)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                if isinstance(value, str) and value.startswith("slurm://"):
                    obj[idx] = self._slurm_url_to_path(value)
                else:
                    self._convert_slurm_urls(value)

    @staticmethod
    def _slurm_url_to_path(value: str) -> str:
        parsed = urlparse(value)
        if parsed.netloc:
            return f"/{parsed.netloc}{parsed.path}"
        return parsed.path or value.replace("slurm://", "/", 1)

    def _detect_default_account(self) -> str | None:
        result = self._ssh(
            "sacctmgr -nP show assoc where user=$(whoami) format=account",
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            logger.warning(
                "Could not auto-detect SLURM account: %s",
                (result.stderr or result.stdout or "").strip(),
            )
            return None

        accounts = []
        seen = set()
        for line in result.stdout.splitlines():
            account = line.strip()
            if account and account not in seen:
                accounts.append(account)
                seen.add(account)
        if accounts:
            self.account = accounts[0]
            logger.info("Auto-selected SLURM account: %s", self.account)
            return self.account
        return None

    def _read_remote_log_file(
        self,
        remote_file_path: str,
        tail_lines: int | None = None,
    ) -> str | None:
        if tail_lines is not None:
            read_command = (
                f"tail -n {int(tail_lines)} "
                f"{shlex.quote(remote_file_path)} 2>/dev/null"
            )
        else:
            read_command = f"cat {shlex.quote(remote_file_path)} 2>/dev/null"

        result = self._ssh(read_command, check=False, timeout=TIMEOUT_SECONDS)
        if result.returncode != 0 or not result.stdout:
            return None
        return result.stdout

    def _get_job_metadata(self, job_id: str) -> dict[str, Any] | None:
        metadata = self._jobs.get(job_id)
        if metadata:
            return metadata
        self._jobs = self._load_state()
        metadata = self._jobs.get(job_id)
        if metadata:
            return metadata
        metadata = self._discover_job_metadata(job_id)
        if metadata:
            self._jobs[job_id] = metadata
            self._persist_runtime_state(job_id, required=True)
        return metadata

    def _discover_job_metadata(self, job_id: str) -> dict[str, Any] | None:
        """Discover a SLURM job by backend job name via squeue/sacct lookup."""
        job_name = self.get_job_name(job_id)
        discovered = self._find_slurm_job_by_name(job_name)
        if not discovered:
            return None

        slurm_job_id, slurm_status = discovered
        job_dir = self.base_results_dir
        return {
            "slurm_job_id": slurm_job_id,
            "failed_slurm_job_ids": [],
            "retry_count": 0,
            "job_name": job_name,
            "job_dir": job_dir,
            "log_dir": f"{job_dir}/slurm-logs",
            "remote_script": f"{job_dir}/sbatch/job_{job_id}.sbatch",
            "remote_entrypoint": f"{job_dir}/entrypoints/job_{job_id}.sh",
            "remote_specs": f"{job_dir}/specs/{job_id}.json",
            "remote_env": f"{job_dir}/env/{job_id}.json",
            "remote_meta": f"{job_dir}/meta/{job_id}.json",
            "partition": self.partition,
            "account": self.account,
            "hostname": self.login_hostname,
            "submission_attempted": True,
            "launch_uncertain": False,
            "message": f"SLURM job {slurm_job_id}: {slurm_status}",
        }

    def _find_slurm_jobs_by_name(self, job_name: str) -> list[tuple[str, str]]:
        """Enumerate every scheduler job with this name, newest IDs first."""
        quoted_name = shlex.quote(job_name)
        matches: dict[str, str] = {}
        result = self._ssh(
            f"squeue --name={quoted_name} --noheader --format='%i|%T|%j'",
            check=False,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 3 and parts[2] == job_name:
                    scheduler_id = self._normalize_slurm_id(parts[0])
                    matches[scheduler_id] = parts[1].split("+", 1)[0].upper()

        result = self._ssh(
            f"sacct -nP -X --name={quoted_name} --format=JobID,JobName,State --noheader",
            check=False,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 3 and parts[1] == job_name:
                    state = parts[2].split("+", 1)[0].upper()
                    scheduler_id = self._normalize_slurm_id(parts[0])
                    matches.setdefault(scheduler_id, state)
        return sorted(
            matches.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else -1,
            reverse=True,
        )

    def _find_slurm_job_by_name(self, job_name: str) -> tuple[str, str] | None:
        """Return a job only when scheduler-name discovery is unambiguous."""
        matches = self._find_slurm_jobs_by_name(job_name)
        return matches[0] if len(matches) == 1 else None

    def _launch_reconciliation_candidates(
        self,
        metadata: dict[str, Any],
        job_id: str,
    ) -> list[tuple[str, str]]:
        """Exclude the pre-attempt and already failed IDs from fenced recovery."""
        excluded = {
            str(value)
            for value in metadata.get("failed_slurm_job_ids", [])
            if value
        }
        pre_launch = str(metadata.get("pre_launch_slurm_job_id", "") or "")
        if pre_launch:
            excluded.add(pre_launch)
        job_name = metadata.get("job_name") or self.get_job_name(job_id)
        return [
            candidate
            for candidate in self._find_slurm_jobs_by_name(job_name)
            if candidate[0] not in excluded
        ]

    @staticmethod
    def _normalize_slurm_id(slurm_job_id: str) -> str:
        return slurm_job_id.split(".", 1)[0].split("_", 1)[0]

    def _set_job_message(self, job_id: str, message: str) -> None:
        self._job_messages[job_id] = message
        metadata = self._jobs.get(job_id)
        if metadata is not None:
            metadata["message"] = message
            self._save_state()

    def _load_state(self) -> dict[str, dict[str, Any]]:
        lock_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_SH)
                if STATE_PATH.exists():
                    loaded = json.loads(STATE_PATH.read_text())
                    if isinstance(loaded, dict):
                        return loaded
        except Exception as exc:
            logger.warning("Failed to read SLURM state file %s: %s", STATE_PATH, exc)
        return {}

    def _save_state(self) -> None:
        lock_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".lock")
        tmp_path = STATE_PATH.with_name(f"{STATE_PATH.name}.{os.getpid()}.tmp")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                existing = {}
                if STATE_PATH.exists():
                    try:
                        loaded = json.loads(STATE_PATH.read_text())
                        if isinstance(loaded, dict):
                            existing = loaded
                    except Exception:
                        logger.warning("Ignoring unreadable SLURM state file during save: %s", STATE_PATH)
                existing.update(self._jobs)
                tmp_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
                os.replace(tmp_path, STATE_PATH)
        except Exception as exc:
            logger.warning("Failed to write SLURM state file %s: %s", STATE_PATH, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
