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

import json
import logging
import math
import os
import pwd
import re
import shlex
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tao_sdk.platforms.base import BaseHandler

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 5 * 60
MAX_JOB_RETRIES = 10
STATE_PATH = Path(os.path.expanduser("~/.tao_sdk_slurm_jobs.json"))
DEFAULT_JOB_DIR_TEMPLATE = "/lustre/fsw/portfolios/edgeai/users/{user}"
DEFAULT_PARTITION = "polar,polar3,polar4,grizzly"
DEFAULT_ACCOUNT = "edgeai_tao-ptm_image-foundation-model-clip"

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

        self._creds = creds or {}
        self.ssh_key_path = ssh_key_path or self._find_ssh_key()
        self.ssh_options = self._get_ssh_options()

        self.base_results_dir = (
            base_results_dir
            or DEFAULT_JOB_DIR_TEMPLATE.format(user=self.login_user)
        ).rstrip("/")
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

        self._jobs[job_id] = {
            "job_name": job_name,
            "job_dir": job_dir,
            "log_dir": log_dir,
            "remote_script": remote_script,
            "remote_entrypoint": remote_entrypoint,
            "remote_specs": remote_specs,
            "remote_env": remote_env,
            "remote_meta": remote_meta,
            "partition": partition,
            "allowed_nodes": allowed_nodes,
            "account": account,
            "hostname": self.login_hostname,
            "message": "Preparing container image for SLURM execution",
        }
        self._save_state()

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

        result = self._ssh(
            f"sbatch --export=ALL {shlex.quote(remote_script)}",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            self._set_job_message(job_id, f"SLURM sbatch failed: {detail}")
            raise RuntimeError(f"sbatch failed for {remote_script}: {detail}")

        sbatch_output = result.stdout.strip()
        slurm_job_id = self.get_slurm_job_id_from_output(sbatch_output)
        if not slurm_job_id:
            raise RuntimeError(
                f"Could not parse SLURM job id from sbatch output: {sbatch_output!r}"
            )

        self._jobs[job_id] = {
            "slurm_job_id": slurm_job_id,
            "failed_slurm_job_ids": [],
            "retry_count": 0,
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
            "message": (
                f"Job submitted to SLURM cluster "
                f"(Job ID: {slurm_job_id}). Waiting for resources..."
            ),
        }
        self._save_state()
        self._set_job_message(job_id, self._jobs[job_id]["message"])
        logger.info("Submitted SLURM job %s for TAO job %s", slurm_job_id, job_id)
        return sbatch_output

    def get_tao_job_status(self, job_id: str) -> str | None:
        """Get TAO SDK status by checking SLURM state."""
        metadata = self._get_job_metadata(job_id)
        if not metadata:
            self._set_job_message(
                job_id,
                "SLURM job metadata not found; submission may still be in progress",
            )
            return "Pending"

        slurm_job_id = metadata.get("slurm_job_id")
        if not slurm_job_id:
            self._set_job_message(job_id, "SLURM job id not recorded yet")
            return "Pending"

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
            if retry_count < MAX_JOB_RETRIES and self._is_retriable_failure(
                slurm_status,
                job_id,
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
            else:
                failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
                if slurm_job_id not in failed_ids:
                    failed_ids.append(slurm_job_id)
                    self._save_state()
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
        """Cancel a SLURM job using scancel."""
        metadata = self._get_job_metadata(job_id)
        if not metadata:
            return True
        slurm_job_id = metadata.get("slurm_job_id")
        if not slurm_job_id:
            return True

        slurm_status = self.get_slurm_job_status(slurm_job_id)
        if slurm_status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NOT_FOUND"):
            self._set_job_message(
                job_id,
                f"SLURM job {slurm_job_id} already terminated with status: {slurm_status}",
            )
            return True

        result = self._ssh(
            f"scancel {shlex.quote(str(slurm_job_id))}",
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            self._set_job_message(job_id, f"SLURM job {slurm_job_id} canceled")
            return True
        detail = (result.stderr or result.stdout or "").strip()
        if "Invalid job id" in detail or "does not exist" in detail:
            return True
        logger.warning("Failed to cancel SLURM job %s: %s", slurm_job_id, detail)
        self._set_job_message(job_id, f"Failed to cancel SLURM job {slurm_job_id}: {detail}")
        return False

    def delete_job(self, job_id: str) -> bool:
        return self.cancel_job(job_id)

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

        result = self._ssh(
            f"sbatch --export=ALL {shlex.quote(remote_script)}",
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return False, None, (result.stderr or result.stdout or "").strip()

        new_slurm_job_id = self.get_slurm_job_id_from_output(result.stdout)
        if not new_slurm_job_id:
            return False, None, f"could not parse sbatch output: {result.stdout!r}"

        failed_ids = metadata.setdefault("failed_slurm_job_ids", [])
        if failed_slurm_job_id and failed_slurm_job_id not in failed_ids:
            failed_ids.append(failed_slurm_job_id)
        metadata["retry_count"] = retry_count + 1
        metadata["slurm_job_id"] = new_slurm_job_id
        metadata["message"] = (
            f"Job resubmitted after infrastructure failure "
            f"(attempt {retry_count + 1}/{MAX_JOB_RETRIES}); "
            f"new SLURM Job ID: {new_slurm_job_id}"
        )
        self._save_state()
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
        """Convert a Docker image to SQSH using remote ``srun enroot import``."""
        self._ssh(f"mkdir -p {shlex.quote(sqsh_cache_dir)}", check=True)
        docker_image = self._to_enroot_image(image)

        srun_parts = [
            "srun",
            "-n1",
            f"-p {shlex.quote(partition)}",
            f"--mem={int(memory_gb)}G",
            f"-t {int(timeout_minutes)}:00",
        ]
        if account:
            srun_parts.append(f"-A {shlex.quote(account)}")
        srun_parts.extend([
            "enroot",
            "import",
            "-o",
            shlex.quote(sqsh_path),
            "--",
            shlex.quote(docker_image),
        ])

        start = time.time()
        logger.info(
            "Running remote SQSH conversion command on partition %s with %dGB memory",
            partition,
            memory_gb,
        )
        result = self._ssh(
            " ".join(srun_parts),
            check=False,
            timeout=timeout_minutes * 60 + 60,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            logger.warning(
                "SQSH conversion failed after %.1fs: %s",
                elapsed,
                (result.stderr or result.stdout or "").strip(),
            )
            return False
        if self._check_sqsh_exists(sqsh_path):
            logger.info("SQSH conversion completed in %.1fs: %s", elapsed, sqsh_path)
            return True
        logger.warning("SQSH conversion command completed but %s was not created", sqsh_path)
        return False

    def _get_sqsh_filename(self, image: str) -> str:
        return image.replace("/", "_").replace(":", "_") + ".sqsh"

    def _check_sqsh_exists(self, sqsh_path: str) -> bool:
        result = self._ssh(
            f"test -f {shlex.quote(sqsh_path)} && echo exists || echo not_found",
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
    ) -> subprocess.CompletedProcess:
        attempts = max(len(self.login_hostnames), 1)
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
            self._save_state()
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
            "message": f"SLURM job {slurm_job_id}: {slurm_status}",
        }

    def _find_slurm_job_by_name(self, job_name: str) -> tuple[str, str] | None:
        quoted_name = shlex.quote(job_name)
        result = self._ssh(
            f"squeue --name={quoted_name} --noheader --format='%i|%T|%j'",
            check=False,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 3 and parts[2] == job_name:
                    return self._normalize_slurm_id(parts[0]), parts[1].upper()

        result = self._ssh(
            f"sacct -nP -X --name={quoted_name} --format=JobID,JobName,State --noheader",
            check=False,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            matches: list[tuple[str, str]] = []
            for line in result.stdout.splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 3 and parts[1] == job_name:
                    state = parts[2].split("+", 1)[0].upper()
                    matches.append((self._normalize_slurm_id(parts[0]), state))
            if matches:
                return matches[-1]
        return None

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
        try:
            if STATE_PATH.exists():
                loaded = json.loads(STATE_PATH.read_text())
                if isinstance(loaded, dict):
                    return loaded
        except Exception as exc:
            logger.warning("Failed to read SLURM state file %s: %s", STATE_PATH, exc)
        return {}

    def _save_state(self) -> None:
        try:
            STATE_PATH.write_text(json.dumps(self._jobs, indent=2, sort_keys=True))
        except Exception as exc:
            logger.warning("Failed to write SLURM state file %s: %s", STATE_PATH, exc)
