# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Docker execution handler for TAO SDK.

This handler mirrors the Docker lifecycle shape from TAO Core's local Docker
backend while following the SDK handler principles used by Brev:

* Standalone SDK module with no TAO Core imports or MongoDB dependency.
* Lazily constructed from SDK credentials/environment.
* Uses SDK-baked ``entrypoint["command"]`` as the container command.
* Queries Docker directly for status/logs/cancellation by backend job name.
* Keeps platform/control-plane metadata in SDK state and Docker labels, not in
  the training container environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import platform
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tao_sdk.artifacts import validate_job_id
from tao_sdk.platforms.base import BaseHandler

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.path.expanduser("~/.tao_sdk_docker_jobs.json"))
DEFAULT_DOCKER_NETWORK = "tao_default"
DEFAULT_DOCKER_USERNAME = "$oauthtoken"


def _normalize_gpu_ids(gpu_ids: list | str | None) -> list[str]:
    """Normalize a GPU selection into a list of Docker device-id strings.

    Accepts ``[0, 1]``, ``["0", "1"]``, ``"0,1"``, ``"1"``, or a GPU UUID, and
    returns ``[]`` when nothing was requested. Docker's ``DeviceRequest``
    expects string device ids; integers are the natural thing for a caller to
    pass, so coerce rather than reject.
    """
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, str):
        candidates = gpu_ids.split(",")
    elif isinstance(gpu_ids, (list, tuple, set)):
        candidates = list(gpu_ids)
    else:
        candidates = [gpu_ids]

    normalized = []
    for candidate in candidates:
        text = str(candidate).strip()
        if text:
            normalized.append(text)
    return normalized


class DockerContainerIdentityError(RuntimeError):
    """A live container does not match the durable TAO job identity."""


class DockerHandler(BaseHandler):
    """Execution handler for single-node local Docker jobs."""

    backend_type = "local-docker"

    def __init__(
        self,
        creds: dict[str, Any] | None = None,
        docker_host: str | None = None,
        docker_network: str | None = None,
        docker_username: str | None = None,
        ngc_key: str | None = None,
        pull_policy: str | None = None,
        auto_remove: bool | None = None,
        client: Any | None = None,
        docker_module: Any | None = None,
        validate: bool = True,
    ):
        self._creds = creds or {}
        self.docker_host = docker_host or self._config_value("DOCKER_HOST")
        self.docker_network = (
            docker_network
            or self._config_value("DOCKER_NETWORK")
            or DEFAULT_DOCKER_NETWORK
        )
        self.docker_username = (
            docker_username
            or self._config_value("DOCKER_USERNAME")
            or DEFAULT_DOCKER_USERNAME
        )
        self.ngc_key = ngc_key or self._config_value("NGC_KEY")
        self.pull_policy = (
            pull_policy
            or self._config_value("DOCKER_PULL_POLICY")
            or "missing"
        ).lower()
        self.auto_remove = (
            auto_remove
            if auto_remove is not None
            else self._bool_config("DOCKER_AUTO_REMOVE", False)
        )

        self._docker = docker_module or self._import_docker()
        self._client = client or self._make_client()
        self._job_messages: dict[str, str] = {}
        self._jobs = self._load_state()

        if validate:
            self._ping()

        logger.info(
            "DockerHandler initialized for host=%s network=%s",
            self.docker_host or "default",
            self.docker_network,
        )

    @classmethod
    def from_credentials(cls, creds: dict) -> "DockerHandler":
        """Construct a DockerHandler from SDK credentials/environment."""
        try:
            return cls(creds=creds or {})
        except Exception as e:
            from tao_sdk.models import CredentialError

            raise CredentialError(
                "Failed to initialize local Docker backend. Ensure the Docker "
                "Python package is installed, the Docker daemon is reachable, "
                "and DOCKER_HOST is correct if you are not using the default "
                f"local socket. Original error: {e}"
            ) from e

    def get_job_name(self, job_id: str) -> str:
        """Generate a Docker-safe backend job/container name."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(job_id)).strip("-")
        return f"tao-job-{safe}"[:128]

    def get_backend_identity(self) -> str:
        """Return a non-secret durable identity for the configured daemon."""
        try:
            daemon_id = str((self._client.info() or {}).get("ID", "")).strip()
        except Exception as exc:
            logger.warning("Could not verify Docker daemon identity: %s", exc)
            return ""
        if not daemon_id:
            logger.warning("Docker daemon did not report a stable identity")
            return ""
        endpoint = self._normalized_docker_endpoint()
        target = json.dumps(
            {"daemon_id": daemon_id, "endpoint": endpoint},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        return f"docker-sha256:{digest}"

    def get_runtime_identity(
        self,
        job_id: str,
        *,
        container_id: str = "",
        launch_uncertain: bool = True,
    ) -> dict:
        """Build the durable coordinates needed to recover a Docker job."""
        return {
            "container_name": self.get_job_name(job_id),
            "container_id": str(container_id or ""),
            "launch_uncertain": bool(launch_uncertain),
        }

    def restore_job_runtime(
        self,
        job_id: str,
        runtime: dict | None,
        *,
        status: str | None = None,
    ) -> bool:
        """Restore validated JobStore runtime identity into the handler cache."""
        if not isinstance(runtime, dict):
            self._jobs.pop(job_id, None)
            return False
        container_id = runtime.get("container_id", "")
        launch_uncertain = runtime.get("launch_uncertain")
        expected = self.get_runtime_identity(
            job_id,
            container_id=container_id if isinstance(container_id, str) else "",
            launch_uncertain=(
                launch_uncertain if isinstance(launch_uncertain, bool) else True
            ),
        )
        if (
            runtime.get("container_name") != expected["container_name"]
            or not isinstance(container_id, str)
            or not isinstance(launch_uncertain, bool)
        ):
            self._jobs.pop(job_id, None)
            return False
        state = dict(self._jobs.get(job_id, {}))
        state.update({"job_id": job_id, **expected})
        if status is not None:
            state["status"] = str(status)
        self._jobs[job_id] = state
        return True

    def get_job_runtime_identity(self, job_id: str) -> dict | None:
        """Return durable runtime coordinates currently held in the cache."""
        state = self._jobs.get(job_id)
        if not state:
            return None
        return self.get_runtime_identity(
            job_id,
            container_id=str(state.get("container_id", "")),
            launch_uncertain=bool(state.get("launch_uncertain", False)),
        )

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
        mounts: list | None = None,
        shm_size: int | str | None = None,
        run_as_user: bool | None = None,
        container_user: str | int | None = None,
        gpu_ids: list | str | None = None,
        **kwargs,
    ) -> dict:
        """Create a local Docker container for an SDK job.

        ``run_as_user`` is tri-state. ``None`` automatically maps local jobs
        with a writable ``/results`` bind mount to the invoking UID/GID, ``True``
        requests that mapping explicitly, and ``False`` preserves the image's
        configured user. ``container_user`` is a Docker-native explicit user
        override (for example ``"1000:1000"``) and takes precedence.

        Automatic UID/GID mapping is intentionally disabled for remote Docker
        daemons because IDs from the SDK client host need not exist on the
        Docker host. A remote writable ``/results`` bind therefore requires an
        explicit ``container_user`` or reviewed ``run_as_user=False`` opt-out.
        """
        if num_nodes and int(num_nodes) != 1:
            raise ValueError("local-docker supports single-node jobs only")
        if not entrypoint or "command" not in entrypoint:
            raise ValueError(
                "entrypoint with 'command' key is required. "
                "Ensure the model's config.json has an 'actions' section."
            )

        job_id = job_id or str(int(time.time()))
        container_name = self.get_job_name(job_id)
        cloud_metadata = cloud_metadata or {}
        specs = specs or {}
        docker_env_vars = dict(docker_env_vars or {})

        volumes = self._build_volumes(mounts)
        resolved_user = self._resolve_container_user(
            volumes=volumes,
            run_as_user=run_as_user,
            container_user=container_user,
        )
        self._prepare_local_writable_bind_sources(volumes)
        supplementary_groups = self._supplementary_groups(resolved_user)
        num_gpus = self._resolve_num_gpus(resource_shape, kwargs)
        command = ["/bin/bash", "-c", entrypoint["command"]]
        env = self._build_environment(
            docker_env_vars=docker_env_vars,
            cloud_metadata=cloud_metadata,
            job_id=job_id,
        )
        if resolved_user and self._is_non_root_container_user(resolved_user):
            self._add_non_root_environment_defaults(
                env, resolved_user, volumes, job_id=job_id
            )

        if dedicated_node_group:
            logger.warning(
                "local-docker ignores dedicated_node_group=%s", dedicated_node_group
            )
        if kwargs.get("allowed_nodes"):
            logger.warning("local-docker ignores allowed_nodes=%s", kwargs["allowed_nodes"])

        self._update_job_status(job_id, "Pending", f"Preparing Docker image {image}")
        self._ensure_image(image, job_id)
        self._remove_existing_container(container_name, job_id=job_id)
        self._ensure_network(job_id)

        run_kwargs = {
            "image": image,
            "command": command,
            "name": container_name,
            "network": self.docker_network,
            "detach": True,
            "auto_remove": self.auto_remove,
            "environment": env,
            "volumes": volumes,
            "labels": {
                "tao-toolkit": "true",
                "tao-sdk-job-id": job_id,
                "tao-sdk-backend": self.backend_type,
                "tao-sdk-network": network or "",
                "tao-sdk-action": action or "",
            },
        }
        if resolved_user:
            run_kwargs["user"] = resolved_user
        if supplementary_groups:
            run_kwargs["group_add"] = supplementary_groups
        if shm_size is not None:
            run_kwargs["shm_size"] = shm_size
        else:
            run_kwargs["tmpfs"] = {"/dev/shm": ""}
        self._add_gpu_run_kwargs(run_kwargs, env, num_gpus, gpu_ids)

        logger.info("Starting Docker container %s with image %s", container_name, image)
        # Persist the deterministic container identity before sending the
        # create request.  If the transport times out, Docker may still finish
        # the request after the client has observed the exception.  An
        # immediate NotFound response is therefore not proof that no writer
        # can appear under this name.
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._jobs[job_id] = {
            "job_id": job_id,
            "container_name": container_name,
            "container_id": "",
            "image": image,
            "network": network,
            "action": action,
            "status": "Pending",
            "num_gpus": num_gpus,
            "created_at": created_at,
            "submission_attempted": True,
            "launch_uncertain": True,
        }
        self._update_job_status(
            job_id,
            "Pending",
            f"Submitting Docker container {container_name}",
        )
        try:
            container = self._client.containers.run(**run_kwargs)
        except BaseException as exc:
            self._update_job_status(
                job_id,
                "Pending",
                f"Docker container {container_name} launch outcome is uncertain: {exc}",
            )
            raise
        container_id = getattr(container, "id", "")
        self._jobs[job_id] = {
            "job_id": job_id,
            "container_name": container_name,
            "container_id": container_id,
            "image": image,
            "network": network,
            "action": action,
            "status": "Running",
            "num_gpus": num_gpus,
            "created_at": created_at,
            "submission_attempted": True,
            "launch_uncertain": False,
        }
        self._save_state()
        self._update_job_status(
            job_id,
            "Running",
            f"Docker container {container_name} started",
        )
        return self._jobs[job_id]

    def get_tao_job_status(self, job_id: str) -> str | None:
        """Map Docker container state to SDK status strings."""
        container_name = self.get_job_name(job_id)
        try:
            container = self._get_owned_container(job_id)
            container.reload()
        except self._not_found_errors:
            state = self._jobs.get(job_id, {})
            state_status = state.get("status")
            recovered_status = self._recover_removed_container_status(job_id)
            if recovered_status == "Complete":
                return None
            if recovered_status in {"Error", "Canceled"}:
                return recovered_status
            if state.get("launch_uncertain"):
                message = (
                    f"Waiting for Docker to resolve the launch of {container_name}"
                )
                self._job_messages[job_id] = message
                logger.warning("Docker job %s: %s", job_id, message)
                return "Pending"
            if not state.get("submission_attempted") and not state.get("container_id"):
                self._update_job_status(
                    job_id,
                    "Error",
                    f"Docker launch failed before container {container_name} was submitted",
                )
                return "Error"
            if state_status == "Canceled":
                return "Canceled"
            if state_status == "Complete":
                return None
            if state_status == "Error":
                return "Error"
            message = (
                f"Docker container {container_name} is absent without verified "
                "terminal evidence"
            )
            self._job_messages[job_id] = message
            logger.warning("Docker job %s: %s", job_id, message)
            return "Pending"
        except DockerContainerIdentityError as exc:
            self._job_messages[job_id] = str(exc)
            logger.error("Refusing Docker status lookup for %s: %s", job_id, exc)
            return "Pending"
        except Exception as e:
            logger.error("Failed to inspect Docker job %s: %s", job_id, e)
            # A daemon/transport failure is not evidence that the container
            # stopped. Keep the job non-terminal so artifact cleanup cannot
            # race a writer that may still be active.
            self._job_messages[job_id] = f"Docker status unavailable: {e}"
            return "Running"

        state = self._jobs.setdefault(job_id, {"job_id": job_id})
        if state.get("launch_uncertain"):
            state["launch_uncertain"] = False
            state["container_name"] = container_name
            state["container_id"] = getattr(container, "id", "")
            self._save_state()

        status = str(getattr(container, "status", "") or "").lower()
        if status in ("created",):
            self._update_job_status(job_id, "Pending", f"Docker container {status}")
            return "Pending"
        if status in ("running", "restarting"):
            self._update_job_status(job_id, "Running", f"Docker container {status}")
            return "Running"
        if status == "paused":
            self._update_job_status(job_id, "Paused", "Docker container is paused")
            return "Paused"
        if status == "removing":
            recovered_status = self._recover_removed_container_status(job_id)
            if recovered_status == "Complete":
                return None
            if recovered_status in {"Error", "Canceled"}:
                return recovered_status
            message = (
                "Docker container is removing; waiting for exact exit evidence"
            )
            self._job_messages[job_id] = message
            return "Pending"
        if status in ("exited", "dead"):
            exit_code = self._container_exit_code(container)
            if exit_code == 0:
                self._update_job_status(
                    job_id,
                    "Complete",
                    f"Docker container exited successfully with code {exit_code}",
                    exit_code=exit_code,
                )
                return None
            self._update_job_status(
                job_id,
                "Error",
                f"Docker container exited with code {exit_code}",
                exit_code=exit_code,
            )
            return "Error"

        self._update_job_status(
            job_id,
            "Pending",
            f"Docker container status is {status or 'unknown'}",
        )
        return "Pending"

    def normalize_status(self, raw_status: str | None) -> str:
        """Map Docker's complete sentinel to SDK status."""
        if raw_status is None:
            return "Complete"
        return raw_status

    def get_job_message(self, job_id: str) -> str:
        """Get the latest status message for a Docker job."""
        state = self._jobs.get(job_id, {})
        return self._job_messages.get(job_id, "") or state.get("message", "")

    def get_job_logs(self, job_id: str, tail_lines: int | None = None) -> str | None:
        """Fetch container logs directly from Docker."""
        container_name = self.get_job_name(job_id)
        try:
            container = self._get_owned_container(job_id)
            kwargs = {"stdout": True, "stderr": True, "timestamps": False}
            if tail_lines is not None:
                kwargs["tail"] = int(tail_lines)
            logs = container.logs(**kwargs)
            if isinstance(logs, bytes):
                return logs.decode("utf-8", errors="replace")
            return str(logs) if logs is not None else None
        except self._not_found_errors + (DockerContainerIdentityError,):
            logger.warning("Docker container %s not found for logs", container_name)
            return None
        except Exception as e:
            logger.error("Failed to get Docker logs for %s: %s", job_id, e)
            return None

    def delete_job(self, job_id: str) -> bool:
        """Stop and remove a Docker container."""
        container_name = self.get_job_name(job_id)
        try:
            container = self._get_owned_container(job_id)
        except self._not_found_errors:
            state = self._jobs.get(job_id, {})
            recovered_status = self._recover_removed_container_status(job_id)
            if (
                not state.get("submission_attempted")
                and not state.get("container_id")
            ):
                self._update_job_status(
                    job_id,
                    "Error",
                    f"Docker launch failed before container {container_name} "
                    "was submitted",
                )
                return True
            if state.get("launch_uncertain"):
                message = (
                    f"Docker container {container_name} is not visible, but its "
                    "launch outcome is still uncertain"
                )
                self._job_messages[job_id] = message
                logger.warning("Cannot confirm cancellation of %s: %s", job_id, message)
                return False
            recorded_status = recovered_status or str(state.get("status", ""))
            if not state.get("container_id"):
                message = (
                    f"Docker container {container_name} is absent but no bound "
                    "container ID proves which writer was removed"
                )
                self._job_messages[job_id] = message
                logger.warning("Cannot confirm cancellation of %s: %s", job_id, message)
                return False
            if recorded_status not in {"Complete", "Error", "Canceled"}:
                message = (
                    f"Docker container {container_name} is absent without exact "
                    "terminal exit evidence"
                )
                self._job_messages[job_id] = message
                logger.warning("Cannot confirm cancellation of %s: %s", job_id, message)
                return False
            final_status = recorded_status
            message = (
                f"Docker container {container_name} already absent after "
                f"terminal status {final_status}"
                if final_status in {"Complete", "Error"}
                else f"Docker container {container_name} already absent"
            )
            self._update_job_status(
                job_id,
                final_status,
                message,
            )
            return True
        except DockerContainerIdentityError as exc:
            self._job_messages[job_id] = str(exc)
            logger.error("Refusing Docker cancellation for %s: %s", job_id, exc)
            return False
        except Exception as e:
            logger.error("Failed to fetch Docker container %s: %s", container_name, e)
            return False

        state = self._jobs.setdefault(job_id, {"job_id": job_id})
        if state.get("launch_uncertain"):
            state["launch_uncertain"] = False
            state["container_name"] = container_name
            state["container_id"] = getattr(container, "id", "")
            self._save_state()

        try:
            container.reload()
        except self._not_found_errors:
            return self._finalize_disappeared_container(job_id)
        except Exception as exc:
            logger.error(
                "Failed to reload Docker container %s: %s", container_name, exc
            )
            return False

        container_status = str(getattr(container, "status", "")).lower()
        recorded_status = str(state.get("status", ""))
        final_status = (
            recorded_status
            if recorded_status in {"Complete", "Error", "Canceled"}
            else "Canceled"
        )
        exit_code = None
        if container_status in ("exited", "dead"):
            exit_code = self._container_exit_code(container)
            if recorded_status not in {"Complete", "Error", "Canceled"}:
                final_status = "Complete" if exit_code == 0 else "Error"
        elif container_status in (
            "running",
            "restarting",
            "paused",
            "created",
        ):
            # Persist the user's intent before stop: with auto_remove enabled,
            # the daemon may delete the exact container before the next API
            # call or before this process can persist a terminal result.
            final_status = "Canceled"
            state["cancel_requested"] = True
            self._save_state()
            try:
                container.stop(timeout=10)
            except self._not_found_errors:
                return self._finalize_disappeared_container(job_id)
            except Exception as exc:
                logger.error(
                    "Failed to stop Docker container %s: %s", container_name, exc
                )
                return False
        elif container_status == "removing":
            recovered_status = self._recover_removed_container_status(job_id)
            if recovered_status not in {"Complete", "Error", "Canceled"}:
                self._job_messages[job_id] = (
                    "Docker container is removing without exact exit evidence"
                )
                return False
            final_status = recovered_status
            exit_code = state.get("exit_code")

        try:
            container.remove(force=True)
        except self._not_found_errors:
            return self._finalize_disappeared_container(
                job_id,
                observed_status=final_status,
                observed_exit_code=exit_code,
            )
        except Exception as exc:
            logger.error(
                "Failed to delete Docker container %s: %s", container_name, exc
            )
            logger.error(traceback.format_exc())
            return False
        message = (
            f"Docker container {container_name} was already terminal with "
            f"status {final_status} and was removed"
            if final_status in {"Complete", "Error"}
            else f"Docker container {container_name} stopped and removed"
        )
        self._update_job_status(
            job_id,
            final_status,
            message,
            exit_code=exit_code,
        )
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a Docker job."""
        return self.delete_job(job_id)

    def get_job_replicas(self, job_id: str) -> list[dict]:
        """Return single-container replica details for the job."""
        container_name = self.get_job_name(job_id)
        try:
            container = self._get_owned_container(job_id)
            container.reload()
            attrs = getattr(container, "attrs", {}) or {}
            state = attrs.get("State", {})
            return [{
                "name": getattr(container, "name", container_name),
                "id": getattr(container, "id", ""),
                "status": getattr(container, "status", ""),
                "image": self._container_image(container),
                "exit_code": state.get("ExitCode"),
                "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
                "node": "local-docker",
            }]
        except self._not_found_errors + (DockerContainerIdentityError,):
            return []
        except Exception as e:
            logger.error("Failed to get Docker replica details for %s: %s", job_id, e)
            return []

    def get_available_instances(self, **kwargs) -> dict:
        """Return best-effort local Docker compute capacity."""
        gpu_info = self._query_gpus()
        docker_info = {}
        try:
            docker_info = self._client.info()
        except Exception as e:
            logger.warning("Failed to query Docker info: %s", e)

        gpu_count = len(gpu_info)
        cpu_cores = docker_info.get("NCPU")
        system_memory = docker_info.get("MemTotal")
        gpu_memory = gpu_info[0].get("memory_mb") if gpu_info else None
        instance_type = gpu_info[0].get("name") if gpu_info else "CPU"

        instances: dict[str, dict] = {}
        max_shape = gpu_count if gpu_count > 0 else 0
        for count in range(0, max_shape + 1):
            platform_id = f"local-docker-gpu-{count}"
            instances[platform_id] = {
                "cluster": "local-docker",
                "gpu_type": f"gpu.{count}",
                "instance_type": instance_type if count else "CPU",
                "gpu_count": count,
                "cpu_cores": cpu_cores,
                "system_memory": system_memory,
                "gpu_memory": gpu_memory if count else None,
                "backend_type": self.backend_type,
            }
        return instances

    def resolve_resource_shape(
        self,
        num_gpus: int,
        preferred_node_group: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Carry the requested local GPU count through resource_shape."""
        requested = max(int(num_gpus or 0), 0)
        try:
            available = max(
                (i.get("gpu_count", 0) for i in self.get_available_instances().values()),
                default=0,
            )
            if requested > available:
                logger.warning(
                    "Requested %d GPU(s) but local Docker detected %d GPU(s)",
                    requested,
                    available,
                )
        except Exception:
            logger.debug("Could not validate local Docker GPU availability", exc_info=True)
        return f"gpu.{requested}", preferred_node_group

    def resolve_cpu_shape(
        self,
        preferred_node_group: str | None = None,
        preference: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Local Docker CPU jobs use the same host with gpu.0 resource shape."""
        return "gpu.0", preferred_node_group

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    @property
    def _not_found_errors(self):
        return (getattr(self._docker.errors, "NotFound", Exception),)

    def _config_value(self, key: str) -> str:
        return str(self._creds.get(key) or os.environ.get(key) or "")

    def _bool_config(self, key: str, default: bool = False) -> bool:
        val = self._config_value(key)
        if not val:
            return default
        return val.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _import_docker():
        try:
            return importlib.import_module("docker")
        except ImportError as e:
            raise ImportError(
                "Docker Python package is required for local-docker. "
                "Install it with `pip install docker` or `pip install tao-sdk[docker]`."
            ) from e

    def _make_client(self):
        if self.docker_host:
            return self._docker.DockerClient(base_url=self.docker_host)
        return self._docker.from_env()

    def _normalized_docker_endpoint(self) -> str:
        """Normalize equivalent Docker endpoints before hashing their identity."""
        value = str(self.docker_host or "").strip()
        if not value:
            if os.name == "nt":
                return "npipe:////./pipe/docker_engine"
            default_socket = Path("/var/run/docker.sock").resolve(strict=False)
            return f"unix://{default_socket}"

        if value.startswith("/"):
            return f"unix://{Path(value).expanduser().resolve(strict=False)}"

        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if scheme == "unix":
            socket_path = parsed.path or parsed.netloc
            return f"unix://{Path(socket_path).expanduser().resolve(strict=False)}"
        if scheme == "npipe":
            return f"npipe://{(parsed.netloc + parsed.path).lower()}"

        # Do not persist passwords embedded in an SSH/TCP URL. The username,
        # host, port, and path are sufficient to distinguish daemon targets.
        hostname = (parsed.hostname or "").lower()
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        username = parsed.username or ""
        user_prefix = f"{username}@" if username else ""
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{user_prefix}{hostname}{port}"
        path = parsed.path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))

    def _ping(self):
        try:
            self._client.ping()
        except AttributeError:
            pass

    def _ensure_network(self, job_id: str):
        """Ensure the configured Docker network exists before container launch."""
        if not self.docker_network:
            return

        try:
            self._client.networks.get(self.docker_network)
            return
        except AttributeError:
            logger.debug("Docker client has no networks API; skipping network precheck")
            return
        except self._not_found_errors:
            pass
        except Exception as e:
            raise RuntimeError(
                f"Failed to inspect Docker network {self.docker_network!r}: {e}"
            ) from e

        self._update_job_status(
            job_id,
            "Pending",
            f"Creating Docker network {self.docker_network}",
        )
        try:
            self._client.networks.create(self.docker_network)
        except Exception as e:
            # If a concurrent launcher created it after our miss, treat that as
            # success. Otherwise surface a clear pre-launch failure.
            try:
                self._client.networks.get(self.docker_network)
                return
            except Exception:
                raise RuntimeError(
                    f"Docker network {self.docker_network!r} does not exist "
                    f"and could not be created: {e}"
                ) from e

    def _ensure_image(self, image: str, job_id: str):
        if self.pull_policy == "never":
            return
        if self.pull_policy not in ("missing", "always"):
            logger.warning("Unknown DOCKER_PULL_POLICY=%s; using 'missing'", self.pull_policy)
            self.pull_policy = "missing"

        image_exists = self._image_exists(image)
        if image_exists and self.pull_policy == "missing":
            self._update_job_status(job_id, "Pending", f"Docker image {image} already exists")
            return

        self._login_if_needed(image)
        self._update_job_status(job_id, "Pending", f"Pulling Docker image {image}")
        repository, tag = self._split_image_name(image)
        if tag:
            self._client.images.pull(repository, tag=tag)
        else:
            self._client.images.pull(repository)
        self._update_job_status(job_id, "Pending", f"Docker image {image} ready")

    def _image_exists(self, image: str) -> bool:
        try:
            self._client.images.get(image)
            return True
        except self._not_found_errors:
            return False
        except Exception as e:
            logger.warning("Failed to inspect Docker image %s: %s", image, e)
            return False

    def _login_if_needed(self, image: str):
        registry, _, _ = self._parse_image(image)
        if registry != "nvcr.io" or not self.ngc_key:
            return
        try:
            self._client.login(
                username=self.docker_username,
                password=self.ngc_key,
                registry=registry,
            )
            logger.info("Logged in to Docker registry %s", registry)
        except Exception as e:
            logger.warning("Docker login to %s failed: %s", registry, e)

    def _remove_existing_container(self, container_name: str, *, job_id: str):
        try:
            existing = self._client.containers.get(container_name)
        except self._not_found_errors:
            return
        if not self._container_identity_matches(existing, job_id):
            error = DockerContainerIdentityError(
                f"refusing to replace container {container_name}: its TAO labels "
                f"do not identify job {job_id}"
            )
            error.writer_quiescent = True
            raise error
        try:
            existing.stop(timeout=5)
        except Exception:
            logger.debug("Existing container %s was not running", container_name)
        try:
            existing.remove(force=True)
        except Exception as e:
            logger.warning("Failed to remove existing container %s: %s", container_name, e)

    def _get_owned_container(self, job_id: str):
        """Return only the container bound to this job's ID and labels.

        A deterministic name is sufficient only while a launch is unresolved.
        Once a container ID is recorded, a different container reusing the same
        name must never be inspected, stopped, or removed on this job's behalf.
        """
        state = self._jobs.get(job_id, {})
        container_name = self.get_job_name(job_id)
        recorded_id = str(state.get("container_id", "") or "")
        lookup = recorded_id or container_name
        try:
            container = self._client.containers.get(lookup)
        except self._not_found_errors as original:
            if not recorded_id:
                raise
            try:
                replacement = self._client.containers.get(container_name)
            except self._not_found_errors as replacement_error:
                raise original from replacement_error
            replacement_id = str(getattr(replacement, "id", "") or "")
            replacement_description = replacement_id or "an unidentified container"
            raise DockerContainerIdentityError(
                f"container name {container_name} now resolves to "
                f"{replacement_description}, not recorded container {recorded_id}"
            ) from original

        if not self._container_identity_matches(
            container,
            job_id,
            expected_id=recorded_id,
        ):
            raise DockerContainerIdentityError(
                f"Docker container returned for job {job_id} does not match its "
                "recorded ID, name, and TAO labels"
            )

        if not recorded_id:
            state = self._jobs.setdefault(job_id, {"job_id": job_id})
            state["container_name"] = container_name
            state["container_id"] = str(getattr(container, "id", "") or "")
            state["launch_uncertain"] = False
            self._save_state()
        return container

    def _container_identity_matches(
        self,
        container,
        job_id: str,
        *,
        expected_id: str = "",
    ) -> bool:
        container_id = str(getattr(container, "id", "") or "")
        if not container_id or (expected_id and container_id != expected_id):
            return False
        container_name = getattr(container, "name", "")
        if isinstance(container_name, str) and container_name:
            if container_name.lstrip("/") != self.get_job_name(job_id):
                return False
        labels = self._container_labels(container)
        return (
            labels.get("tao-sdk-job-id") == job_id
            and labels.get("tao-sdk-backend") == self.backend_type
        )

    @staticmethod
    def _container_labels(container) -> dict[str, str]:
        labels = getattr(container, "labels", None)
        if isinstance(labels, dict):
            return {str(key): str(value) for key, value in labels.items()}
        attrs = getattr(container, "attrs", {}) or {}
        if not isinstance(attrs, dict):
            return {}
        labels = (attrs.get("Config", {}) or {}).get("Labels", {})
        if not isinstance(labels, dict):
            return {}
        return {str(key): str(value) for key, value in labels.items()}

    def _recover_removed_container_status(self, job_id: str) -> str | None:
        """Recover an auto-removed container's exact exit result from events."""
        state = self._jobs.get(job_id, {})
        recorded = str(state.get("status", "") or "")
        if recorded in {"Complete", "Error", "Canceled"}:
            return recorded
        recovered_exit_code = self._recover_removed_container_exit_code(job_id)
        if recovered_exit_code is None:
            return None
        status = (
            "Canceled"
            if state.get("cancel_requested") is True
            else ("Complete" if recovered_exit_code == 0 else "Error")
        )
        message = (
            f"Docker cancellation completed after exit code {recovered_exit_code}"
            if status == "Canceled"
            else f"Docker container auto-removed after exit code {recovered_exit_code}"
        )
        self._update_job_status(
            job_id,
            status,
            message,
            exit_code=recovered_exit_code,
        )
        return status

    def _recover_removed_container_exit_code(self, job_id: str) -> int | None:
        """Read an exact, label-bound die event for the recorded container ID."""
        state = self._jobs.get(job_id, {})
        container_id = str(state.get("container_id", "") or "")
        if not container_id:
            return None
        try:
            events = self._client.events(
                since=0,
                until=int(time.time()),
                filters={"container": container_id, "event": "die"},
                decode=True,
            )
            recovered_exit_code = None
            for event in events:
                if not isinstance(event, dict):
                    continue
                actor = event.get("Actor", {}) or {}
                actor_id = str(actor.get("ID", "") or event.get("id", "") or "")
                attributes = actor.get("Attributes", {}) or {}
                if (
                    actor_id != container_id
                    or not isinstance(attributes, dict)
                    or str(attributes.get("tao-sdk-job-id", "")) != job_id
                    or str(attributes.get("tao-sdk-backend", "")) != self.backend_type
                ):
                    continue
                action = str(event.get("Action", "") or event.get("status", "")).lower()
                if action != "die":
                    continue
                try:
                    recovered_exit_code = int(attributes.get("exitCode"))
                except (TypeError, ValueError):
                    continue
        except Exception as exc:
            logger.warning(
                "Could not recover Docker exit event for removed job %s: %s",
                job_id,
                exc,
            )
            return None
        if recovered_exit_code is None:
            return None
        return recovered_exit_code

    def _exact_container_is_absent(self, job_id: str) -> bool:
        """Prove the bound ID is absent without accepting a name replacement."""
        state = self._jobs.get(job_id, {})
        if not state.get("container_id") or state.get("launch_uncertain"):
            return False
        try:
            self._get_owned_container(job_id)
        except self._not_found_errors:
            return True
        except DockerContainerIdentityError as exc:
            self._job_messages[job_id] = str(exc)
            return False
        except Exception as exc:
            logger.warning(
                "Could not verify Docker container absence for %s: %s",
                job_id,
                exc,
            )
            return False
        return False

    def _finalize_disappeared_container(
        self,
        job_id: str,
        *,
        observed_status: str | None = None,
        observed_exit_code: int | None = None,
    ) -> bool:
        """Finish a stop/remove race only with exact exit and absence proof."""
        if not self._exact_container_is_absent(job_id):
            return False
        state = self._jobs.get(job_id, {})
        recorded = str(state.get("status", "") or "")
        exit_code = (
            observed_exit_code
            if isinstance(observed_exit_code, int)
            and not isinstance(observed_exit_code, bool)
            else None
        )
        final_status = (
            observed_status
            if observed_status in {"Complete", "Error", "Canceled"}
            and exit_code is not None
            else None
        )
        if final_status is None:
            recovered = self._recover_removed_container_status(job_id)
            if recovered in {"Complete", "Error", "Canceled"}:
                final_status = recovered
                recovered_exit = state.get("exit_code")
                if (
                    isinstance(recovered_exit, int)
                    and not isinstance(recovered_exit, bool)
                ):
                    exit_code = recovered_exit
        if final_status is None and recorded in {"Complete", "Error", "Canceled"}:
            final_status = recorded
        if final_status is None:
            self._job_messages[job_id] = (
                "Docker container disappeared without exact terminal exit evidence"
            )
            return False
        self._update_job_status(
            job_id,
            final_status,
            f"Docker container {self.get_job_name(job_id)} is exactly absent "
            f"after terminal status {final_status}",
            exit_code=exit_code,
        )
        return True

    def _build_environment(
        self,
        docker_env_vars: dict[str, Any],
        cloud_metadata: dict,
        job_id: str,
    ) -> dict[str, str]:
        env = {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env["CLOUD_METADATA"] = json.dumps(cloud_metadata, separators=(",", ":"))
        for key, value in docker_env_vars.items():
            if value is not None:
                env[str(key)] = str(value)
        return env

    @staticmethod
    def _mount_mode_is_writable(mode: str) -> bool:
        options = {item.strip().lower() for item in mode.split(",")}
        return "ro" not in options

    def _local_writable_bind_sources(self, volumes: dict | None) -> list[Path]:
        if self._docker_host_is_remote() or not volumes:
            return []
        sources = []
        for source, config in volumes.items():
            path = Path(source)
            mode = str(config.get("mode", "rw"))
            if path.is_absolute() and self._mount_mode_is_writable(mode):
                sources.append(path)
        return sources

    def _has_writable_results_bind(self, volumes: dict | None) -> bool:
        if not volumes:
            return False
        return any(
            Path(source).is_absolute()
            and str(config.get("bind", "")).rstrip("/") == "/results"
            and self._mount_mode_is_writable(str(config.get("mode", "rw")))
            for source, config in volumes.items()
        )

    def _has_local_writable_results_bind(self, volumes: dict | None) -> bool:
        return (
            not self._docker_host_is_remote()
            and self._has_writable_results_bind(volumes)
        )

    def _prepare_local_writable_bind_sources(self, volumes: dict | None) -> None:
        """Create missing local writable bind roots before Docker can do so as root."""
        for source in self._local_writable_bind_sources(volumes):
            if source.exists():
                continue
            try:
                source.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise RuntimeError(
                    f"Could not create writable Docker bind source {source}: {e}"
                ) from e

    def _docker_host_is_remote(self) -> bool:
        value = str(self.docker_host or "").strip().lower()
        if not value:
            return False
        return not (
            value.startswith("unix://")
            or value.startswith("npipe://")
            or value.startswith("/")
        )

    @staticmethod
    def _invoking_user() -> str | None:
        getuid = getattr(os, "getuid", None)
        getgid = getattr(os, "getgid", None)
        if not callable(getuid) or not callable(getgid):
            return None
        return f"{getuid()}:{getgid()}"

    def _resolve_container_user(
        self,
        volumes: dict | None,
        run_as_user: bool | None,
        container_user: str | int | None,
    ) -> str | None:
        if run_as_user is not None and not isinstance(run_as_user, bool):
            raise TypeError("run_as_user must be True, False, or None")

        if (
            self._has_local_writable_results_bind(volumes)
            and (container_user is not None or run_as_user is not False)
            and self._daemon_uses_user_namespace()
        ):
            raise RuntimeError(
                "Docker rootless/userns-remap mode cannot guarantee that a "
                "container UID:GID maps to the submitting host user for a "
                "writable /results bind. Use an ordinary rootful daemon, or "
                "set run_as_user=False only for a reviewed image whose host "
                "artifact ownership is managed externally."
            )

        if container_user is not None:
            resolved = str(container_user).strip()
            if not resolved:
                raise ValueError("container_user must not be empty")
            if (
                self._has_writable_results_bind(volumes)
                and not self._is_non_root_container_user(resolved)
            ):
                raise ValueError(
                    "A writable /results bind cannot use an explicit root "
                    "container_user; provide the submitting non-root UID:GID, "
                    "or set run_as_user=False only for a reviewed root-required "
                    "image whose artifacts will be managed externally"
                )
            return resolved

        if run_as_user is False:
            return None

        remote = self._docker_host_is_remote()
        if remote:
            if run_as_user is True or (
                run_as_user is None and self._has_writable_results_bind(volumes)
            ):
                raise ValueError(
                    "Cannot infer UID/GID for a remote Docker host with a "
                    "writable /results bind; provide container_user explicitly "
                    "or set run_as_user=False for a reviewed root-required image"
                )
            logger.info(
                "Remote Docker host detected; preserving the image user because "
                "automatic client-host UID/GID mapping is unsafe"
            )
            return None

        if run_as_user is None and not self._has_local_writable_results_bind(volumes):
            return None

        resolved = self._invoking_user()
        if resolved is None:
            if run_as_user is True:
                raise RuntimeError(
                    "Invoking UID/GID is unavailable on this platform; provide "
                    "container_user explicitly"
                )
            logger.warning(
                "Could not determine the invoking UID/GID; preserving the image user"
            )
        elif (
            self._has_local_writable_results_bind(volumes)
            and not self._is_non_root_container_user(resolved)
        ):
            raise RuntimeError(
                "The invoking Docker SDK process is root, so automatic user "
                "mapping would create root-owned results. Provide the submitting "
                "non-root UID:GID with container_user, or use run_as_user=False "
                "only for a reviewed root-required image."
            )
        return resolved

    def _daemon_uses_user_namespace(self) -> bool:
        """Return whether Docker remaps container UIDs away from host UIDs."""
        try:
            info = self._client.info()
        except Exception as exc:
            raise RuntimeError(
                "Could not inspect Docker daemon security options before "
                "configuring writable /results ownership"
            ) from exc
        if not isinstance(info, dict):
            raise RuntimeError(
                "Docker daemon returned invalid security information before "
                "configuring writable /results ownership"
            )
        if info.get("Rootless") is True:
            return True
        security_options = info.get("SecurityOptions") or []
        if isinstance(security_options, str):
            security_options = [security_options]
        return any(
            "rootless" in str(option).lower()
            or "userns" in str(option).lower()
            for option in security_options
        )

    @staticmethod
    def _is_non_root_container_user(container_user: str) -> bool:
        user = str(container_user).split(":", 1)[0].strip().lower()
        return user not in {"", "0", "root"}

    def _supplementary_groups(self, container_user: str | None) -> list[str]:
        """Return host supplementary groups only for the invoking local user."""
        if not container_user or self._docker_host_is_remote():
            return []
        invoking_user = self._invoking_user()
        if not invoking_user or str(container_user) != invoking_user:
            return []
        getgroups = getattr(os, "getgroups", None)
        getgid = getattr(os, "getgid", None)
        if not callable(getgroups) or not callable(getgid):
            return []
        primary_gid = int(getgid())
        return [
            str(group_id)
            for group_id in sorted({int(group) for group in getgroups()})
            if group_id != primary_gid
        ]

    def _prepare_local_runtime_home(
        self,
        volumes: dict | None,
        job_id: str,
    ) -> str | None:
        """Prepare a job-scoped host-owned HOME under a writable /results bind."""
        if self._docker_host_is_remote() or not volumes:
            return None
        safe_job_id = validate_job_id(job_id)
        for source, config in volumes.items():
            target = str(config.get("bind", "")).rstrip("/") or "/"
            mode = str(config.get("mode", "rw"))
            source_path = Path(source)
            if (
                target != "/results"
                or not source_path.is_absolute()
                or not self._mount_mode_is_writable(mode)
            ):
                continue

            source_root = source_path.resolve()
            runtime_root = (source_root / ".tao-runtime").resolve()
            job_runtime_root = (runtime_root / safe_job_id).resolve()
            home = (job_runtime_root / "home").resolve()
            if (
                source_root not in runtime_root.parents
                or runtime_root not in job_runtime_root.parents
                or runtime_root not in home.parents
                or job_runtime_root not in home.parents
            ):
                raise ValueError(
                    f"Docker runtime HOME escapes the /results bind source: {home}"
                )
            try:
                for relative in (
                    "",
                    ".cache/huggingface",
                    ".cache/torch",
                    ".cache/triton",
                    ".cache/torchinductor",
                    ".cache/matplotlib",
                ):
                    path = (home / relative).resolve()
                    if path != home and home not in path.parents:
                        raise ValueError(
                            f"Docker runtime cache escapes runtime HOME: {path}"
                        )
                    path.mkdir(mode=0o700, parents=True, exist_ok=True)
            except PermissionError as e:
                raise RuntimeError(
                    f"Could not prepare writable Docker runtime HOME under "
                    f"{source_root}: {e}"
                ) from e
            return f"/results/.tao-runtime/{safe_job_id}/home"
        return None

    def _add_non_root_environment_defaults(
        self,
        env: dict[str, str],
        container_user: str,
        volumes: dict | None,
        *,
        job_id: str,
    ) -> None:
        user_id = str(container_user).split(":", 1)[0].strip()
        invoking_user = self._invoking_user()
        invoking_user_id = (
            invoking_user.split(":", 1)[0].strip() if invoking_user else ""
        )
        home = env.get("HOME", "")
        if not home and user_id == invoking_user_id:
            home = self._prepare_local_runtime_home(volumes, job_id) or ""
        if not home:
            # A runtime directory created by the SDK host user with mode 0700
            # is unusable by a different explicit container UID.  Keep that
            # user's default HOME inside the container instead.
            home = f"/tmp/tao-home-{user_id}"
        cache = env.get("XDG_CACHE_HOME", "") or f"{home}/.cache"
        env.setdefault("USER", user_id)
        env.setdefault("LOGNAME", user_id)
        env.setdefault("HOME", home)
        env.setdefault("XDG_CACHE_HOME", cache)
        env.setdefault("HF_HOME", f"{cache}/huggingface")
        env.setdefault("TORCH_HOME", f"{cache}/torch")
        env.setdefault("TRITON_CACHE_DIR", f"{cache}/triton")
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{cache}/torchinductor")
        env.setdefault("MPLCONFIGDIR", f"{cache}/matplotlib")

    def _build_volumes(self, mounts: list | None) -> dict | None:
        if not mounts:
            return None
        volumes: dict[str, dict[str, str]] = {}
        for mount in mounts:
            source = target = mode = None
            if isinstance(mount, str):
                parts = mount.split(":", 2)
                if len(parts) == 1:
                    source = target = parts[0]
                    mode = "rw"
                elif len(parts) == 2:
                    source, target = parts
                    mode = "rw"
                else:
                    source, target, mode = parts
            elif isinstance(mount, dict):
                source = (
                    mount.get("source")
                    or mount.get("host_path")
                    or mount.get("from")
                    or mount.get("from_path")
                )
                target = (
                    mount.get("target")
                    or mount.get("container_path")
                    or mount.get("mount_path")
                    or mount.get("path")
                )
                if mount.get("read_only") is True:
                    mode = "ro"
                else:
                    mode = mount.get("mode", "rw")
            else:
                logger.warning("Unsupported Docker mount object %r; skipping", mount)
                continue

            if not source or not target:
                logger.warning("Invalid Docker mount %r; expected source and target", mount)
                continue
            volumes[str(source)] = {"bind": str(target), "mode": str(mode or "rw")}
        return volumes or None

    def _add_gpu_run_kwargs(
        self,
        run_kwargs: dict,
        env: dict,
        num_gpus: int,
        gpu_ids: list | str | None = None,
    ):
        """Attach the GPU request to ``run_kwargs``.

        ``gpu_ids`` selects *which* physical GPUs the container gets. Without
        it Docker resolves a count-based request to the first N devices, so
        ``gpu_count=1`` can only ever land on GPU 0 — on a shared host whose
        GPU 0 is busy every job OOMs while other GPUs sit idle, with no way to
        steer it (``NVIDIA_VISIBLE_DEVICES`` is overwritten below).
        """
        if num_gpus == 0:
            env.setdefault("NVIDIA_VISIBLE_DEVICES", "void")
            return

        device_ids = _normalize_gpu_ids(gpu_ids)

        if self._should_use_nvidia_runtime():
            env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
            # Honour an explicit selection; only fall back to "all" when the
            # caller did not name devices.
            env["NVIDIA_VISIBLE_DEVICES"] = ",".join(device_ids) if device_ids else "all"
            run_kwargs["runtime"] = "nvidia"
            return

        device_request_cls = getattr(getattr(self._docker, "types", None), "DeviceRequest", None)
        if not device_request_cls:
            logger.warning("Docker module has no DeviceRequest; launching without GPU request")
            return

        if device_ids:
            # device_ids and count are mutually exclusive in the Docker API.
            run_kwargs["device_requests"] = [
                device_request_cls(device_ids=list(device_ids), capabilities=[["gpu"]])
            ]
            logger.info("Pinning container to GPU(s) %s", ",".join(device_ids))
        elif num_gpus < 0:
            run_kwargs["device_requests"] = [
                device_request_cls(count=-1, capabilities=[["gpu"]])
            ]
        else:
            run_kwargs["device_requests"] = [
                device_request_cls(count=num_gpus, capabilities=[["gpu"]])
            ]

    @staticmethod
    def _should_use_nvidia_runtime() -> bool:
        try:
            return "tegra" in platform.uname().release.lower()
        except Exception:
            return False

    @staticmethod
    def _resolve_num_gpus(resource_shape: str | None, kwargs: dict) -> int:
        value = kwargs.get("num_gpus")
        if value is None:
            value = kwargs.get("num_gpu")
        if value is not None:
            return int(value)
        if not resource_shape:
            return 1
        match = re.search(r"(?:^|[.:-])(\d+)$", str(resource_shape))
        if match:
            return int(match.group(1))
        return 1

    @staticmethod
    def _parse_image(image: str) -> tuple[str, str, str | None]:
        if not image:
            raise ValueError("Docker image cannot be empty")

        last_slash = image.rfind("/")
        last_colon = image.rfind(":")
        tag = None
        repository_part = image
        if last_colon > last_slash:
            repository_part = image[:last_colon]
            tag = image[last_colon + 1:]

        first = repository_part.split("/", 1)[0]
        if "/" in repository_part and (
            "." in first or ":" in first or first == "localhost"
        ):
            registry = first
            repository = repository_part.split("/", 1)[1]
        else:
            registry = "docker.io"
            repository = repository_part
        return registry, repository, tag

    @classmethod
    def _split_image_name(cls, image: str) -> tuple[str, str | None]:
        registry, repository, tag = cls._parse_image(image)
        if registry == "docker.io":
            return repository, tag
        return f"{registry}/{repository}", tag

    @staticmethod
    def _container_exit_code(container) -> int | None:
        attrs = getattr(container, "attrs", {}) or {}
        state = attrs.get("State", {})
        exit_code = state.get("ExitCode")
        try:
            return int(exit_code)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _container_image(container) -> str:
        image = getattr(container, "image", None)
        tags = getattr(image, "tags", None)
        if tags:
            return tags[0]
        return str(image or "")

    def _query_gpus(self) -> list[dict]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return []

        if result.returncode != 0:
            return []

        gpus = []
        for idx, line in enumerate(result.stdout.splitlines()):
            if not line.strip():
                continue
            name, _, memory = line.partition(",")
            try:
                memory_mb = int(memory.strip())
            except ValueError:
                memory_mb = None
            gpus.append({"id": idx, "name": name.strip(), "memory_mb": memory_mb})
        return gpus

    # ------------------------------------------------------------------
    # Local state
    # ------------------------------------------------------------------

    def _update_job_status(
        self,
        job_id: str,
        status: str,
        message: str,
        exit_code: int | None = None,
    ):
        self._job_messages[job_id] = message
        state = self._jobs.setdefault(job_id, {"job_id": job_id})
        state["status"] = status
        state["message"] = message
        if exit_code is not None:
            state["exit_code"] = exit_code
        self._save_state()
        logger.info("Docker job %s: [%s] %s", job_id, status, message)

    def _load_state(self) -> dict[str, dict]:
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {
                    str(k): v for k, v in data.items()
                    if isinstance(v, dict)
                }
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning("Failed to load Docker state file %s: %s", STATE_PATH, e)
        return {}

    def _save_state(self):
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = STATE_PATH.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._jobs, f, indent=2, sort_keys=True)
            tmp_path.replace(STATE_PATH)
        except Exception as e:
            logger.warning("Failed to save Docker state file %s: %s", STATE_PATH, e)
