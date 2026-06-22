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

from tao_sdk.platforms.base import BaseHandler

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.path.expanduser("~/.tao_sdk_docker_jobs.json"))
DEFAULT_DOCKER_NETWORK = "tao_default"
DEFAULT_DOCKER_USERNAME = "$oauthtoken"


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
        **kwargs,
    ) -> dict:
        """Create a local Docker container for an SDK job."""
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

        num_gpus = self._resolve_num_gpus(resource_shape, kwargs)
        command = ["/bin/bash", "-c", entrypoint["command"]]
        env = self._build_environment(
            docker_env_vars=docker_env_vars,
            cloud_metadata=cloud_metadata,
            job_id=job_id,
        )
        volumes = self._build_volumes(mounts)

        if dedicated_node_group:
            logger.warning(
                "local-docker ignores dedicated_node_group=%s", dedicated_node_group
            )
        if kwargs.get("allowed_nodes"):
            logger.warning("local-docker ignores allowed_nodes=%s", kwargs["allowed_nodes"])

        self._update_job_status(job_id, "Pending", f"Preparing Docker image {image}")
        self._ensure_image(image, job_id)
        self._remove_existing_container(container_name)
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
        if shm_size is not None:
            run_kwargs["shm_size"] = shm_size
        else:
            run_kwargs["tmpfs"] = {"/dev/shm": ""}
        self._add_gpu_run_kwargs(run_kwargs, env, num_gpus)

        logger.info("Starting Docker container %s with image %s", container_name, image)
        container = self._client.containers.run(**run_kwargs)
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
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
            container = self._get_container(container_name, fallback_name=job_id)
            container.reload()
        except self._not_found_errors:
            state = self._jobs.get(job_id, {})
            state_status = state.get("status")
            if state_status == "Canceled":
                return "Canceled"
            if state_status == "Complete":
                return None
            if state_status == "Error":
                return "Error"
            self._update_job_status(
                job_id,
                "Error",
                f"Docker container {container_name} was not found",
            )
            return "Error"
        except Exception as e:
            logger.error("Failed to inspect Docker job %s: %s", job_id, e)
            self._update_job_status(job_id, "Error", f"Docker status failed: {e}")
            return "Error"

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
        if status in ("removing",):
            self._update_job_status(job_id, "Canceled", "Docker container is removing")
            return "Canceled"
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
            container = self._get_container(container_name, fallback_name=job_id)
            kwargs = {"stdout": True, "stderr": True, "timestamps": False}
            if tail_lines is not None:
                kwargs["tail"] = int(tail_lines)
            logs = container.logs(**kwargs)
            if isinstance(logs, bytes):
                return logs.decode("utf-8", errors="replace")
            return str(logs) if logs is not None else None
        except self._not_found_errors:
            logger.warning("Docker container %s not found for logs", container_name)
            return None
        except Exception as e:
            logger.error("Failed to get Docker logs for %s: %s", job_id, e)
            return None

    def delete_job(self, job_id: str) -> bool:
        """Stop and remove a Docker container."""
        container_name = self.get_job_name(job_id)
        try:
            container = self._get_container(container_name, fallback_name=job_id)
        except self._not_found_errors:
            self._update_job_status(
                job_id,
                "Canceled",
                f"Docker container {container_name} already absent",
            )
            return True
        except Exception as e:
            logger.error("Failed to fetch Docker container %s: %s", container_name, e)
            return False

        try:
            if str(getattr(container, "status", "")).lower() in (
                "running",
                "restarting",
                "paused",
                "created",
            ):
                container.stop(timeout=10)
            container.remove(force=True)
            self._update_job_status(
                job_id,
                "Canceled",
                f"Docker container {container_name} stopped and removed",
            )
            return True
        except Exception as e:
            logger.error("Failed to delete Docker container %s: %s", container_name, e)
            logger.error(traceback.format_exc())
            return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a Docker job."""
        return self.delete_job(job_id)

    def get_job_replicas(self, job_id: str) -> list[dict]:
        """Return single-container replica details for the job."""
        container_name = self.get_job_name(job_id)
        try:
            container = self._get_container(container_name, fallback_name=job_id)
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
        except self._not_found_errors:
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

    def _ping(self):
        try:
            self._client.ping()
        except AttributeError:
            return

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

    def _remove_existing_container(self, container_name: str):
        try:
            existing = self._client.containers.get(container_name)
        except self._not_found_errors:
            return
        try:
            existing.stop(timeout=5)
        except Exception:
            logger.debug("Existing container %s was not running", container_name)
        try:
            existing.remove(force=True)
        except Exception as e:
            logger.warning("Failed to remove existing container %s: %s", container_name, e)

    def _get_container(self, container_name: str, fallback_name: str | None = None):
        try:
            return self._client.containers.get(container_name)
        except self._not_found_errors:
            if fallback_name and fallback_name != container_name:
                return self._client.containers.get(fallback_name)
            raise

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

    def _add_gpu_run_kwargs(self, run_kwargs: dict, env: dict, num_gpus: int):
        if num_gpus == 0:
            env.setdefault("NVIDIA_VISIBLE_DEVICES", "void")
            return

        if self._should_use_nvidia_runtime():
            env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
            env["NVIDIA_VISIBLE_DEVICES"] = "all"
            run_kwargs["runtime"] = "nvidia"
            return

        device_request_cls = getattr(getattr(self._docker, "types", None), "DeviceRequest", None)
        if not device_request_cls:
            logger.warning("Docker module has no DeviceRequest; launching without GPU request")
            return

        if num_gpus < 0:
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
