# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base execution handler protocol for TAO SDK.

Defines the interface that all execution handlers (Brev, Docker, SLURM,
Kubernetes) must implement. The SDK interacts with handlers exclusively
through this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutionHandler(Protocol):
    """Protocol for execution handlers.

    Each handler manages job lifecycle on a specific backend (Brev, Docker,
    SLURM, Kubernetes). The SDK calls these methods without knowing which
    backend is in use.
    """

    def create_job(
        self,
        image: str,
        network: str,
        action: str,
        cloud_metadata: dict,
        specs: dict,
        job_id: str,
        docker_env_vars: dict,
        resource_shape: str | None,
        dedicated_node_group: str | None,
        num_nodes: int,
        entrypoint: dict | None = None,
        **kwargs,
    ) -> object:
        """Submit a job to the backend.

        Args:
            image: Container image URI.
            network: Network architecture name.
            action: TAO action (train, eval, export, etc.).
            cloud_metadata: Cloud storage credentials and config.
            specs: Training spec dict.
            job_id: Unique job identifier.
            docker_env_vars: Environment variables for the container.
            resource_shape: GPU resource shape (e.g., 'gpu.a100.8').
            dedicated_node_group: Node group for dedicated resources.
            num_nodes: Number of nodes for distributed training.
            entrypoint: Optional custom entrypoint from skill config.
                If provided, overrides the default container command.
                Schema: {"command": str, "args_template": str}
        """
        ...

    def get_job_name(self, job_id: str) -> str:
        """Generate the backend-specific job name from a job ID."""
        ...

    def get_tao_job_status(self, job_id: str) -> str | None:
        """Get job status mapped to TAO status strings.

        Returns:
            One of: 'Pending', 'Running', 'Paused', 'Canceled', 'Error',
            or None (meaning backend reports complete — check logs for
            actual success/failure).
        """
        ...

    def normalize_status(self, raw_status: str | None) -> str:
        """Map raw backend status to SDK status string."""
        ...

    def get_job_message(self, job_id: str) -> str:
        """Get the last detailed status message for a job."""
        ...

    def get_job_logs(self, job_id: str, tail_lines: int | None = None) -> str | None:
        """Fetch container logs for a job."""
        ...

    def delete_job(self, job_id: str) -> bool:
        """Delete/cancel a job on the backend."""
        ...

    def get_available_instances(self, **kwargs) -> dict:
        """List available compute resources on the backend."""
        ...

    def resolve_resource_shape(
        self, num_gpus: int, preferred_node_group: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Match a GPU count to an available resource shape.

        Returns:
            Tuple of (resource_shape, node_group). Either may be None.
        """
        ...

    def resolve_cpu_shape(
        self,
        preferred_node_group: str | None = None,
        preference: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Find an appropriate CPU resource shape.

        Returns:
            Tuple of (resource_shape, node_group).
        """
        ...

    def get_volumes(self, node_group_id: str | None = None) -> list[dict]:
        """Query available storage volumes (NFS, Lustre, etc.).

        Returns list of dicts with name, from_path, type.
        """
        ...

    def get_mount_objects(
        self, volumes: list[dict] | None = None,
        path_prefix: str = "/",
    ) -> tuple[str | None, list | None]:
        """Build mount objects for attaching storage to jobs.

        Returns (mount_path, mounts_list). (None, None) if unsupported.
        """
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job. Handles backend-specific cancellation logic."""
        ...

    @classmethod
    def from_credentials(cls, creds: dict) -> "ExecutionHandler":
        """Construct a handler from a credentials dict."""
        ...


class BaseHandler:
    """Base class with default implementations for optional handler methods.

    Handlers can inherit from this to get sensible defaults for methods
    that not all platforms support (e.g., Brev has no shared storage).
    """

    def resolve_resource_shape(
        self, num_gpus: int, preferred_node_group: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Default: no resource shape resolution."""
        return None, preferred_node_group

    def resolve_cpu_shape(
        self,
        preferred_node_group: str | None = None,
        preference: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Default: no CPU shape resolution."""
        return None, preferred_node_group

    def get_volumes(self, node_group_id: str | None = None) -> list[dict]:
        """Default: no shared storage."""
        return []

    def get_mount_objects(
        self, volumes: list[dict] | None = None,
        path_prefix: str = "/",
    ) -> tuple[str | None, list | None]:
        """Default: no mount support."""
        return None, None

    def cancel_job(self, job_id: str) -> bool:
        """Default: delegate to delete_job."""
        return self.delete_job(job_id)

    def get_failure_analysis(self, job_id: str) -> dict | None:
        """Default: no failure analysis."""
        return None

    def get_job_replicas(self, job_id: str) -> list[dict]:
        """Default: no replica info."""
        return []

    def get_storage_permissions(
        self, volume_name: str, node_group_id: str,
    ) -> list[dict]:
        """Default: no storage permissions."""
        return []
