# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common interface for platform SDKs.

Each platform (Brev, Docker, SLURM, Kubernetes) implements this protocol. The agent calls
create_job() with platform-agnostic params (image, command, gpu_count,
inputs, outputs). Platform-specific params go in **kwargs.
"""

from __future__ import annotations

from os import PathLike
from typing import Iterator, Protocol, Sequence

from tao_sdk.models import Job, JobStatus


class PlatformSDK(Protocol):
    """Protocol that all platform SDKs implement."""

    def create_job(
        self,
        image: str,
        command: str,
        gpu_count: int = 1,
        env_vars: dict | None = None,
        inputs: dict | None = None,
        outputs: list | None = None,
        **kwargs,
    ) -> Job:
        """Submit a job to the platform.

        Args:
            image: Container image URI.
            command: Full shell command to run inside the container.
                Agent builds this, including spec file heredocs.
            gpu_count: Number of GPUs. Platform auto-resolves to
                the appropriate resource shape.
            env_vars: Additional env vars for the container. Platform
                auto-injects S3/NGC/HF credentials from os.environ.
            inputs: Files to download into the container before the
                command runs. Dict of {container_path: s3_uri}.
            outputs: Container paths to upload to S3 after the command
                completes. List of container paths.
            **kwargs: Platform-specific params (e.g., instance_id for Brev).

        Returns:
            Job object with .id and .results_dir.
        """
        ...

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status."""
        ...

    def get_job_logs(self, job_id: str, tail: int | None = None) -> str | None:
        """Fetch job logs."""
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        ...

    def get_failure_analysis(self, job_id: str) -> dict | None:
        """Get failure analysis for a failed job."""
        ...


class PythonScriptSDK(Protocol):
    """Capability implemented by SDKs that launch Python scripts directly.

    This is deliberately separate from :class:`PlatformSDK`: a Python script
    job has no container image or shell command, and conflating the two shapes
    makes it too easy for callers to accidentally depend on container-only
    behavior.
    """

    def create_python_job(
        self,
        script: str | PathLike[str],
        specs: dict | None = None,
        config_format: str = "json",
        script_args: Sequence[str] | None = None,
        env_vars: dict[str, str] | None = None,
        inputs: dict | None = None,
        outputs: dict | list | None = None,
        upload_excludes: Sequence[str] | None = None,
        cwd: str | PathLike[str] | None = None,
        network_arch: str = "",
        action: str = "train",
        hooks: dict | None = None,
        gpu_count: int = 1,
        gpu_ids: Sequence[int] | str | None = None,
    ) -> Job:
        """Launch a script with the configured virtual-environment Python."""
        ...

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get current job status."""
        ...

    def get_job_logs(self, job_id: str, tail: int | None = None) -> str:
        """Read the combined stdout/stderr log for a job."""
        ...

    def iter_job_log_chunks(
        self, job_id: str, chunk_size: int = 256 * 1024,
    ) -> Iterator[str]:
        """Stream a local job log without materializing the complete file."""
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job and its process group."""
        ...

    def get_failure_analysis(self, job_id: str) -> dict | None:
        """Get local process failure details for a failed job."""
        ...


class ArtifactCleanupSDK(Protocol):
    """Optional capability for preflighted, job-scoped artifact reclamation."""

    def validate_artifact_retention(self, **platform_kwargs) -> bool:
        """Fail before launch if this output route cannot be reclaimed safely."""
        ...

    def delete_job_artifacts(self, job_id: str) -> bool:
        """Delete persisted results belonging to one confirmed terminal job."""
        ...
