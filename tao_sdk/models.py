# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models for the TAO Execution SDK. No TAO Core dependencies."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


class CredentialError(Exception):
    """Raised when required credentials are missing from environment variables."""
    pass


@dataclass
class Job:
    """Represents a submitted TAO job."""
    id: str
    network_arch: str
    action: str
    workspace_id: str
    backend_type: str = ""
    status: str = "Pending"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    backend_job_id: str = ""
    parent_job_id: str = ""
    results_dir: str = ""
    local_results_dir: str = ""
    base_checkpoint: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "job_id": self.id,
            "network_arch": self.network_arch,
            "action": self.action,
            "workspace_id": self.workspace_id,
            "backend_type": self.backend_type,
            "status": self.status,
            "created_at": self.created_at,
            "backend_job_id": self.backend_job_id,
            "parent_job_id": self.parent_job_id,
            "results_dir": self.results_dir,
            "local_results_dir": self.local_results_dir,
            "base_checkpoint": self.base_checkpoint,
        }


@dataclass
class JobStatus:
    """Current status of a TAO job."""
    job_id: str
    status: str
    message: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Checkpoint:
    """A model checkpoint produced during training."""
    path: str
    epoch: int
    metric_name: str = ""
    metric_value: float = 0.0
