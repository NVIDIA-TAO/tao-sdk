# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA TAO Execution SDK.

Usage::

    from tao_sdk.platforms.brev import BrevSDK         # Brev instances
    from tao_sdk.platforms.slurm import SlurmSDK       # SLURM clusters

    sdk = BrevSDK()
    job = sdk.create_job(image='...', command='...', gpu_count=8)

The platform SDKs live under ``tao_sdk.platforms.<name>``. Each platform
package also exports its low-level handler (e.g., ``BrevHandler``) for code
that needs direct API access.
"""

from tao_sdk.job_store import JobStore
from tao_sdk.models import Checkpoint, CredentialError, Job, JobStatus
from tao_sdk.platforms.protocol import ArtifactCleanupSDK, PythonScriptSDK
from tao_sdk.platforms.virtualenv import VirtualEnvSDK

__all__ = [
    "Job", "JobStatus", "Checkpoint", "CredentialError", "JobStore",
    "ArtifactCleanupSDK", "PythonScriptSDK", "VirtualEnvSDK",
]
