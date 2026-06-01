# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM platform support — runs jobs on a remote SLURM cluster via SSH.

Exposes both the low-level ``SlurmHandler`` (used by the SDK internally and
discovered via the ``tao_sdk.platforms`` entry-point group) and the user-facing
``SlurmSDK`` client class.
"""

from tao_sdk.platforms.slurm.handler import SlurmHandler
from tao_sdk.platforms.slurm.sdk import SlurmSDK

__all__ = ["SlurmHandler", "SlurmSDK"]
