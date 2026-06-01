# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Docker platform support — runs jobs on the local Docker daemon.

Exposes both the low-level ``DockerHandler`` (used by the SDK internally and
discovered via the ``tao_sdk.platforms`` entry-point group) and the user-facing
``DockerSDK`` client class.
"""

from tao_sdk.platforms.docker.handler import DockerHandler
from tao_sdk.platforms.docker.sdk import DockerSDK

__all__ = ["DockerHandler", "DockerSDK"]
