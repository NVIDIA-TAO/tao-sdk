# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform handlers and SDKs for TAO.

Each platform (Brev, Docker, SLURM, Kubernetes) is a sub-package under ``platforms/`` containing
both the low-level handler (executes jobs against the platform API) and the
user-facing SDK class (the public client). The ``protocol`` and ``base`` modules
define the contract platforms implement; ``registry`` discovers handlers via the
``tao_sdk.platforms`` entry-point group.
"""

from tao_sdk.platforms.base import BaseHandler, ExecutionHandler
from tao_sdk.platforms.protocol import PlatformSDK
from tao_sdk.platforms.registry import get_handler_class

__all__ = ["BaseHandler", "ExecutionHandler", "PlatformSDK", "get_handler_class"]
