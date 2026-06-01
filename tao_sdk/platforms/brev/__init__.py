# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev platform support — instance-based GPU compute via the brev CLI.

Exposes both the low-level ``BrevHandler`` (used by the SDK internally and
discovered via the ``tao_sdk.platforms`` entry-point group) and the user-facing
``BrevSDK`` client class.
"""

from tao_sdk.platforms.brev.handler import BrevHandler
from tao_sdk.platforms.brev.sdk import BrevSDK

__all__ = ["BrevHandler", "BrevSDK"]
