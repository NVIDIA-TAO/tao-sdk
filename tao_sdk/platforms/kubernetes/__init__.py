# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes platform support — runs jobs as single-pod k8s Jobs.

Exposes both the low-level ``KubernetesHandler`` (used by the SDK internally
and discovered via the ``tao_sdk.platforms`` entry-point group) and the
user-facing ``KubernetesSDK`` client class.

Requires the ``kubernetes`` Python client (``pip install nvidia-tao-sdk[kubernetes]``)
and an authenticated cluster connection — kubeconfig (``~/.kube/config``) or
in-cluster service account.
"""

from tao_sdk.platforms.kubernetes.handler import KubernetesHandler
from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

__all__ = ["KubernetesHandler", "KubernetesSDK"]
