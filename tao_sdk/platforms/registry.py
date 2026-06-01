# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform handler registry for TAO SDK.

Discovers platform handlers via Python entry points (group:
``tao_sdk.platforms``). The SDK discovers them at runtime without hard imports —
each platform registers its handler class via the ``tao_sdk.platforms``
entry-point group in ``pyproject.toml``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_HANDLER_CACHE: dict[str, type] = {}


def get_handler_class(backend_type: str) -> type:
    """Load a handler class by backend type name.

    Searches Python entry points in the ``tao_sdk.platforms`` group.
    Results are cached after first lookup.

    Args:
        backend_type: Backend name (e.g., 'brev', 'docker').

    Returns:
        The handler class (not an instance).

    Raises:
        NotImplementedError: If no handler is found for the backend type.
    """
    if backend_type in _HANDLER_CACHE:
        return _HANDLER_CACHE[backend_type]

    # Try entry points
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="tao_sdk.platforms")
        for ep in eps:
            if ep.name == backend_type:
                cls = ep.load()
                _HANDLER_CACHE[backend_type] = cls
                logger.debug("Loaded handler '%s' from entry point: %s", backend_type, ep.value)
                return cls
    except Exception as e:
        logger.debug("Entry point discovery failed for '%s': %s", backend_type, e)

    # Fallback: direct import. Used when entry-point metadata is unavailable
    # (e.g., running from a source checkout that hasn't been pip-installed).
    _FALLBACK_IMPORTS = {
        "brev":       ("tao_sdk.platforms.brev", "BrevHandler"),
        "docker":     ("tao_sdk.platforms.docker", "DockerHandler"),
        "local-docker": ("tao_sdk.platforms.docker", "DockerHandler"),  # alias
        "slurm":      ("tao_sdk.platforms.slurm", "SlurmHandler"),
        "kubernetes": ("tao_sdk.platforms.kubernetes", "KubernetesHandler"),
        "k8s":        ("tao_sdk.platforms.kubernetes", "KubernetesHandler"),  # alias
        "local-k8s":  ("tao_sdk.platforms.kubernetes", "KubernetesHandler"),  # kind/minikube/Docker Desktop k8s
    }
    if backend_type in _FALLBACK_IMPORTS:
        module_path, class_name = _FALLBACK_IMPORTS[backend_type]
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            _HANDLER_CACHE[backend_type] = cls
            logger.debug("Loaded handler '%s' via fallback import: %s.%s",
                         backend_type, module_path, class_name)
            return cls
        except ImportError:
            pass

    available = list(_HANDLER_CACHE.keys())
    raise NotImplementedError(
        f"Backend '{backend_type}' not found. "
        f"Install the handler extra (e.g., pip install 'nvidia-tao-sdk[{backend_type}]') "
        f"or check that the package registers a tao_sdk.platforms entry point. "
        f"Currently available: {available or 'none'}"
    )


def clear_cache():
    """Clear the handler class cache. Useful for testing."""
    _HANDLER_CACHE.clear()
