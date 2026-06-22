# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Docker platform handler.

Validates package wiring, init paths, and a few invariants. Heavier tests of
the docker.py handler internals (subprocess + docker client interactions) live
in the legacy test file imported alongside the handler when needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_handler_class_exposed_at_package_level():
    from tao_sdk.platforms.docker import DockerHandler, DockerSDK
    assert DockerHandler.__module__ == "tao_sdk.platforms.docker.handler"
    assert DockerSDK.__module__ == "tao_sdk.platforms.docker.sdk"


def test_handler_inherits_from_base():
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.base import BaseHandler
    assert issubclass(DockerHandler, BaseHandler)


def test_handler_get_job_name_is_docker_safe():
    from tao_sdk.platforms.docker.handler import DockerHandler
    handler = DockerHandler.__new__(DockerHandler)
    name = handler.get_job_name("3F8F-2D24/abc-XYZ")
    assert name.startswith("tao-job-")
    assert len(name) <= 128
    assert all(c.isalnum() or c in "_.-" for c in name)


def test_registry_entry_point_resolves_docker():
    """Entry point under tao_sdk.platforms group should resolve DockerHandler."""
    from importlib.metadata import entry_points
    eps = list(entry_points(group="tao_sdk.platforms"))
    docker_eps = [e for e in eps if e.name == "docker"]
    assert docker_eps, f"docker entry point missing; have: {[e.name for e in eps]}"
    cls = docker_eps[0].load()
    assert cls.__name__ == "DockerHandler"


def test_sdk_init_raises_when_docker_missing(monkeypatch):
    """If the docker Python package isn't importable, DockerSDK init raises CredentialError."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    from tao_sdk.models import CredentialError
    # Force the handler's _import_docker() to fail by patching importlib.
    with patch("tao_sdk.platforms.docker.handler.DockerHandler._import_docker", side_effect=ImportError("no docker")):
        with pytest.raises(CredentialError):
            DockerSDK()


# ---------------------------------------------------------------------------
# _results_root_prefix — what gets injected as TAO_RESULTS_ROOT
# ---------------------------------------------------------------------------

def test_docker_results_root_prefix_is_results_when_bind_mount_present():
    """A /results bind mount signals 'I have local persistence' — prefix is
    /results so the runner writes there directly, no cloud upload."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    sdk = object.__new__(DockerSDK)
    mounts = [{"host_path": "/host/out", "container_path": "/results"}]
    assert sdk._results_root_prefix(mounts=mounts) == "/results"


def test_docker_results_root_prefix_is_none_without_mount():
    """No /results bind mount → no TAO_RESULTS_ROOT injection. script_runner
    falls back to S3 (if S3_BUCKET_NAME) or container-ephemeral with warning."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    sdk = object.__new__(DockerSDK)
    assert sdk._results_root_prefix() is None


def test_docker_results_root_prefix_ignores_other_mounts():
    """Mounts at non-/results paths don't qualify."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    sdk = object.__new__(DockerSDK)
    mounts = [{"host_path": "/host/data", "container_path": "/data"}]
    assert sdk._results_root_prefix(mounts=mounts) is None


# ---------------------------------------------------------------------------
# Docker network prereq
# ---------------------------------------------------------------------------

def _network_test_handler():
    from tao_sdk.platforms.docker.handler import DockerHandler

    class NotFound(Exception):
        pass

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_network = "tao_default"
    handler._docker = SimpleNamespace(errors=SimpleNamespace(NotFound=NotFound))
    handler._client = MagicMock()
    handler._job_messages = {}
    handler._jobs = {}
    return handler, NotFound


def test_docker_handler_creates_missing_network():
    handler, NotFound = _network_test_handler()
    handler._client.networks.get.side_effect = NotFound("missing")

    handler._ensure_network("job-1")

    handler._client.networks.create.assert_called_once_with("tao_default")
    assert handler.get_job_message("job-1") == "Creating Docker network tao_default"


def test_docker_handler_does_not_create_existing_network():
    handler, _ = _network_test_handler()
    handler._client.networks.get.return_value = object()

    handler._ensure_network("job-1")

    handler._client.networks.create.assert_not_called()
