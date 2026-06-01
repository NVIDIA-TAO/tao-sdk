# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for tao_sdk tests."""

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _unset_backend():
    """Ensure BACKEND is not set — prevents MongoDB imports."""
    os.environ.pop("BACKEND", None)
    # Clear handler registry cache between tests
    from tao_sdk.platforms.registry import clear_cache
    clear_cache()


@pytest.fixture
def platform_env(tmp_path, monkeypatch):
    """Set minimal platform env vars for SDK initialization."""
    monkeypatch.setenv("BREV_API_TOKEN", "test-token")
    # State file in tmp dir so tests don't pollute working directory
    return str(tmp_path / "tao_session_state.json")


@pytest.fixture
def cloud_env(platform_env, monkeypatch):
    """Set cloud storage env vars on top of platform_env."""
    monkeypatch.setenv("ACCESS_KEY", "test-key")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.test.example.com")
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("NGC_KEY", "test-ngc-key")
    return platform_env


@pytest.fixture
def mock_handler():
    """A MagicMock that stands in for a platform handler."""
    handler = MagicMock()
    handler.get_job_name.return_value = "tao-job-00000000-0000-0000-0000"
    handler.normalize_status.side_effect = lambda raw: "Complete" if raw is None else raw
    handler.get_tao_job_status.return_value = "Pending"
    handler.get_job_logs.return_value = "Epoch 10/50 - loss: 0.32\nEpoch 20/50 - loss: 0.18"
    handler.delete.return_value = True
    handler.create_job.return_value = MagicMock()
    handler.create_image_pull_secret.return_value = "tao-image-pull-secret"
    handler.get_mount_objects.return_value = (None, None)
    handler.get_volumes.return_value = []
    handler.resolve_resource_shape.return_value = (None, None)
    handler.resolve_cpu_shape.return_value = (None, None)
    handler.cancel_job.return_value = True
    handler.get_storage_permissions.return_value = []
    handler.from_credentials = classmethod(lambda cls, creds: handler)
    return handler
