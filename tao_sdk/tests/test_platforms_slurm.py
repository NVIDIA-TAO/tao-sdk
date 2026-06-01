# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the SLURM platform handler.

Validates package wiring, init paths, and credential handling. Real-cluster
behavior (SSH, sbatch, squeue) needs an actual SLURM cluster — out of scope
for unit tests.
"""

from __future__ import annotations

import pytest


def test_handler_class_exposed_at_package_level():
    from tao_sdk.platforms.slurm import SlurmHandler, SlurmSDK
    assert SlurmHandler.__module__ == "tao_sdk.platforms.slurm.handler"
    assert SlurmSDK.__module__ == "tao_sdk.platforms.slurm.sdk"


def test_handler_inherits_from_base():
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.base import BaseHandler
    assert issubclass(SlurmHandler, BaseHandler)


def test_registry_entry_point_resolves_slurm():
    from importlib.metadata import entry_points
    eps = list(entry_points(group="tao_sdk.platforms"))
    slurm_eps = [e for e in eps if e.name == "slurm"]
    assert slurm_eps
    cls = slurm_eps[0].load()
    assert cls.__name__ == "SlurmHandler"


def test_handler_requires_user(monkeypatch):
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    monkeypatch.delenv("SLURM_USER", raising=False)
    monkeypatch.delenv("SSH_USER", raising=False)
    with pytest.raises(ValueError, match="SLURM_USER"):
        SlurmHandler(slurm_user="", slurm_hostnames=["host"])


def test_handler_requires_hostnames():
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    with pytest.raises(ValueError, match="SLURM_HOSTNAME"):
        SlurmHandler(slurm_user="alice", slurm_hostnames=[])


def test_from_credentials_raises_credential_error_when_missing(monkeypatch):
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.models import CredentialError
    monkeypatch.delenv("SLURM_USER", raising=False)
    monkeypatch.delenv("SSH_USER", raising=False)
    monkeypatch.delenv("SLURM_HOSTNAME", raising=False)
    monkeypatch.delenv("SSH_HOST", raising=False)
    with pytest.raises(CredentialError, match="SLURM"):
        SlurmHandler.from_credentials({})


def test_from_credentials_error_message_no_secrets_json(monkeypatch):
    """The 'secrets.json' wording should be gone from the error message."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.models import CredentialError
    monkeypatch.delenv("SLURM_USER", raising=False)
    monkeypatch.delenv("SSH_USER", raising=False)
    monkeypatch.delenv("SLURM_HOSTNAME", raising=False)
    monkeypatch.delenv("SSH_HOST", raising=False)
    try:
        SlurmHandler.from_credentials({})
    except CredentialError as e:
        assert "secrets.json" not in str(e)
        assert ".env" in str(e)
    else:
        pytest.fail("Expected CredentialError")


def test_results_root_prefix_is_lustre_path():
    """SLURM never uses S3 — Lustre is the persistence layer (avoids
    GPU-idle scheduler kill). The prefix gets injected into the container
    as ``TAO_RESULTS_ROOT``; script_runner appends the job id at runtime."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.slurm.sdk import SlurmSDK
    handler = SlurmHandler(
        slurm_user="alice", slurm_hostnames=["login01"],
        base_results_dir="/lustre/fsw/tao",
    )
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    assert sdk._results_root_prefix() == "/lustre/fsw/tao/results"


def test_results_root_prefix_uses_handler_default_when_not_set():
    """Handler auto-fills base_results_dir from SLURM_BASE_RESULTS_DIR or
    a per-user Lustre template — matches TAO Core's convention. SDK never
    raises; just reads what the handler computed."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.slurm.sdk import SlurmSDK
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    prefix = sdk._results_root_prefix()
    assert prefix.endswith("/results")
    assert handler.base_results_dir in prefix
