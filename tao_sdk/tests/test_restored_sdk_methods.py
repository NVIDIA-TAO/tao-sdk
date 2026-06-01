# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the SDK methods restored from the pre-refactor TaoExecutionSDK.

The old single-class ``TaoExecutionSDK`` had these methods that got dropped
when we split into per-platform SDKs (Brev, Docker, SLURM, K8s):

  - ``get_job(job_id)``       — store passthrough
  - ``list_jobs(status=...)`` — store passthrough
  - ``get_checkpoints(job_id)`` — list checkpoint-extension files in results

Tests verify each method exists with the right shape on the right SDK class.
"""

from __future__ import annotations

import pytest


GENERIC_METHODS = ("get_job", "list_jobs", "get_checkpoints")


@pytest.mark.parametrize("module_path,class_name", [
    ("tao_sdk.platforms.brev", "BrevSDK"),
    ("tao_sdk.platforms.docker", "DockerSDK"),
    ("tao_sdk.platforms.slurm", "SlurmSDK"),
    ("tao_sdk.platforms.kubernetes", "KubernetesSDK"),
])
def test_generic_methods_present_on_every_sdk(module_path, class_name):
    import importlib
    cls = getattr(importlib.import_module(module_path), class_name)
    for method in GENERIC_METHODS:
        assert hasattr(cls, method), f"{class_name} missing {method}"
        assert callable(getattr(cls, method))


def test_get_checkpoints_filters_by_extension(tmp_path, monkeypatch):
    """get_checkpoints filters list_path output to checkpoint extensions only."""
    from tao_sdk.platforms.brev import BrevSDK
    sdk = BrevSDK.__new__(BrevSDK)

    # Mock the dependencies the method actually uses.
    sdk.get_job_results_dir = lambda job_id: "s3://bucket/results/job-1"
    sdk.list_path = lambda path: [
        "best.pth", "last.ckpt", "model.engine", "model.onnx",
        "log.txt", "config.yaml", "metrics.json",
        "weights.tlt", "stats.csv",
    ]

    out = sdk.get_checkpoints("job-1")
    assert sorted(out) == sorted([
        "best.pth", "last.ckpt", "model.engine", "model.onnx", "weights.tlt",
    ])


def test_get_checkpoints_returns_empty_when_no_results():
    from tao_sdk.platforms.brev import BrevSDK
    sdk = BrevSDK.__new__(BrevSDK)
    sdk.get_job_results_dir = lambda job_id: ""
    assert sdk.get_checkpoints("nonexistent") == []


def test_get_checkpoints_handles_listing_failure():
    from tao_sdk.platforms.brev import BrevSDK
    sdk = BrevSDK.__new__(BrevSDK)
    sdk.get_job_results_dir = lambda job_id: "s3://bucket/results/job-1"
    def boom(_):
        raise RuntimeError("S3 down")
    sdk.list_path = boom
    assert sdk.get_checkpoints("job-1") == []


def test_slurm_get_checkpoints_skips_lustre():
    """SLURM's get_checkpoints returns [] for lustre:// (no listing impl yet)."""
    from tao_sdk.platforms.slurm import SlurmSDK
    sdk = SlurmSDK.__new__(SlurmSDK)
    sdk.get_job_results_dir = lambda job_id: "lustre:///lustre/fsw/.../results/job-1"
    assert sdk.get_checkpoints("job-1") == []


# ---------------------------------------------------------------------------
# _results_root_prefix — what gets injected as TAO_RESULTS_ROOT
# ---------------------------------------------------------------------------

def test_brev_results_root_prefix_is_results_when_bind_mount_present():
    from tao_sdk.platforms.brev import BrevSDK
    sdk = BrevSDK.__new__(BrevSDK)
    mounts = [{"host_path": "/host/out", "container_path": "/results"}]
    assert sdk._results_root_prefix(mounts=mounts) == "/results"


def test_brev_results_root_prefix_is_none_without_mount():
    from tao_sdk.platforms.brev import BrevSDK
    sdk = BrevSDK.__new__(BrevSDK)
    assert sdk._results_root_prefix() is None
