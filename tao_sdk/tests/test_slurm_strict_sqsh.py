# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for migrated SLURM SQSH and retry policy settings."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_handler(monkeypatch, tmp_path, *, creds=None):
    from tao_sdk.platforms.slurm import handler as module

    monkeypatch.setattr(module, "STATE_PATH", tmp_path / "slurm-state.json")
    for key in (
        "SLURM_USE_SQSH",
        "SLURM_STRICT_SQSH",
        "SLURM_MAX_JOB_RETRIES",
    ):
        monkeypatch.delenv(key, raising=False)
    return module.SlurmHandler(
        slurm_user="alice",
        slurm_hostnames=["login01"],
        creds=creds,
        base_results_dir="/lustre/fsw/tao",
    )


def test_slurm_policy_defaults_remain_backward_compatible(monkeypatch, tmp_path):
    from tao_sdk.platforms.slurm.handler import MAX_JOB_RETRIES

    handler = _make_handler(monkeypatch, tmp_path)

    assert handler.strict_sqsh is False
    assert handler.max_job_retries == MAX_JOB_RETRIES == 10


def test_strict_sqsh_refuses_registry_image_fallback(monkeypatch, tmp_path):
    handler = _make_handler(
        monkeypatch,
        tmp_path,
        creds={"SLURM_STRICT_SQSH": "true"},
    )
    monkeypatch.setattr(handler, "_check_sqsh_exists", lambda _path: False)
    convert = MagicMock(return_value=False)
    monkeypatch.setattr(handler, "_convert_to_sqsh_via_srun", convert)

    with pytest.raises(
        RuntimeError,
        match="SLURM_STRICT_SQSH.*refusing registry-image fallback",
    ):
        handler._prepare_container_image(
            "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt",
            "/lustre/fsw/tao/sqsh",
        )

    convert.assert_called_once()


def test_non_strict_sqsh_preserves_registry_image_fallback(monkeypatch, tmp_path):
    image = "nvcr.io/nvidia/tao/tao-toolkit:7.0.1-pyt"
    handler = _make_handler(monkeypatch, tmp_path)
    monkeypatch.setattr(handler, "_check_sqsh_exists", lambda _path: False)
    monkeypatch.setattr(
        handler,
        "_convert_to_sqsh_via_srun",
        MagicMock(return_value=False),
    )

    assert handler._prepare_container_image(image, "/lustre/fsw/tao/sqsh") == image


def test_strict_sqsh_requires_sqsh_conversion_enabled(monkeypatch, tmp_path):
    with pytest.raises(
        ValueError,
        match="SLURM_STRICT_SQSH requires SLURM_USE_SQSH",
    ):
        _make_handler(
            monkeypatch,
            tmp_path,
            creds={
                "SLURM_USE_SQSH": "false",
                "SLURM_STRICT_SQSH": "true",
            },
        )


def test_strict_sqsh_rejects_ambiguous_boolean(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="SLURM_STRICT_SQSH must be one of"):
        _make_handler(
            monkeypatch,
            tmp_path,
            creds={"SLURM_STRICT_SQSH": "sometimes"},
        )


@pytest.mark.parametrize("value", (-1, "-1", "1.5", "many", True))
def test_slurm_max_job_retries_rejects_invalid_values(
    monkeypatch,
    tmp_path,
    value,
):
    with pytest.raises(ValueError, match="SLURM_MAX_JOB_RETRIES"):
        _make_handler(
            monkeypatch,
            tmp_path,
            creds={"SLURM_MAX_JOB_RETRIES": value},
        )


def test_slurm_max_job_retries_accepts_zero_and_positive_caps(
    monkeypatch,
    tmp_path,
):
    zero = _make_handler(
        monkeypatch,
        tmp_path,
        creds={"SLURM_MAX_JOB_RETRIES": 0},
    )
    positive = _make_handler(
        monkeypatch,
        tmp_path,
        creds={"SLURM_MAX_JOB_RETRIES": "3"},
    )

    assert zero.max_job_retries == 0
    assert positive.max_job_retries == 3


def test_configured_slurm_retry_cap_blocks_resubmission(monkeypatch, tmp_path):
    handler = _make_handler(
        monkeypatch,
        tmp_path,
        creds={"SLURM_MAX_JOB_RETRIES": "1"},
    )
    metadata = handler._default_runtime_identity("job-1")
    metadata.update({
        "slurm_job_id": "42",
        "retry_count": 1,
        "submission_attempted": True,
    })
    handler._jobs["job-1"] = metadata
    monkeypatch.setattr(handler, "_select_recorded_hostname", MagicMock(return_value=True))
    monkeypatch.setattr(
        handler,
        "get_slurm_job_status",
        MagicMock(return_value="NODE_FAIL"),
    )
    monkeypatch.setattr(
        handler,
        "_is_retriable_failure",
        MagicMock(return_value=True),
    )
    resubmit = MagicMock()
    monkeypatch.setattr(handler, "_resubmit_failed_job", resubmit)

    assert handler.get_tao_job_status("job-1") == "Error"
    resubmit.assert_not_called()
