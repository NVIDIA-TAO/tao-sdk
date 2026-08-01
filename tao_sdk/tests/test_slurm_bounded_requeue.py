# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for bounded scheduler-side SLURM self-requeues."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock

import pytest


def _make_handler(monkeypatch, tmp_path, retry_cap):
    from tao_sdk.platforms.slurm import handler as module

    monkeypatch.setattr(module, "STATE_PATH", tmp_path / "slurm-state.json")
    monkeypatch.delenv("SLURM_MAX_JOB_RETRIES", raising=False)
    return module.SlurmHandler(
        slurm_user="alice",
        slurm_hostnames=["login01"],
        creds={"SLURM_MAX_JOB_RETRIES": retry_cap},
        base_results_dir="/lustre/fsw/tao",
    )


def _build_script(handler):
    return handler._build_slurm_script(
        job_name="bounded-requeue",
        num_nodes=1,
        num_gpus=1,
        cpus_per_task=1,
        time_hours=4,
        account=None,
        partition="polar3",
        log_dir="/lustre/fsw/tao/slurm-logs",
        mail_user=None,
        image="/lustre/fsw/tao/image.sqsh",
        container_mounts="/lustre",
        use_timeout=True,
        timeout_hours=3.8,
        use_requeue=True,
        max_job_retries=handler.max_job_retries,
        use_srun=False,
        command="true",
        docker_env_vars={},
        exclusive=False,
    )


def _run_script(script, tmp_path, *, restart_count, timeout_status=124):
    trace = tmp_path / "scontrol.trace"
    trace.unlink(missing_ok=True)
    environment = {
        **os.environ,
        "SLURM_JOB_ID": "12345",
        "SCONTROL_TRACE": str(trace),
        "FAKE_TIMEOUT_STATUS": str(timeout_status),
    }
    if restart_count is None:
        environment.pop("SLURM_RESTART_COUNT", None)
    else:
        environment["SLURM_RESTART_COUNT"] = str(restart_count)
    harness = "\n".join([
        'timeout() { return "$FAKE_TIMEOUT_STATUS"; }',
        'scontrol() { printf "%s\\n" "$*" >> "$SCONTROL_TRACE"; }',
        script,
    ])
    result = subprocess.run(
        ["bash", "-c", harness],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )
    calls = trace.read_text(encoding="utf-8").splitlines() if trace.exists() else []
    return result, calls


def test_timeout_requeues_below_cap_and_stops_at_cap(monkeypatch, tmp_path):
    script = _build_script(_make_handler(monkeypatch, tmp_path, "3"))

    initial, initial_calls = _run_script(
        script, tmp_path, restart_count=None
    )
    assert initial.returncode == 0
    assert initial_calls == ["requeue 12345"]

    below, below_calls = _run_script(script, tmp_path, restart_count=2)
    assert below.returncode == 0
    assert below_calls == ["requeue 12345"]

    exhausted, exhausted_calls = _run_script(
        script, tmp_path, restart_count=3
    )
    assert exhausted.returncode == 124
    assert exhausted_calls == []
    assert "Maximum timeout self-requeue count (3) exhausted" in exhausted.stderr


def test_non_timeout_failure_is_preserved(monkeypatch, tmp_path):
    script = _build_script(_make_handler(monkeypatch, tmp_path, "3"))
    result, calls = _run_script(
        script,
        tmp_path,
        restart_count=0,
        timeout_status=42,
    )
    assert result.returncode == 42
    assert calls == []


def test_invalid_restart_count_fails_without_requeue(monkeypatch, tmp_path):
    script = _build_script(_make_handler(monkeypatch, tmp_path, "3"))
    result, calls = _run_script(
        script,
        tmp_path,
        restart_count="not-an-integer",
    )
    assert result.returncode == 124
    assert calls == []
    assert "Invalid SLURM_RESTART_COUNT: not-an-integer" in result.stderr


def test_zero_cap_disables_timeout_self_requeue(monkeypatch, tmp_path):
    script = _build_script(_make_handler(monkeypatch, tmp_path, 0))
    result, calls = _run_script(script, tmp_path, restart_count=None)
    assert result.returncode == 124
    assert calls == []
    assert "Maximum timeout self-requeue count (0) exhausted" in result.stderr


@pytest.mark.parametrize(
    ("scheduler_state", "expected"),
    [
        ("CANCELLED by 41852", "CANCELLED"),
        ("COMPLETED+", "COMPLETED"),
        ("RUNNING", "RUNNING"),
    ],
)
def test_scheduler_state_annotations_are_normalized(
    monkeypatch,
    tmp_path,
    scheduler_state,
    expected,
):
    handler = _make_handler(monkeypatch, tmp_path, "3")
    monkeypatch.setattr(
        handler,
        "_ssh",
        MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=scheduler_state + "\n",
                stderr="",
            )
        ),
    )
    assert handler.get_slurm_job_status("123") == expected


def test_sacct_cancelled_by_user_is_canonicalized(monkeypatch, tmp_path):
    handler = _make_handler(monkeypatch, tmp_path, "3")
    responses = iter([
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="CANCELLED by 41852|\n",
            stderr="",
        ),
    ])
    monkeypatch.setattr(handler, "_ssh", MagicMock(side_effect=responses))
    assert handler.get_slurm_job_status("123") == "CANCELLED"
