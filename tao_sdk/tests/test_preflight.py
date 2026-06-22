# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

from tao_sdk.preflight import (
    check_gpu_architecture_compatibility,
    check_nvidia_container_runtime,
    ensure_aws_cli,
    normalize_supported_sms,
    parse_compute_capabilities,
    query_host_sms,
)


def test_ensure_aws_cli_returns_existing_path():
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 0)

    result = ensure_aws_cli(which=lambda name: "/usr/bin/aws", run=fake_run)

    assert result.ok is True
    assert result.details["path"] == "/usr/bin/aws"
    assert calls == []


def test_ensure_aws_cli_reports_install_command_without_auto_install():
    result = ensure_aws_cli(
        which=lambda name: None,
        python_executable="/python",
        auto_install=False,
    )

    assert result.ok is False
    assert result.details["install_command"] == "/python -m pip install awscli"


def test_ensure_aws_cli_installs_when_requested():
    paths = iter([None, "/tmp/bin/aws"])
    commands = []

    def fake_which(name):
        return next(paths)

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = ensure_aws_cli(
        which=fake_which,
        run=fake_run,
        python_executable="/python",
        auto_install=True,
    )

    assert result.ok is True
    assert result.details["path"] == "/tmp/bin/aws"
    assert commands == [["/python", "-m", "pip", "install", "awscli"]]


def test_parse_compute_capabilities_normalizes_to_sm_names():
    assert parse_compute_capabilities("8.0\n12.1\n") == ["sm_80", "sm_121"]


def test_normalize_supported_sms_accepts_common_spellings():
    assert normalize_supported_sms("sm_80, sm90 12.0") == {
        "sm_80",
        "sm_90",
        "sm_120",
    }


def test_gpu_architecture_check_blocks_unsupported_sm_121():
    result = check_gpu_architecture_compatibility(
        host_sms=["sm_121"],
        supported_sms=["sm_80", "sm_90", "sm_100", "sm_120"],
        image="nvcr.io/nvidia/tao/tao-toolkit:7.0.0-cosmos-rl",
    )

    assert result.ok is False
    assert result.details["unsupported_sms"] == ["sm_121"]


def test_gpu_architecture_check_allows_unknown_support_list():
    result = check_gpu_architecture_compatibility(["sm_121"], None)

    assert result.ok is True
    assert "unknown" in result.message


def test_query_host_sms_uses_nvidia_smi_output():
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="12.1\n", stderr="")

    result = query_host_sms(run=fake_run)

    assert result.ok is True
    assert result.details["sms"] == ["sm_121"]


def test_check_nvidia_container_runtime_requires_nvidia_runtime():
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout='{"runc":{}}\n', stderr="")

    result = check_nvidia_container_runtime(run=fake_run)

    assert result.ok is False
    assert "NVIDIA" in result.message
