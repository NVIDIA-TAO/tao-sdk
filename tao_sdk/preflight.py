# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable host preflight helpers for TAO launch flows."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class PreflightResult:
    """Structured preflight outcome."""

    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def ensure_aws_cli(
    *,
    auto_install: bool = False,
    python_executable: str = sys.executable,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> PreflightResult:
    """Verify ``aws`` exists and optionally install ``awscli`` with pip.

    This is explicit opt-in because installing host tools is a side effect.
    Skills can call it after the user approves dependency remediation, then
    rerun S3 path checks with the returned CLI path.
    """
    aws = which("aws")
    if aws:
        return PreflightResult(True, "aws CLI found", {"path": aws})

    install_command = [python_executable, "-m", "pip", "install", "awscli"]
    if not auto_install:
        return PreflightResult(
            False,
            "aws CLI not found",
            {"install_command": " ".join(install_command)},
        )

    result = run(
        install_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return PreflightResult(
            False,
            "awscli installation failed",
            {
                "install_command": " ".join(install_command),
                "stderr": (result.stderr or result.stdout or "").strip(),
            },
        )

    aws = which("aws")
    if not aws:
        return PreflightResult(
            False,
            "awscli installed but aws is not on PATH",
            {"install_command": " ".join(install_command)},
        )
    return PreflightResult(True, "aws CLI installed", {"path": aws})


def parse_compute_capabilities(output: str) -> list[str]:
    """Parse ``nvidia-smi --query-gpu=compute_cap`` output into ``sm_###``."""
    sms: list[str] = []
    for raw in (output or "").splitlines():
        value = raw.strip().strip(",")
        if not value or value.lower().startswith("compute"):
            continue
        match = re.search(r"(\d+)\s*[.]\s*(\d+)", value)
        if not match:
            continue
        sms.append(f"sm_{int(match.group(1))}{int(match.group(2))}")
    return sms


def query_host_sms(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: int = 10,
) -> PreflightResult:
    """Return host GPU compute capabilities as ``sm_###`` strings."""
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive around host tools
        return PreflightResult(False, f"nvidia-smi failed: {exc}")

    if result.returncode != 0:
        return PreflightResult(
            False,
            "nvidia-smi compute capability query failed",
            {"stderr": (result.stderr or result.stdout or "").strip()},
        )
    sms = parse_compute_capabilities(result.stdout)
    if not sms:
        return PreflightResult(False, "no GPU compute capabilities reported")
    return PreflightResult(True, "host GPU architectures detected", {"sms": sms})


def normalize_supported_sms(values: str | Iterable[str] | None) -> set[str]:
    """Normalize image-supported architectures to ``sm_###`` strings."""
    if values is None:
        return set()
    if isinstance(values, str):
        raw_values = re.split(r"[\s,;]+", values)
    else:
        raw_values = [str(value) for value in values]

    supported: set[str] = set()
    for raw in raw_values:
        value = raw.strip()
        if not value:
            continue
        match = re.match(r"sm[_-]?(\d+)$", value, re.IGNORECASE)
        if match:
            supported.add(f"sm_{match.group(1)}")
            continue
        match = re.match(r"(\d+)\.(\d+)$", value)
        if match:
            supported.add(f"sm_{int(match.group(1))}{int(match.group(2))}")
    return supported


def check_gpu_architecture_compatibility(
    host_sms: Iterable[str],
    supported_sms: str | Iterable[str] | None,
    *,
    image: str | None = None,
) -> PreflightResult:
    """Check host GPU architectures against an image's supported SM list."""
    host = normalize_supported_sms(host_sms)
    supported = normalize_supported_sms(supported_sms)
    if not host:
        return PreflightResult(False, "no host GPU architectures supplied")
    if not supported:
        return PreflightResult(
            True,
            "image GPU architecture support is unknown; compatibility not enforced",
            {"host_sms": sorted(host), "image": image},
        )

    unsupported = sorted(host - supported)
    if unsupported:
        return PreflightResult(
            False,
            "host GPU architecture is not supported by the selected image",
            {
                "host_sms": sorted(host),
                "supported_sms": sorted(supported),
                "unsupported_sms": unsupported,
                "image": image,
            },
        )
    return PreflightResult(
        True,
        "host GPU architecture is supported by the selected image",
        {"host_sms": sorted(host), "supported_sms": sorted(supported), "image": image},
    )


def check_nvidia_container_runtime(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: int = 10,
) -> PreflightResult:
    """Verify Docker can report the NVIDIA container runtime."""
    try:
        result = run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive around host tools
        return PreflightResult(False, f"docker info failed: {exc}")

    if result.returncode != 0:
        return PreflightResult(
            False,
            "docker info failed",
            {"stderr": (result.stderr or result.stdout or "").strip()},
        )
    if "nvidia" not in (result.stdout or "").lower():
        return PreflightResult(False, "NVIDIA container runtime is not registered")
    return PreflightResult(True, "NVIDIA container runtime is registered")


__all__ = [
    "PreflightResult",
    "check_gpu_architecture_compatibility",
    "check_nvidia_container_runtime",
    "ensure_aws_cli",
    "normalize_supported_sms",
    "parse_compute_capabilities",
    "query_host_sms",
]
