# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``tao_sdk.script_runner.build_entrypoint``.

Covers mode inference, argparse-CLI construction, base64 heredoc round-trip,
and HF/NGC URI passthrough. Does not exercise the in-container runtime —
that's covered by ``test_script_runner.py``.

Output destination resolution (``results_root``, ``TAO_JOB_ID``,
``TAO_RESULTS_ROOT``) is purely a runtime concern now and is exercised in
``test_script_runner.py::TestOutputDestinationBranches``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from tao_sdk.script_runner import build_entrypoint


SCRIPT_RUNNER_PATH = Path(__file__).parent.parent / "script_runner.py"


def _decode_heredoc(command: str) -> str:
    m = re.search(r"echo '([A-Za-z0-9+/=]+)' \| base64 -d", command)
    assert m, "expected base64 heredoc not found"
    return base64.b64decode(m.group(1)).decode("utf-8")


def _extract_arg(command: str, flag: str) -> str | None:
    """Pull a CLI arg value from the python invocation line."""
    m = re.search(rf"{flag} '([^']*)'", command)
    if m:
        return m.group(1)
    m = re.search(rf"{flag} (\S+)", command)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Mode inference (the single most important contract)
# ---------------------------------------------------------------------------

def test_specs_only_infers_config_mode():
    ep = build_entrypoint(command="x {config_path}",
                          specs={"train": {"epochs": 1}})
    assert "--mode config" in ep["command"]
    assert "--config-format toml" in ep["command"]


def test_args_only_infers_args_mode():
    ep = build_entrypoint(command="mining --input {input}",
                          args={"input": "/data/in"})
    assert "--mode args" in ep["command"]


def test_path_keyed_inputs_only_infers_passthrough():
    ep = build_entrypoint(command="run.sh",
                          inputs={"/data/in": "s3://b/in"},
                          outputs=["/results/"])
    assert "--mode passthrough" in ep["command"]


def test_no_extras_infers_passthrough_with_empty_io():
    ep = build_entrypoint(command="echo hi")
    assert "--mode passthrough" in ep["command"]


def test_specs_and_args_simultaneously_raises():
    with pytest.raises(ValueError, match="either"):
        build_entrypoint(command="x", specs={}, args={})


# ---------------------------------------------------------------------------
# Heredoc / base64 round-trip
# ---------------------------------------------------------------------------

def test_heredoc_decodes_to_script_runner_source():
    ep = build_entrypoint(command="x", inputs={"/a": "s3://b/a"})
    decoded = _decode_heredoc(ep["command"])
    expected = SCRIPT_RUNNER_PATH.read_text(encoding="utf-8")
    assert decoded == expected, (
        f"heredoc payload doesn't match script_runner.py source "
        f"(decoded={len(decoded)} bytes, expected={len(expected)} bytes)"
    )


def test_heredoc_writes_to_expected_path():
    ep = build_entrypoint(command="x")
    assert "> /tmp/_tao_runner.py" in ep["command"]


def test_pip_install_line_present():
    ep = build_entrypoint(command="x")
    assert "pip install" in ep["command"]
    assert "fsspec" in ep["command"]
    assert "fsspec>=2024.6.0" in ep["command"]
    assert "s3fs>=2024.6.0" in ep["command"]
    assert "huggingface_hub" in ep["command"]


# ---------------------------------------------------------------------------
# CLI argument forwarding
# ---------------------------------------------------------------------------

def test_passthrough_forwards_inputs_outputs_verbatim():
    inputs = {"/data/in": "s3://b/in", "/models/cosmos": "hf_model://nvidia/Cosmos"}
    outputs = ["/results/", "/checkpoints/"]
    ep = build_entrypoint(command="x", inputs=inputs, outputs=outputs)
    assert json.loads(_extract_arg(ep["command"], "--inputs")) == inputs
    assert json.loads(_extract_arg(ep["command"], "--outputs")) == outputs


def test_passthrough_handles_hf_and_ngc_uris():
    """HF + NGC URIs must reach script_runner verbatim — the in-container
    download_remote() dispatches by scheme. build_entrypoint shouldn't filter."""
    inputs = {
        "/models/cosmos": "hf_model://nvidia/Cosmos-Reason2-8B",
        "/models/clip":   "ngc://nvidia/tao/clip:v1.0",
        "/data/in":       "s3://bucket/data/in",
    }
    ep = build_entrypoint(command="run", inputs=inputs)
    forwarded = json.loads(_extract_arg(ep["command"], "--inputs"))
    assert forwarded == inputs  # all 3 schemes preserved
    assert "--mode passthrough" in ep["command"]


def test_config_mode_forwards_specs_as_json():
    specs = {"policy": {"lora": {"r": 16}}, "train": {"epochs": 5}}
    ep = build_entrypoint(command="cmd {config_path}",
                          specs=specs, config_format="yaml")
    forwarded = json.loads(_extract_arg(ep["command"], "--specs"))
    assert forwarded == specs
    assert "--config-format yaml" in ep["command"]


def test_upload_excludes_passed_through():
    ep = build_entrypoint(command="x", inputs={"/a": "s3://b/a"},
                          upload_excludes=["inputs/", "*.tmp"])
    assert json.loads(_extract_arg(ep["command"], "--upload-excludes")) == [
        "inputs/", "*.tmp"
    ]


def test_flags_for_args_mode():
    ep = build_entrypoint(command="cmd --x {x}",
                          args={"x": "v"}, flags=["--verbose", "--dry-run"])
    assert json.loads(_extract_arg(ep["command"], "--flags")) == [
        "--verbose", "--dry-run"
    ]


def test_mount_path_passed_through_when_set():
    ep = build_entrypoint(command="x", inputs={"/a": "s3://b/a"},
                          mount_path="/lustre/cache")
    assert "--mount-path /lustre/cache" in ep["command"]


def test_mount_path_omitted_when_none():
    ep = build_entrypoint(command="x", inputs={"/a": "s3://b/a"})
    assert "--mount-path" not in ep["command"]


def test_no_job_id_in_cli():
    """job_id is resolved at runtime from TAO_JOB_ID (injected by SDK), not
    threaded through build_entrypoint."""
    ep = build_entrypoint(command="x", specs={"a": 1})
    assert "--job-id" not in ep["command"]


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

def test_return_shape():
    ep = build_entrypoint(command="x")
    assert set(ep.keys()) == {"command", "args_template"}
    assert isinstance(ep["command"], str)
    assert ep["args_template"] == ""


def test_command_contains_required_pieces():
    ep = build_entrypoint(command="my_train --x {config_path}",
                          specs={"a": 1})
    assert "pip install" in ep["command"]
    assert "base64 -d > /tmp/_tao_runner.py" in ep["command"]
    assert "python /tmp/_tao_runner.py" in ep["command"]
    # The user's command template is shlex-quoted, so look for it inside quotes
    assert "my_train --x {config_path}" in ep["command"]
