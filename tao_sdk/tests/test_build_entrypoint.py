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
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def test_decoded_runner_executes_without_installed_tao_sdk(tmp_path):
    ep = build_entrypoint(command="x")
    runner_path = tmp_path / "decoded_runner.py"
    runner_path.write_text(_decode_heredoc(ep["command"]), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-I", "-S", str(runner_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "TAONext Script Runner" in result.stdout


def test_pip_install_line_present():
    ep = build_entrypoint(command="x")
    assert "pip install" in ep["command"]
    assert "fsspec" in ep["command"]
    assert "fsspec>=2024.6.0" in ep["command"]
    assert "s3fs>=2024.6.0" in ep["command"]
    assert "huggingface_hub" in ep["command"]


def test_pip_install_constraints_are_shell_safe(tmp_path):
    install_line = build_entrypoint(command="x")["command"].splitlines()[0]
    captured_args = tmp_path / "pip-args.txt"
    fake_pip = (
        "pip() { printf '%s\\n' \"$@\" > "
        f"{shlex.quote(str(captured_args))}; }}\n"
    )

    result = subprocess.run(
        ["bash", "-c", fake_pip + install_line],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert captured_args.read_text(encoding="utf-8").splitlines() == [
        "install",
        "fsspec>=2024.6.0",
        "s3fs>=2024.6.0",
        "toml",
        "pandas",
        "pyyaml",
        "huggingface_hub",
    ]
    assert not (tmp_path / "=2024.6.0").exists()


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


def test_config_mode_normalizes_numpy_without_mutating_caller():
    specs = {
        "train": {
            "lr": np.float64(0.25),
            "epochs": np.int64(4),
            "shape": np.array([16, 32]),
        }
    }

    ep = build_entrypoint(command="cmd {config_path}", specs=specs)

    assert json.loads(_extract_arg(ep["command"], "--specs")) == {
        "train": {"lr": 0.25, "epochs": 4, "shape": [16, 32]}
    }
    assert isinstance(specs["train"]["epochs"], np.integer)
    assert isinstance(specs["train"]["shape"], np.ndarray)


def test_config_mode_rejects_invalid_values_with_location():
    with pytest.raises(ValueError, match=r"specs\['lr'\].*finite"):
        build_entrypoint(command="cmd {config_path}", specs={"lr": float("nan")})
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        build_entrypoint(command="cmd {config_path}", specs={1: "value"})
    with pytest.raises(TypeError, match="TOML does not support null"):
        build_entrypoint(
            command="cmd {config_path}", specs={"value": None}, config_format="toml"
        )


def test_toml_allows_null_only_at_declared_output_paths():
    specs = {"train": {"results_dir": None}, "description": "test"}

    ep = build_entrypoint(
        command="cmd {config_path}",
        specs=specs,
        outputs={"train.results_dir": {"type": "folder"}},
        config_format="toml",
    )

    assert json.loads(_extract_arg(ep["command"], "--specs")) == specs


def test_toml_allows_composite_declared_output_placeholders():
    specs = {"output": {"nested": None}, "description": "test"}

    ep = build_entrypoint(
        command="cmd {config_path}",
        specs=specs,
        outputs={"output": {"type": "folder"}},
        config_format="toml",
    )

    assert json.loads(_extract_arg(ep["command"], "--specs")) == specs


def test_config_mode_rejects_unknown_format():
    with pytest.raises(ValueError, match="config_format"):
        build_entrypoint(
            command="cmd {config_path}", specs={"value": 1}, config_format="ini"
        )


def test_args_mode_normalizes_array_like_values():
    ep = build_entrypoint(
        command="cmd --values {values}",
        args={"values": np.array([np.int64(2), np.int64(4)])},
    )

    assert json.loads(_extract_arg(ep["command"], "--specs")) == {
        "values": [2, 4]
    }


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
