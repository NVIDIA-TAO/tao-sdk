# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for cross-format configuration normalization."""

from __future__ import annotations

import json
from decimal import Decimal
from enum import Enum

import numpy as np
import pytest
import yaml

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

from tao_sdk.platforms.virtualenv import VirtualEnvSDK
from tao_sdk.spec_serialization import normalize_specs


class _Mode(Enum):
    FAST = "fast"


@pytest.mark.parametrize("config_format", ["json", "yaml", "toml"])
def test_write_spec_normalizes_supported_values_across_formats(
    config_format, tmp_path,
):
    raw = {
        "train": {
            "lr": np.float64(0.125),
            "epochs": np.int64(3),
            "enabled": np.bool_(True),
            "name": np.str_("model"),
            "shape": np.array([16, 32], dtype=np.int32),
            "metadata": (tmp_path / "data", _Mode.FAST),
        }
    }
    expected = {
        "train": {
            "lr": 0.125,
            "epochs": 3,
            "enabled": True,
            "name": "model",
            "shape": [16, 32],
            "metadata": [str(tmp_path / "data"), "fast"],
        }
    }
    config_path = tmp_path / f"spec.{config_format}"

    VirtualEnvSDK._write_spec(config_path, raw, config_format)

    content = config_path.read_text(encoding="utf-8")
    if config_format == "json":
        parsed = json.loads(content)
    elif config_format == "yaml":
        parsed = yaml.safe_load(content)
    else:
        parsed = tomllib.loads(content)
    assert parsed == expected
    assert isinstance(raw["train"]["epochs"], np.integer)
    assert isinstance(raw["train"]["shape"], np.ndarray)
    assert isinstance(raw["train"]["metadata"], tuple)


def test_normalize_specs_rejects_non_string_keys_with_location():
    with pytest.raises(TypeError, match=r"specs\['train'\].*keys must be strings"):
        normalize_specs({"train": {1: "value"}})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), np.float64("-inf")],
)
def test_normalize_specs_rejects_non_finite_numbers_with_location(value):
    with pytest.raises(ValueError, match=r"specs\['train'\]\['lr'\].*finite"):
        normalize_specs({"train": {"lr": value}})


def test_normalize_specs_rejects_cycles_with_location():
    cyclic = {}
    cyclic["nested"] = cyclic

    with pytest.raises(ValueError, match=r"specs\['nested'\].*circular"):
        normalize_specs(cyclic)


def test_normalize_specs_rejects_unsupported_values_with_location():
    with pytest.raises(TypeError, match=r"specs\['lr'\].*decimal.Decimal"):
        normalize_specs({"lr": Decimal("0.1")})


def test_toml_rejects_null_with_location(tmp_path):
    with pytest.raises(TypeError, match=r"specs\['train'\]\['value'\].*null"):
        VirtualEnvSDK._write_spec(
            tmp_path / "spec.toml",
            {"train": {"value": None}},
            "toml",
        )
