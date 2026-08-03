# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize user-provided configuration values at SDK serialization boundaries."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any


def _child_path(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}[{key!r}]"


def _type_name(value: Any) -> str:
    value_type = type(value)
    module = value_type.__module__
    if module == "builtins":
        return value_type.__name__
    return f"{module}.{value_type.__name__}"


def _is_numpy_value(value: Any) -> bool:
    module = type(value).__module__
    return module == "numpy" or module.startswith("numpy.")


def _normalize_value(
    value: Any,
    *,
    path: str,
    allow_none: bool,
    active_containers: set[int],
) -> Any:
    if _is_numpy_value(value):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path}: circular reference in configuration value")
        active_containers.add(identity)
        try:
            converter = getattr(value, "tolist", None)
            if not callable(converter):
                converter = getattr(value, "item", None)
            if not callable(converter):
                raise TypeError(f"{path}: unsupported value type {_type_name(value)}")
            converted = converter()
            if type(converted) is type(value):
                raise TypeError(
                    f"{path}: {_type_name(value)} cannot be converted to a native Python value"
                )
            return _normalize_value(
                converted,
                path=path,
                allow_none=allow_none,
                active_containers=active_containers,
            )
        finally:
            active_containers.remove(identity)

    if isinstance(value, Enum):
        return _normalize_value(
            value.value,
            path=path,
            allow_none=allow_none,
            active_containers=active_containers,
        )
    if isinstance(value, os.PathLike):
        converted = os.fspath(value)
        if not isinstance(converted, str):
            raise TypeError(f"{path}: path-like values must resolve to strings")
        return converted
    if value is None:
        if not allow_none:
            raise TypeError(f"{path}: TOML does not support null values")
        return None
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{path}: numeric values must be finite")
        return converted

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path}: circular reference in configuration value")
        active_containers.add(identity)
        try:
            normalized: dict[str, Any] = {}
            for key, nested in value.items():
                if isinstance(key, Enum):
                    key = key.value
                if not isinstance(key, str):
                    raise TypeError(
                        f"{path}: mapping keys must be strings, got {_type_name(key)}"
                    )
                normalized_key = str(key)
                if normalized_key in normalized:
                    raise ValueError(
                        f"{path}: duplicate mapping key after normalization: {normalized_key!r}"
                    )
                normalized[normalized_key] = _normalize_value(
                    nested,
                    path=_child_path(path, normalized_key),
                    allow_none=allow_none,
                    active_containers=active_containers,
                )
            return normalized
        finally:
            active_containers.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path}: circular reference in configuration value")
        active_containers.add(identity)
        try:
            return [
                _normalize_value(
                    nested,
                    path=_child_path(path, index),
                    allow_none=allow_none,
                    active_containers=active_containers,
                )
                for index, nested in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)

    raise TypeError(f"{path}: unsupported value type {_type_name(value)}")


def normalize_data(value: Any, *, root: str, allow_none: bool = True) -> Any:
    """Return a fresh tree containing only strict JSON-compatible native values."""
    return _normalize_value(
        value,
        path=root,
        allow_none=allow_none,
        active_containers=set(),
    )


def normalize_specs(specs: dict | None, *, allow_none: bool = True) -> dict:
    """Validate and normalize the public ``specs`` argument."""
    if specs is None:
        return {}
    if not isinstance(specs, dict):
        raise TypeError(f"specs must be a dictionary or None, got {_type_name(specs)}")
    return normalize_data(specs, root="specs", allow_none=allow_none)
