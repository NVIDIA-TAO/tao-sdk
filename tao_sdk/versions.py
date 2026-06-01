# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve TAO container image and wheel version references against ``versions.yaml``.

The skill bank declares its canonical image and wheel registry in a single file
(``versions.yaml`` at the bank's repo root). Skills reference images and wheels
by dotted key (e.g., ``tao_toolkit.pyt``); this module resolves the key to the
actual image string at runtime.

Both forms are valid in ``references/skill_info.yaml``:

    container_image: tao_toolkit.pyt                              # key reference (preferred)
    container_image: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt    # absolute path (also valid)

Discovery order for the manifest path:

  1. Explicit path passed to ``load_manifest(path=...)``.
  2. ``$TAO_VERSIONS_FILE`` env var.
  3. ``./versions.yaml`` from the current working directory.
  4. ``$TAO_SKILL_BANK_PATH/versions.yaml``.
  5. ``../tao-skills-external/versions.yaml`` (sibling of ``tao-sdk/``).

Example:

    >>> from tao_sdk.versions import resolve_image, looks_like_key
    >>> resolve_image("tao_toolkit.pyt")
    'nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt'
    >>> looks_like_key("tao_toolkit.pyt")
    True
    >>> looks_like_key("nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt")
    False
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class ManifestNotFoundError(FileNotFoundError):
    """Raised when ``versions.yaml`` cannot be located."""


class KeyNotFoundError(KeyError):
    """Raised when a key reference doesn't resolve in the manifest."""


def _candidate_paths() -> list[Path]:
    paths = []
    env_path = os.environ.get("TAO_VERSIONS_FILE")
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.cwd() / "versions.yaml")
    bank = os.environ.get("TAO_SKILL_BANK_PATH")
    if bank:
        paths.append(Path(bank) / "versions.yaml")
    # Submodule layout: tao-sdk/tao-skills-external/versions.yaml (the typical
    # dev checkout — skill bank is a submodule of the SDK repo).
    repo_root = Path(__file__).resolve().parent.parent
    paths.append(repo_root / "tao-skills-external" / "versions.yaml")
    # Sibling layout: tao-sdk/ next to tao-skills-external/.
    paths.append(repo_root.parent / "tao-skills-external" / "versions.yaml")
    return paths


def _default_path() -> Path:
    for p in _candidate_paths():
        if p.is_file():
            return p
    raise ManifestNotFoundError(
        "versions.yaml not found. Tried: "
        + ", ".join(str(p) for p in _candidate_paths())
        + ". Set $TAO_VERSIONS_FILE or $TAO_SKILL_BANK_PATH to point at the manifest."
    )


@lru_cache(maxsize=4)
def load_manifest(path: str | None = None) -> dict[str, Any]:
    """Load ``versions.yaml`` from the discovered or supplied path.

    Cached per path. Pass ``path`` to override discovery (useful for tests).
    """
    resolved = Path(path) if path else _default_path()
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{resolved}: top-level must be a mapping, got {type(data).__name__}")
    return data


def looks_like_key(value: str) -> bool:
    """Return True if ``value`` looks like a key reference (not an absolute image URI).

    A key reference is dotted, contains no ``/``, and has no characters that
    only appear in registry URIs.
    """
    if not value or not isinstance(value, str):
        return False
    if "/" in value:
        return False
    if ":" in value:
        return False
    # Dotted-key shape: at least `<group>.<name>`, alphanumeric + underscore + dot.
    return bool(value) and all(part.replace("_", "").isalnum() for part in value.split("."))


def resolve_image(key: str, manifest: dict[str, Any] | None = None) -> str:
    """Resolve ``tao_toolkit.pyt`` to its image string from the manifest's ``images`` tree.

    Raises:
        KeyNotFoundError: when the key doesn't resolve.
        TypeError: when an intermediate node isn't a mapping.
    """
    return _resolve(key, manifest, "images")


def resolve_wheel(key: str, manifest: dict[str, Any] | None = None) -> str:
    """Resolve ``tao_sdk_brev`` to its wheel spec from the manifest's ``wheels`` tree."""
    return _resolve(key, manifest, "wheels")


def _resolve(key: str, manifest: dict[str, Any] | None, group: str) -> str:
    m = manifest if manifest is not None else load_manifest()
    node: Any = m.get(group, {})
    if not isinstance(node, dict):
        raise TypeError(f"versions.yaml `{group}` is not a mapping")
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyNotFoundError(f"`{group}` key not found in versions.yaml: {key}")
        node = node[part]
    if not isinstance(node, str):
        raise TypeError(f"`{group}` key {key!r} resolves to non-string: {type(node).__name__}")
    return node


def resolve_container_image(value: str, manifest: dict[str, Any] | None = None) -> str:
    """Resolve a ``container_image`` field value, accepting either form.

    Returns the value unchanged if it's an absolute image URI; resolves it
    via the manifest if it looks like a key reference.
    """
    if looks_like_key(value):
        return resolve_image(value, manifest)
    return value
