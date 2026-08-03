#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight job runner with S3 I/O via fsspec.

Replaces TAO Core's container_handler. Downloads declared inputs from S3,
rewrites spec paths to local, runs the command, uploads results. Supports
selective download (only annotation-referenced files) and Lustre/NFS mounts.

Gets inlined into containers at runtime via heredoc — no container rebuild
or dependency installation needed beyond fsspec.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

try:
    import fsspec
except ImportError:
    # fsspec is a container-runtime dependency (installed via _DEFAULT_INSTALL_DEPS
    # inside the job container, where this module is inlined via heredoc). It is
    # deliberately not an SDK dependency: the SDK process only calls
    # build_entrypoint, which never touches fsspec. Bind the name so the module
    # imports in environments without fsspec (dev, CI, downstream like tao-automl).
    # The in-container runtime helpers that use it will fail clearly if it is
    # genuinely absent at runtime.
    fsspec = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _is_remote(path: str) -> bool:
    return "://" in str(path)


def _is_ngc(path: str) -> bool:
    return str(path).startswith("ngc://")


def _is_hf(path: str) -> bool:
    return str(path).startswith("hf_model://") or str(path).startswith("hf_dataset://")


def download_ngc_model(ngc_path: str, destination: str):
    """Download a model from NGC Model Registry using the NGC SDK.

    The NGC SDK (ngcsdk) is pre-installed in TAO toolkit containers.

    Args:
        ngc_path: NGC URI in format 'ngc://org/team/model:version'
                  or just 'org/team/model:version'.
        destination: Local directory to download to.
    """
    # Strip ngc:// prefix if present
    model_path = ngc_path.replace("ngc://", "")

    ngc_key = os.environ.get("NGC_KEY", "")
    if not ngc_key:
        raise ValueError("NGC_KEY environment variable required for NGC model download")

    os.makedirs(destination, exist_ok=True)

    # Check if already downloaded
    if os.listdir(destination):
        logger.info("NGC model already present at %s", destination)
        return destination

    # Parse org/team/model:version
    parts = model_path.split("/")
    if len(parts) == 3:
        org, team, model_ver = parts
    elif len(parts) == 2:
        org, model_ver = parts
        team = ""
    else:
        raise ValueError(f"Invalid NGC path format: {model_path}")

    from ngcsdk import Client
    clt = Client()
    try:
        clt.configure(api_key=ngc_key, org_name=org, team_name=team)
    except Exception as e:
        # Org validation may fail if the key isn't from this org.
        # For public models (e.g., nvidia/tao/*), continue with the
        # unconfigured client — it can still download public models.
        # This matches TAO Core's allow_fallback pattern.
        if "Invalid org" in str(e) or "Invalid team" in str(e) or "doesn't match" in str(e):
            logger.info("NGC key not from org '%s', continuing with unconfigured client "
                        "(public model download)", org)
        else:
            raise
    clt.registry.model.download_version(model_path, destination=destination)
    logger.info("Downloaded NGC model: %s → %s", model_path, destination)
    return destination


def download_hf_model(hf_path: str, destination: str):
    """Download a model or dataset from HuggingFace Hub.

    Args:
        hf_path: HuggingFace URI in format 'hf_model://org/model'
                 or 'hf_dataset://org/dataset'.
        destination: Local directory to download to.
    """
    is_dataset = hf_path.startswith("hf_dataset://")
    repo_id = hf_path.replace("hf_model://", "").replace("hf_dataset://", "")

    hf_token = os.environ.get("HF_TOKEN", "")

    os.makedirs(destination, exist_ok=True)

    # Check if already downloaded
    if os.listdir(destination):
        logger.info("HF model already present at %s", destination)
        return destination

    try:
        from huggingface_hub import snapshot_download
        kwargs = {
            "repo_id": repo_id,
            "local_dir": destination,
        }
        if is_dataset:
            kwargs["repo_type"] = "dataset"
        if hf_token:
            kwargs["token"] = hf_token
        snapshot_download(**kwargs)
        logger.info("Downloaded HF model: %s → %s", repo_id, destination)
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    return destination


def download_remote(remote_path: str, local_path: str):
    """Route download to the correct handler based on URI scheme.

    Supports: s3://, ngc://, hf_model://, hf_dataset://
    """
    if _is_ngc(remote_path):
        return download_ngc_model(remote_path, local_path)
    elif _is_hf(remote_path):
        return download_hf_model(remote_path, local_path)
    else:
        # S3 / fsspec (existing behavior)
        download_file(remote_path, local_path)
        return local_path


def _download_ptm_from_specs(specs: dict, results_dir: str, _prefix: str = ""):
    """Scan spec values for NGC/HuggingFace URIs and download them.

    Recursively walks the spec dict. When a value starts with ngc:// or
    hf_model://, downloads the model to a local path inside the container
    and rewrites the spec value to the local path.

    This mirrors TAO Core's download_files_from_spec() behavior for
    pretrained model resolution.
    """
    if not isinstance(specs, dict):
        return

    for key, value in specs.items():
        full_key = f"{_prefix}.{key}" if _prefix else key

        if isinstance(value, dict):
            _download_ptm_from_specs(value, results_dir, full_key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                item_key = f"{full_key}[{i}]"
                if isinstance(item, dict):
                    _download_ptm_from_specs(item, results_dir, item_key)
                elif isinstance(item, str) and (_is_ngc(item) or _is_hf(item)):
                    value[i] = _stage_ptm_uri(item, results_dir, item_key)
        elif isinstance(value, str) and (_is_ngc(value) or _is_hf(value)):
            specs[key] = _stage_ptm_uri(value, results_dir, full_key)


def _stage_ptm_uri(value: str, results_dir: str, full_key: str) -> str:
    """Download one ngc:// or hf_* URI and return the local path to use."""
    # Download to a deterministic local path
    safe_key = full_key.replace(".", "_").replace("[", "_").replace("]", "")
    local_dir = os.path.join(results_dir, "ptm", safe_key)
    logger.info("Downloading PTM for %s: %s → %s", full_key, value, local_dir)
    download_remote(value, local_dir)

    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        raise RuntimeError(f"PTM download produced no files at {local_dir}")

    # NGC downloads of single-file checkpoints (.tlt, .pth, etc.) need
    # the file path, not the dir. HF multi-shard downloads need the dir
    # (config.json + index + shards) so from_pretrained() can read it.
    # Match TAO Core's search_for_ptm extension list — .safetensors and
    # .bin are intentionally excluded so HF dirs fall through to the
    # ``local_path = local_dir`` default below.
    local_path = local_dir
    if os.path.isdir(local_dir):
        model_extensions = ('.pth', '.pth.tar', '.pt', '.ckpt', '.hdf5', '.tlt')
        for root, dirs, files in os.walk(local_dir):
            for f in files:
                if any(f.endswith(ext) for ext in model_extensions):
                    local_path = os.path.join(root, f)
                    logger.info("Found model file: %s", local_path)
                    break
            if local_path != local_dir:
                break

    logger.info("Rewrote spec %s → %s", full_key, local_path)
    return local_path


_RESUME_LEAF_KEY = "resume_training_checkpoint_path"
_CHECKPOINT_EXTENSIONS = (".pth", ".pth.tar", ".pt", ".ckpt")


def _iter_resume_leaves(specs: dict, _prefix: str = ""):
    for key, value in specs.items():
        full_key = f"{_prefix}.{key}" if _prefix else key
        if isinstance(value, dict):
            yield from _iter_resume_leaves(value, full_key)
        elif key == _RESUME_LEAF_KEY:
            yield full_key, value


def _find_latest_checkpoint(search_roots) -> str | None:
    best = None
    for root_dir in search_roots:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            # A staged pretrained model or downloaded input is not a
            # training-state checkpoint.
            dirs[:] = [d for d in dirs if d not in ("ptm", "inputs")]
            for name in files:
                if not name.endswith(_CHECKPOINT_EXTENSIONS):
                    continue
                path = os.path.join(root, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                if stat.st_size == 0:
                    continue
                candidate = (stat.st_mtime, path)
                if best is None or candidate > best:
                    best = candidate
    return best[1] if best else None


def _apply_auto_resume(specs: dict, outputs) -> str | None:
    """Point empty resume keys at the latest checkpoint from a prior attempt.

    SLURM wall-time requeue and node-failure requeue re-run the identical
    entrypoint with the spec frozen at create_job() time, so a resumable job
    would otherwise restart from epoch 0. Declared output dirs are job-scoped:
    a checkpoint can exist there only if an earlier attempt of this same job
    wrote it. Only output dirs are searched — inputs and PTMs stage elsewhere
    (and are pruned defensively) so a freshly downloaded model file can never
    masquerade as training state. Set TAO_AUTO_RESUME=0 to disable.
    """
    if os.environ.get("TAO_AUTO_RESUME", "1").lower() in ("0", "false", "no"):
        return None
    empty_leaves = [key for key, value in _iter_resume_leaves(specs) if not value]
    if not empty_leaves:
        return None

    output_keys = list(outputs.keys()) if isinstance(outputs, dict) else list(outputs or [])
    search_roots = []
    for key in output_keys:
        value = _get_nested(specs, key)
        if isinstance(value, str) and value.startswith("/"):
            search_roots.append(value)

    checkpoint = _find_latest_checkpoint(search_roots)
    if not checkpoint:
        return None
    for key in empty_leaves:
        _set_nested(specs, key, checkpoint)
        logger.info(
            "Auto-resume: found checkpoint from a previous attempt of this "
            "job; setting %s → %s", key, checkpoint,
        )
    return checkpoint


def _s3_to_lustre(remote_path: str, mount_path: str) -> str:
    """Map an S3 URI to a deterministic Lustre path.

    s3://bucket/path/to/file → {mount_path}/bucket/path/to/file
    """
    after_proto = remote_path.split("://", 1)[1]
    return os.path.join(mount_path, after_proto)


def _parse_dot_part(part: str):
    """Parse a dot-path segment, returning (key, index) or (key, None).

    Examples: "datasets[0]" -> ("datasets", 0), "train" -> ("train", None)
    """
    import re
    m = re.match(r'^(\w+)\[(\d+)\]$', part)
    if m:
        return m.group(1), int(m.group(2))
    return part, None


def _get_nested(d: dict, dotted_key: str):
    """Get a value from a nested dict using dot-notation key.

    Supports array indices: "dataset.train.datasets[0].image_dir"
    """
    parts = dotted_key.split(".")
    for part in parts:
        key, idx = _parse_dot_part(part)
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
        if idx is not None:
            if isinstance(d, list) and idx < len(d):
                d = d[idx]
            else:
                return None
    return d


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using dot-notation key.

    Supports array indices: "dataset.train.datasets[0].image_dir"
    """
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        key, idx = _parse_dot_part(part)
        if key not in d:
            d[key] = {} if idx is None else []
        d = d[key]
        if idx is not None:
            while len(d) <= idx:
                d.append({})
            d = d[idx]
    last_key, last_idx = _parse_dot_part(parts[-1])
    if last_idx is not None:
        if last_key not in d:
            d[last_key] = []
        while len(d[last_key]) <= last_idx:
            d[last_key].append(None)
        d[last_key][last_idx] = value
    else:
        d[last_key] = value


def _parse_selective_config(input_cfg: dict) -> tuple[str | None, str, list[str]]:
    """Parse the selective download config from an input declaration.

    Format: {"selective": {"annotation": "key", "format": "json", "keys": ["video"]}}

    Returns (annotation_ref, format, keys) or (None, "", []) if not selective.
    """
    selective = input_cfg.get("selective")
    if not isinstance(selective, dict):
        return None, "", []
    return (
        selective.get("annotation"),
        selective.get("format", "json"),
        selective.get("keys", ["video"]),
    )


def _input_local_base(
    results_dir: str,
    spec_key: str,
    input_cfg: dict,
    idx: int | None = None,
) -> str:
    """Return the local/container base path for a spec input."""
    configured = input_cfg.get("container_path") if isinstance(input_cfg, dict) else None
    if configured:
        return os.path.join(configured, str(idx)) if idx is not None else configured
    parts = [results_dir, "inputs", spec_key.replace(".", "_")]
    if idx is not None:
        parts.append(str(idx))
    return os.path.join(*parts)


def download_file(remote_path: str, local_path: str):
    """Download a single file from S3 to local."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        logger.info("Already exists: %s", local_path)
        return
    with fsspec.open(remote_path, "rb") as src, open(local_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    logger.info("Downloaded: %s → %s", remote_path, local_path)


_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")


def _is_archive(path: str) -> bool:
    return any(str(path).endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def _strip_archive_suffix(path: str) -> str:
    for suffix in _ARCHIVE_SUFFIXES:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def extract_archive(tar_path: str, dest_dir: str, extract_to_parent: bool = False):
    """Extract a tar/tar.gz archive into a folder named after the archive.

    Idempotent via a sibling `<tar_path>.extracted` marker. Uses the tarfile
    module directly so archives with root-level files and archives with a
    matching top-level folder both land in the same output directory.

    Mirrors TAO Core's _extract_images / download_from_user_storage semantics:
    the caller is responsible for pointing the spec at the extracted directory
    (`<tar_path>` with the archive suffix stripped). For example,
    `images.tar.gz` is extracted into `images/`.
    """
    import tarfile
    marker = tar_path + ".extracted"
    output_dir = (
        dest_dir
        if extract_to_parent
        else os.path.join(dest_dir, os.path.basename(_strip_archive_suffix(tar_path)))
    )

    if os.path.exists(marker) and os.path.isdir(output_dir) and os.listdir(output_dir):
        logger.info("Archive already extracted: %s", tar_path)
        return
    if os.path.exists(marker):
        logger.info("Archive marker is stale, re-extracting: %s", tar_path)
        os.remove(marker)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Extracting %s → %s", tar_path, output_dir)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        archive_root = os.path.basename(output_dir)
        names = [m.name.strip("/") for m in members if m.name.strip("/")]
        first_parts = {name.split("/", 1)[0] for name in names}
        strip_root = first_parts == {archive_root}

        for member in members:
            raw_name = member.name.strip("/")
            if not raw_name:
                continue
            if strip_root:
                if raw_name == archive_root:
                    continue
                rel_name = raw_name.split("/", 1)[1] if raw_name.startswith(archive_root + "/") else raw_name
            else:
                rel_name = raw_name
            if not rel_name:
                continue
            rel_path = os.path.normpath(rel_name)
            if rel_path.startswith("..") or os.path.isabs(rel_path):
                raise ValueError(f"Unsafe archive member path: {member.name}")

            target = os.path.join(output_dir, rel_path)
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            if not member.isfile():
                logger.warning("Skipping unsupported archive member: %s", member.name)
                continue

            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    with open(marker, "w") as f:
        f.write("ok")
    logger.info("Extraction complete: %s", tar_path)


def extract_archives_in_dir(directory: str, extract_to_parent: bool = False):
    """Walk a directory and extract any tar/tar.gz files in place."""
    for root, _, files in os.walk(directory):
        for f in files:
            if _is_archive(f):
                extract_archive(os.path.join(root, f), root, extract_to_parent=extract_to_parent)


def download_folder(remote_path: str, local_path: str):
    """Download an entire folder from S3 to local."""
    fs, _ = fsspec.core.url_to_fs(remote_path)
    root = remote_path.split("://", 1)[1].rstrip("/")
    proto = remote_path.split("://")[0]
    # Use fs.find() instead of fs.glob() — find is fully recursive
    # and works correctly with the old s3fs version in TAO containers.
    files = fs.find(root)
    count = 0
    for f in files:
        if fs.isdir(f):
            continue
        rel = f[len(root):].lstrip("/")
        local_file = os.path.join(local_path, rel)
        remote_file = f"{proto}://{f}"
        download_file(remote_file, local_file)
        count += 1
    logger.info("Downloaded folder: %s (%d files)", remote_path, count)


def parse_references(annotation_path: str, fmt: str = "json",
                     keys: list[str] | None = None) -> set[str]:
    """Extract file references from an annotation file.

    Supports multiple formats — all model-specific parsing is driven
    by the skill config, not hardcoded here.

    Args:
        annotation_path: Local path to the annotation file.
        fmt: Annotation format ('json', 'jsonl', 'csv', 'parquet').
        keys: Column/field names to extract as file paths.

    Returns:
        Set of relative file paths referenced in the annotation.
    """
    keys = keys or ["video"]
    paths = set()

    if fmt == "json":
        with open(annotation_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for entry in data:
            for key in keys:
                val = entry.get(key, "")
                if val:
                    paths.add(val)

    elif fmt == "jsonl":
        with open(annotation_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for key in keys:
                    val = entry.get(key, "")
                    if val:
                        paths.add(val)

    elif fmt == "csv":
        import csv
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in keys:
                    val = row.get(key, "")
                    if val:
                        paths.add(val)

    elif fmt == "parquet":
        import pandas as pd
        df = pd.read_parquet(annotation_path)
        for key in keys:
            if key in df.columns:
                paths.update(df[key].dropna().astype(str).tolist())

    return paths


def download_selective(remote_dir: str, local_dir: str,
                       annotation_data=None, video_key: str = "video",
                       annotation_path: str = None, fmt: str = "json",
                       keys: list[str] | None = None):
    """Download only files referenced in the annotation.

    Supports two calling patterns:
      - Legacy: annotation_data (list) + video_key (string)
      - New: annotation_path (file) + fmt + keys (from skill config)
    """
    if annotation_path and os.path.exists(annotation_path):
        paths = parse_references(annotation_path, fmt, keys)
    elif annotation_data is not None:
        # Legacy: in-memory annotation data with single key
        paths = set()
        for entry in annotation_data:
            val = entry.get(video_key, "")
            if val:
                paths.add(val)
    else:
        logger.warning("No annotation data for selective download")
        return

    root = remote_dir.rstrip("/")
    logger.info("Selective download: %d files from %s", len(paths), remote_dir)
    for rel_path in paths:
        remote_file = f"{root}/{rel_path}"
        local_file = os.path.join(local_dir, rel_path)
        download_file(remote_file, local_file)


def upload_results(local_dir: str, remote_dir: str,
                   excludes: list[str] | None = None):
    """Upload local results to S3, excluding patterns."""
    import time
    excludes = excludes or []

    # Count files first for progress tracking
    files_to_upload = []
    for dirpath, _, filenames in os.walk(local_dir):
        for fname in filenames:
            local_file = os.path.join(dirpath, fname)
            rel = os.path.relpath(local_file, local_dir)
            if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                continue
            files_to_upload.append((local_file, rel))

    total = len(files_to_upload)
    if total == 0:
        logger.info("No files to upload from %s", local_dir)
        return

    total_bytes = sum(os.path.getsize(f) for f, _ in files_to_upload)
    logger.info("Uploading %d files (%.1f MB) to %s",
                total, total_bytes / 1024 / 1024, remote_dir)

    uploaded_bytes = 0
    start_time = time.time()
    for i, (local_file, rel) in enumerate(files_to_upload, 1):
        file_size = os.path.getsize(local_file)
        remote_file = f"{remote_dir.rstrip('/')}/{rel}"
        with open(local_file, "rb") as src, fsspec.open(remote_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        uploaded_bytes += file_size
        elapsed = time.time() - start_time
        rate_mbps = (uploaded_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
        remaining = total - i
        eta = (remaining * elapsed / i) if i > 0 else 0
        logger.info("Uploaded [%d/%d] %s (%.1f MB, %.1f MB/s, ETA %.0fs)",
                     i, total, rel, file_size / 1024 / 1024, rate_mbps, eta)

    elapsed = time.time() - start_time
    logger.info("Upload complete: %d files, %.1f MB in %.0fs (%.1f MB/s)",
                total, total_bytes / 1024 / 1024, elapsed,
                (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0)


def run(specs: dict, job_id: str, command: str, inputs: dict,
        outputs: dict, config_format: str = "toml",
        upload_excludes: list[str] | None = None,
        mount_path: str | None = None):
    """Run a job with S3 I/O.

    Args:
        specs: The spec dict with S3 paths.
        job_id: Unique job identifier.
        command: Command template (use {config_path} for spec file).
        inputs: Input declarations from skill config.
        outputs: Output declarations from skill config.
        config_format: How to write the spec file (toml, yaml, json).
        upload_excludes: Patterns to exclude from upload.
        mount_path: If set, use Lustre/NFS mount instead of downloading.
    """
    results_root, results_kind = _resolve_results_root(job_id)
    results_dir = results_root if results_kind == "mount" else f"/results/{job_id}"
    os.makedirs(results_dir, exist_ok=True)

    # Phase 1: Download inputs and rewrite spec paths
    annotation_cache = {}  # cache parsed annotations for selective download

    # Normalize inputs: accept dict ({"key": {...}}), list (["key"]), or mixed
    if isinstance(inputs, list):
        input_keys = {k: {} for k in inputs}
    elif isinstance(inputs, dict):
        input_keys = inputs
    else:
        input_keys = {}

    for spec_key, input_cfg in input_keys.items():
        remote_path = _get_nested(specs, spec_key)
        if isinstance(remote_path, list):
            final_paths = []
            for idx, item in enumerate(remote_path):
                if (not item or not _is_remote(item)
                        or _is_ngc(item) or _is_hf(item)):
                    # NGC/HF URIs are staged scheme-aware by
                    # _download_ptm_from_specs; the fsspec path below
                    # raises "protocol not known" on them.
                    final_paths.append(item)
                    continue
                local_base = _input_local_base(results_dir, spec_key, input_cfg, idx)
                try:
                    fs, _ = fsspec.core.url_to_fs(item)
                    remote_key = item.split("://", 1)[1]
                    input_type = "folder" if fs.isdir(remote_key) else "file"
                except Exception:
                    input_type = "folder" if item.endswith("/") else "file"
                if input_type == "file":
                    fname = os.path.basename(item.rstrip("/"))
                    local_file = os.path.join(local_base, fname)
                    download_file(item, local_file)
                    final_path = local_file
                    if _is_archive(local_file):
                        extract_archive(local_file, os.path.dirname(local_file))
                        final_path = _strip_archive_suffix(local_file)
                    final_paths.append(final_path)
                else:
                    download_folder(item, local_base)
                    extract_archives_in_dir(
                        local_base,
                        extract_to_parent=bool(input_cfg.get("extract_to_parent")),
                    )
                    final_paths.append(local_base)
            _set_nested(specs, spec_key, final_paths)
            continue
        if not remote_path or not _is_remote(remote_path):
            continue
        if _is_ngc(remote_path) or _is_hf(remote_path):
            # NGC/HF URIs have dedicated downloaders; the generic fsspec
            # path below raises "protocol not known" on them. They are
            # staged (and the spec key rewritten) by _download_ptm_from_specs.
            continue

        # Auto-detect file vs folder from the actual remote path
        try:
            fs, _ = fsspec.core.url_to_fs(remote_path)
            remote_key = remote_path.split("://", 1)[1]
            input_type = "folder" if fs.isdir(remote_key) else "file"
        except Exception:
            # Fallback: infer from path pattern
            input_type = "folder" if remote_path.endswith("/") else "file"

        # Determine local destination: Lustre (persistent) or container disk (ephemeral)
        if mount_path:
            lustre_path = _s3_to_lustre(remote_path, mount_path)
            if input_type == "file":
                # For files, lustre_path is the exact file location
                if not os.path.exists(lustre_path):
                    # Miss: download from S3 to Lustre (persists for future jobs)
                    logger.info("Lustre miss, downloading: %s → %s", remote_path, lustre_path)
                    download_file(remote_path, lustre_path)
                else:
                    logger.info("Lustre hit: %s", lustre_path)
                final_path = lustre_path
                if _is_archive(lustre_path):
                    extract_archive(lustre_path, os.path.dirname(lustre_path))
                    final_path = _strip_archive_suffix(lustre_path)
                _set_nested(specs, spec_key, final_path)
                if lustre_path.endswith(".json"):
                    with open(lustre_path) as f:
                        annotation_cache[spec_key] = json.load(f)
                continue

            elif input_type == "folder":
                # For folders, lustre_path is the directory
                ann_ref, sel_fmt, sel_keys = _parse_selective_config(input_cfg)
                video_key = sel_keys[0] if sel_keys else "video"
                if ann_ref and ann_ref in annotation_cache:
                    # Check if all referenced files exist on Lustre
                    paths = {e.get(video_key, "") for e in annotation_cache[ann_ref] if e.get(video_key)}
                    all_present = all(os.path.exists(os.path.join(lustre_path, p)) for p in paths)
                    if all_present:
                        logger.info("Lustre hit (selective, %d files): %s", len(paths), lustre_path)
                        _set_nested(specs, spec_key, lustre_path)
                        continue
                    # Miss: selectively download to Lustre
                    logger.info("Lustre miss (selective), downloading %d files to %s", len(paths), lustre_path)
                    download_selective(remote_path, lustre_path,
                                       annotation_cache[ann_ref], video_key)
                else:
                    if os.path.exists(lustre_path) and os.listdir(lustre_path):
                        logger.info("Lustre hit (folder): %s", lustre_path)
                        extract_archives_in_dir(
                            lustre_path,
                            extract_to_parent=bool(input_cfg.get("extract_to_parent")),
                        )
                        _set_nested(specs, spec_key, lustre_path)
                        continue
                    logger.info("Lustre miss (folder), downloading: %s → %s", remote_path, lustre_path)
                    download_folder(remote_path, lustre_path)
                extract_archives_in_dir(
                    lustre_path,
                    extract_to_parent=bool(input_cfg.get("extract_to_parent")),
                )
                _set_nested(specs, spec_key, lustre_path)
                continue

        # No mount: download to container-local disk
        local_base = _input_local_base(results_dir, spec_key, input_cfg)

        if input_type == "file":
            fname = os.path.basename(remote_path.rstrip("/"))
            local_file = os.path.join(local_base, fname)
            download_file(remote_path, local_file)
            final_path = local_file
            if _is_archive(local_file):
                extract_archive(local_file, os.path.dirname(local_file))
                final_path = _strip_archive_suffix(local_file)
            _set_nested(specs, spec_key, final_path)
            if local_file.endswith(".json"):
                with open(local_file) as f:
                    annotation_cache[spec_key] = json.load(f)

        elif input_type == "folder":
            ann_ref, sel_fmt, sel_keys = _parse_selective_config(input_cfg)
            video_key = sel_keys[0] if sel_keys else "video"
            if ann_ref and ann_ref in annotation_cache:
                download_selective(remote_path, local_base,
                                   annotation_cache[ann_ref], video_key)
            else:
                download_folder(remote_path, local_base)
            extract_archives_in_dir(
                local_base,
                extract_to_parent=bool(input_cfg.get("extract_to_parent")),
            )
            _set_nested(specs, spec_key, local_base)

    # Normalize outputs: accept dict or list
    if isinstance(outputs, list):
        output_keys = {k: {} for k in outputs}
    elif isinstance(outputs, dict):
        output_keys = outputs
    else:
        output_keys = {}

    # Resolve where outputs should land (platform policy via env vars; not
    # the agent's concern). Auto-fill any model-natural relative defaults
    # like cosmos-rl's ``output_dir: "output"``. Agent URIs / absolute paths
    # are preserved.
    _autofill_output_destinations(specs, None, output_keys, results_root)

    # Three branches:
    #   1. Remote URI → rewrite spec to local; queue upload back at end of run.
    #   2. Absolute local path (mount or ephemeral) → use directly; no upload.
    #   3. Anything else → defensive raise (auto-fill should have prevented).
    output_remote_map = {}
    for spec_key, output_cfg in output_keys.items():
        remote_path = _get_nested(specs, spec_key)
        output_type = output_cfg.get("type", "folder") if isinstance(output_cfg, dict) else "folder"
        if remote_path and _is_remote(remote_path):
            if mount_path:
                local_path = _s3_to_lustre(remote_path, mount_path)
            else:
                local_path = os.path.join(results_dir, spec_key.replace(".", "_"))
            if output_type == "folder":
                os.makedirs(local_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
            output_remote_map[local_path] = remote_path
            _set_nested(specs, spec_key, local_path)
        elif isinstance(remote_path, str) and remote_path.startswith("/"):
            local_path = remote_path
            if output_type == "folder":
                os.makedirs(local_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _set_nested(specs, spec_key, local_path)
        else:
            raise RuntimeError(
                f"Output spec key {spec_key!r} has unexpected non-string value "
                f"{remote_path!r}. Auto-fill should have produced an absolute "
                f"path; this likely means the spec was constructed with an "
                f"unsupported type at this key."
            )

    # Phase 1.5: Download NGC/HuggingFace models referenced in spec values.
    # Scan all spec values for ngc:// or hf_model:// URIs and download them
    # to local paths inside the container. This mirrors TAO Core's
    # download_files_from_spec() behavior.
    _download_ptm_from_specs(specs, results_dir)

    # Phase 1.6: Resume from a previous attempt of this job when one left a
    # checkpoint in a declared output dir (wall-time requeue, node-failure
    # requeue). Runs after output routing so output keys hold real paths.
    _apply_auto_resume(specs, outputs)

    # Phase 2: Write config file
    config_path = os.path.join(results_dir, f"spec.{config_format}")
    if config_format == "toml":
        import toml
        with open(config_path, "w") as f:
            toml.dump(specs, f)
    elif config_format == "yaml":
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(specs, f)
    else:
        with open(config_path, "w") as f:
            json.dump(specs, f, indent=2)
    logger.info("Config written: %s", config_path)

    # Phase 3: Run command
    resolved_command = command.format(config_path=config_path)
    logger.info("Running: %s", resolved_command)
    result = subprocess.run(resolved_command, shell=True)
    if result.returncode != 0:
        logger.error("Command failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    # Phase 4: Upload outputs
    uploaded = 0
    for local_path, remote_path in output_remote_map.items():
        if remote_path and os.path.isdir(local_path):
            upload_results(local_path, remote_path, upload_excludes)
            uploaded += 1
        elif remote_path and os.path.isfile(local_path):
            with open(local_path, "rb") as src, fsspec.open(remote_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("Uploaded: %s", remote_path)
            uploaded += 1
    _log_outputs_summary(output_keys, uploaded, results_root, results_kind)

    logger.info("Job complete.")


def run_args(args_dict: dict, job_id: str, command: str, inputs: dict,
             outputs: dict, upload_excludes: list[str] | None = None,
             mount_path: str | None = None,
             flags: list[str] | None = None):
    """Run a job with S3 I/O in args mode.

    Like run(), but for scripts that take CLI args instead of a config file.
    The args_dict is a flat dict of arg_name → value, and the command is
    invoked with --arg_name value for each entry. Flags are appended as
    --flag_name with no value (for argparse store_true arguments).

    Args:
        args_dict: Flat dict of CLI arg names to values (may contain S3 URIs).
        flags: List of flag names to append as --name (no value).
        job_id: Unique job identifier.
        command: Base command (e.g., "python /path/to/script.py").
        inputs: Input declarations keyed by arg name.
        outputs: Output declarations keyed by arg name.
        upload_excludes: Patterns to exclude from upload.
        mount_path: If set, use Lustre/NFS mount for S3 paths.
    """
    results_root, results_kind = _resolve_results_root(job_id)
    results_dir = results_root if results_kind == "mount" else f"/results/{job_id}"
    os.makedirs(results_dir, exist_ok=True)

    # Phase 1: Download inputs and rewrite arg values to local paths
    annotation_cache = {}

    # Normalize inputs
    if isinstance(inputs, list):
        input_keys = {k: {} for k in inputs}
    elif isinstance(inputs, dict):
        input_keys = inputs
    else:
        input_keys = {}

    for arg_name, input_cfg in input_keys.items():
        remote_path = args_dict.get(arg_name)
        if not remote_path or not _is_remote(remote_path):
            continue
        if _is_ngc(remote_path) or _is_hf(remote_path):
            # NGC/HF URIs have dedicated downloaders; the generic fsspec
            # path below raises "protocol not known" on them. Args mode has
            # no spec walk, so stage here and rewrite the arg in place.
            args_dict[arg_name] = _stage_ptm_uri(
                remote_path, results_dir, arg_name
            )
            continue

        # Auto-detect file vs folder from the actual remote path
        try:
            fs, _ = fsspec.core.url_to_fs(remote_path)
            remote_key = remote_path.split("://", 1)[1]
            input_type = "folder" if fs.isdir(remote_key) else "file"
        except Exception:
            # Fallback: infer from path pattern
            input_type = "folder" if remote_path.endswith("/") else "file"

        # Lustre mode
        if mount_path:
            lustre_path = _s3_to_lustre(remote_path, mount_path)
            if input_type == "file":
                if os.path.exists(lustre_path):
                    logger.info("Lustre hit: %s", lustre_path)
                    args_dict[arg_name] = lustre_path
                    if lustre_path.endswith(".json"):
                        with open(lustre_path) as f:
                            annotation_cache[arg_name] = json.load(f)
                    continue
                logger.info("Lustre miss, downloading: %s → %s", remote_path, lustre_path)
                download_file(remote_path, lustre_path)
                args_dict[arg_name] = lustre_path
                if lustre_path.endswith(".json"):
                    with open(lustre_path) as f:
                        annotation_cache[arg_name] = json.load(f)
                continue

            elif input_type == "folder":
                ann_ref, sel_fmt, sel_keys = _parse_selective_config(input_cfg)
                video_key = sel_keys[0] if sel_keys else "video"
                if ann_ref and ann_ref in annotation_cache:
                    paths = {e.get(video_key, "") for e in annotation_cache[ann_ref] if e.get(video_key)}
                    all_present = all(os.path.exists(os.path.join(lustre_path, p)) for p in paths)
                    if all_present:
                        logger.info("Lustre hit (selective, %d files): %s", len(paths), lustre_path)
                        args_dict[arg_name] = lustre_path
                        continue
                    logger.info("Lustre miss (selective), downloading %d files to %s", len(paths), lustre_path)
                    download_selective(remote_path, lustre_path,
                                       annotation_cache[ann_ref], video_key)
                else:
                    if os.path.exists(lustre_path) and os.listdir(lustre_path):
                        logger.info("Lustre hit (folder): %s", lustre_path)
                        args_dict[arg_name] = lustre_path
                        continue
                    logger.info("Lustre miss (folder), downloading: %s → %s", remote_path, lustre_path)
                    download_folder(remote_path, lustre_path)
                args_dict[arg_name] = lustre_path
                continue

        # No mount: download to container-local disk
        local_base = os.path.join(results_dir, "inputs", arg_name.replace("-", "_"))

        if input_type == "file":
            fname = os.path.basename(remote_path.rstrip("/"))
            local_file = os.path.join(local_base, fname)
            download_file(remote_path, local_file)
            args_dict[arg_name] = local_file
            if local_file.endswith(".json"):
                with open(local_file) as f:
                    annotation_cache[arg_name] = json.load(f)

        elif input_type == "folder":
            ann_ref, sel_fmt, sel_keys = _parse_selective_config(input_cfg)
            video_key = sel_keys[0] if sel_keys else "video"
            if ann_ref and ann_ref in annotation_cache:
                download_selective(remote_path, local_base,
                                   annotation_cache[ann_ref], video_key)
            else:
                download_folder(remote_path, local_base)
            args_dict[arg_name] = local_base

    # Normalize outputs
    if isinstance(outputs, list):
        output_keys_args = {k: {} for k in outputs}
    elif isinstance(outputs, dict):
        output_keys_args = outputs
    else:
        output_keys_args = {}

    # Auto-fill relative defaults using the runtime-resolved destination.
    _autofill_output_destinations(None, args_dict, output_keys_args, results_root)

    # Three branches mirror run(): remote URI → upload; absolute → direct; else defensive.
    output_remote_map = {}
    for arg_name, output_cfg in output_keys_args.items():
        remote_path = args_dict.get(arg_name)
        output_type = output_cfg.get("type", "folder") if isinstance(output_cfg, dict) else "folder"
        if remote_path and _is_remote(remote_path):
            if mount_path:
                local_path = _s3_to_lustre(remote_path, mount_path)
            else:
                local_path = os.path.join(results_dir, "outputs", arg_name.replace("-", "_"))
            if output_type == "folder":
                os.makedirs(local_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
            output_remote_map[local_path] = remote_path
            args_dict[arg_name] = local_path
        elif isinstance(remote_path, str) and remote_path.startswith("/"):
            if output_type == "folder":
                os.makedirs(remote_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(remote_path), exist_ok=True)
            args_dict[arg_name] = remote_path
        else:
            raise RuntimeError(
                f"Output arg {arg_name!r} has unexpected non-string value "
                f"{remote_path!r}. Auto-fill should have produced an absolute "
                f"path."
            )

    # Phase 2: Build and run command with CLI args
    # Build CLI args: --key 'value' for regular args, --key for flags
    cli_parts = []
    for k, v in args_dict.items():
        cli_parts.append(f"--{k} '{v}'")
    for flag in (flags or []):
        cli_parts.append(f"--{flag}")
    cli_args = " ".join(cli_parts)
    full_command = f"{command} {cli_args}"
    logger.info("Running: %s", full_command)
    result = subprocess.run(full_command, shell=True)
    if result.returncode != 0:
        logger.error("Command failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    # Phase 3: Upload outputs
    uploaded = 0
    for local_path, remote_path in output_remote_map.items():
        if remote_path and os.path.isdir(local_path):
            upload_results(local_path, remote_path, upload_excludes)
            uploaded += 1
        elif remote_path and os.path.isfile(local_path):
            with open(local_path, "rb") as src, fsspec.open(remote_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("Uploaded: %s → %s", local_path, remote_path)
            uploaded += 1
    _log_outputs_summary(output_keys_args, uploaded, results_root, results_kind)

    logger.info("Job complete.")


def run_passthrough(inputs: dict | None, command: str, outputs: list | None,
                    job_id: str, upload_excludes: list[str] | None = None,
                    mount_path: str | None = None):
    """Passthrough mode: download inputs by URI scheme, run command verbatim,
    upload outputs.

    No spec rewriting, no config file writing. ``inputs`` is path-keyed:
    ``{container_path: uri}`` where ``uri`` is one of ``s3://``,
    ``hf_model://``, ``ngc://``. Each is dispatched via the existing
    ``download_remote()`` function.

    ``outputs`` is a list of container paths. Each is uploaded recursively
    to ``s3://$S3_BUCKET_NAME/results/<job_id>/<basename>/`` after the
    command completes. ``upload_excludes`` is a list of glob patterns
    relative to each output dir.
    """
    inputs = inputs or {}
    outputs = outputs or []
    upload_excludes = upload_excludes or []

    results_root, results_kind = _resolve_results_root(job_id)
    results_dir = results_root if results_kind == "mount" else f"/results/{job_id}"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Download inputs by URI scheme
    for container_path, uri in inputs.items():
        if not uri:
            continue
        os.makedirs(os.path.dirname(container_path) or ".", exist_ok=True)
        if mount_path:
            local = _s3_to_lustre(uri, mount_path) if uri.startswith("s3://") else container_path
            if os.path.exists(local):
                logger.info("Already cached on mount: %s", local)
                # Symlink mount path into the requested container path so
                # the user's command sees the data where it expects it.
                if local != container_path and not os.path.exists(container_path):
                    os.symlink(local, container_path)
                continue
        logger.info("Downloading %s → %s", uri, container_path)
        download_remote(uri, container_path)
        if os.path.isfile(container_path) and _is_archive(container_path):
            extract_archive(container_path, os.path.dirname(container_path))
        elif os.path.isdir(container_path):
            extract_archives_in_dir(container_path)

    # 2. Run the command verbatim
    logger.info("Running command: %s", command)
    rc = subprocess.call(command, shell=True)
    if rc != 0:
        logger.error("Command failed with exit code %d", rc)
        sys.exit(rc)

    # 3. Upload outputs (passthrough only uploads when results_root resolves
    # to S3 — mount/ephemeral paths are already where the user's command
    # wrote them, so there's nothing to copy).
    uploaded = 0
    if results_kind == "s3":
        for path in outputs:
            if not os.path.exists(path):
                logger.info("Output path does not exist, skipping: %s", path)
                continue
            if os.path.isdir(path):
                remote_dir = f"{results_root}/{os.path.basename(path.rstrip('/'))}"
                upload_results(path, remote_dir, upload_excludes)
                uploaded += 1
            else:
                remote_file = f"{results_root}/{os.path.basename(path)}"
                with open(path, "rb") as src, fsspec.open(remote_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info("Uploaded: %s → %s", path, remote_file)
                uploaded += 1
    _log_outputs_summary(outputs, uploaded, results_root, results_kind)

    logger.info("Job complete.")


# ---------------------------------------------------------------------------
# Public entrypoint builder
# ---------------------------------------------------------------------------

_DEFAULT_INSTALL_DEPS = shlex.join((
    "fsspec>=2024.6.0",
    "s3fs>=2024.6.0",
    "toml",
    "pandas",
    "pyyaml",
    "huggingface_hub",
))


def _resolve_results_root(job_id: str) -> tuple[str, str]:
    """Compute the upload destination prefix from container env vars.

    Returns ``(results_root, kind)`` where ``kind`` is one of:
      - ``"mount"``     — ``TAO_RESULTS_ROOT`` is set (SLURM Lustre, K8s
                          PVC, Docker bind). No upload needed.
      - ``"s3"``        — Falls back to ``s3://$S3_BUCKET_NAME/results/...``.
                          Upload at end of run.
      - ``"ephemeral"`` — Last resort: ``/results/<job_id>``, container-local.
                          A loud end-of-run warning fires; outputs vanish on
                          container exit.

    Inputs and outputs aren't symmetric: input URIs come from the agent (data
    they own), output destinations come from the platform (policy). Resolving
    here, at runtime, means the agent doesn't thread ``job_id`` /
    ``results_root`` through ``build_entrypoint`` — the SDK mints them and
    injects via env, so ``Job.id`` matches the in-container paths.
    """
    prefix = os.environ.get("TAO_RESULTS_ROOT", "").rstrip("/")
    if prefix:
        return f"{prefix}/{job_id}", "mount"
    bucket = os.environ.get("S3_BUCKET_NAME", "").strip()
    if bucket:
        return f"s3://{bucket}/results/{job_id}", "s3"
    return f"/results/{job_id}", "ephemeral"


def _log_outputs_summary(
    output_keys, uploaded: int, results_root: str, kind: str,
) -> None:
    """End-of-run summary so silent zero-upload is impossible to miss.

    The ephemeral case is the loudest — that's the misconfiguration class
    that previously masqueraded as success.
    """
    if not output_keys:
        return
    if kind == "s3":
        logger.info("Uploaded %d/%d declared output paths to %s",
                    uploaded, len(output_keys), results_root)
    elif kind == "mount":
        logger.info("Outputs persisted on mount at %s. No upload needed.",
                    results_root)
    else:  # ephemeral
        logger.warning(
            "No persistence configured (TAO_RESULTS_ROOT and S3_BUCKET_NAME "
            "both unset). Outputs at %s will NOT survive container exit. "
            "Set S3_BUCKET_NAME or attach a /results mount on the next run.",
            results_root,
        )


def _autofill_output_destinations(
    specs: dict | None,
    args_dict: dict | None,
    outputs: dict | list | None,
    results_root: str | None,
) -> None:
    """For each declared output whose value is not already a remote URI or
    absolute local path, rewrite it to ``{results_root}/<safe_key>``.

    No-op when ``results_root`` is ``None``; the caller (``run``/``run_args``)
    raises shortly after when it walks the outputs.
    """
    if not outputs or not results_root:
        return

    output_keys = list(outputs.keys()) if isinstance(outputs, dict) else list(outputs)
    target = specs if specs is not None else args_dict
    if target is None:
        return

    root = results_root.rstrip("/")
    for key in output_keys:
        if specs is not None:
            current = _get_nested(target, key)
        else:
            current = target.get(key)
        already_real = isinstance(current, str) and (_is_remote(current) or current.startswith("/"))
        if already_real:
            continue
        safe_key = key.replace(".", "_").replace("-", "_")
        new_value = f"{root}/{safe_key}"
        if specs is not None:
            _set_nested(target, key, new_value)
        else:
            target[key] = new_value


def build_entrypoint(
    *,
    command: str,
    inputs: dict | None = None,
    outputs: dict | list | None = None,
    specs: dict | None = None,
    args: dict | None = None,
    config_format: str = "toml",
    upload_excludes: list[str] | None = None,
    mount_path: str | None = None,
    flags: list[str] | None = None,
) -> dict:
    """Build a self-contained shell command that inlines this file (the
    ``script_runner`` runtime) as a base64-encoded heredoc into the container,
    then invokes it in the right mode.

    Mode is **inferred** from what the agent passes:

      - ``specs=...`` → ``config`` mode (spec file written, S3/HF/NGC URIs
        downloaded, ``command`` runs with ``{config_path}`` substituted).
      - ``args=...``  → ``args`` mode (CLI args dict substituted).
      - neither       → ``passthrough`` mode (``inputs`` is path-keyed
        ``{container_path: uri}``; ``outputs`` is a list of paths).

    ``job_id`` and the output destination are resolved at runtime in the
    container from env vars injected by the platform SDK
    (``TAO_JOB_ID``, ``TAO_RESULTS_ROOT``, ``S3_BUCKET_NAME``). Agents don't
    thread them through here.

    Returns ``{"command": <full shell string>, "args_template": ""}``.
    Container deps (``fsspec``, ``s3fs``, ``toml``, ``pandas``, ``pyyaml``,
    ``huggingface_hub``) are pip-installed at the top of the command.
    ``tao_sdk`` itself is NOT installed — this module is inlined.
    """
    # Keep this import local: the file is also executed inside containers where
    # the SDK package itself is intentionally not installed.
    from tao_sdk.spec_serialization import normalize_data, normalize_specs

    if specs is not None and args is not None:
        raise ValueError("Pass either `specs` or `args`, not both.")

    normalized_outputs = normalize_data(
        outputs if outputs is not None else ([] if specs is None and args is None else {}),
        root="outputs",
    )

    if specs is not None:
        if not isinstance(config_format, str):
            raise TypeError("config_format must be one of: json, yaml, toml")
        config_format = config_format.lower()
        if config_format not in {"json", "yaml", "toml"}:
            raise ValueError("config_format must be one of: json, yaml, toml")
        mode = "config"
        normalized_payload = normalize_specs(specs)
        if config_format == "toml":
            # The runtime fills declared output paths before writing the TOML
            # file. Probe a copy with only those null placeholders replaced so
            # unrelated nulls still fail at entrypoint construction time.
            toml_probe = json.loads(json.dumps(normalized_payload, allow_nan=False))
            output_keys = (
                list(normalized_outputs)
                if isinstance(normalized_outputs, (dict, list))
                else []
            )
            if not all(isinstance(key, str) for key in output_keys):
                raise TypeError("outputs must declare string spec paths")
            _autofill_output_destinations(
                toml_probe,
                None,
                normalized_outputs,
                "/tmp/tao-results",
            )
            normalize_specs(toml_probe, allow_none=False)
    elif args is not None:
        mode = "args"
        normalized_payload = normalize_data(args, root="args")
    else:
        mode = "passthrough"
        normalized_payload = {}
    payload = json.dumps(normalized_payload, allow_nan=False)

    runner_source = Path(__file__).read_text(encoding="utf-8")
    runner_b64 = base64.b64encode(runner_source.encode("utf-8")).decode("ascii")

    inputs_json = json.dumps(
        normalize_data(
            {} if inputs is None else inputs,
            root="inputs",
        ),
        allow_nan=False,
    )
    outputs_json = json.dumps(normalized_outputs, allow_nan=False)
    excludes_json = json.dumps(
        normalize_data(
            [] if upload_excludes is None else upload_excludes,
            root="upload_excludes",
        ),
        allow_nan=False,
    )
    flags_json = json.dumps(
        normalize_data([] if flags is None else flags, root="flags"),
        allow_nan=False,
    )

    cli = [
        "python /tmp/_tao_runner.py",
        f"--mode {mode}",
        f"--specs {shlex.quote(payload)}",
        f"--command {shlex.quote(command)}",
        f"--inputs {shlex.quote(inputs_json)}",
        f"--outputs {shlex.quote(outputs_json)}",
        f"--upload-excludes {shlex.quote(excludes_json)}",
    ]
    if mode == "config":
        cli.append(f"--config-format {shlex.quote(config_format)}")
    if mount_path:
        cli.append(f"--mount-path {shlex.quote(mount_path)}")
    if flags:
        cli.append(f"--flags {shlex.quote(flags_json)}")

    # Base64 only contains [A-Za-z0-9+/=] which are all shell-safe, so the
    # literal single-quote wrap is sufficient and clearer than shlex.quote.
    parts = [
        f"pip install {_DEFAULT_INSTALL_DEPS} >/dev/null 2>&1 || true",
        f"echo '{runner_b64}' | base64 -d > /tmp/_tao_runner.py",
        " ".join(cli),
    ]
    return {"command": "\n".join(parts), "args_template": ""}


def main():
    parser = argparse.ArgumentParser(description="TAONext Script Runner")
    parser.add_argument("--mode", default="config",
                        choices=["config", "args", "passthrough"],
                        help="config: write spec file; args: pass CLI args; "
                             "passthrough: download path-keyed inputs, run "
                             "command, upload paths in outputs")
    parser.add_argument("--specs", required=True, help="JSON specs/args string (use '{}' in passthrough)")
    parser.add_argument("--command", required=True, help="Command template")
    parser.add_argument("--inputs", required=True, help="JSON inputs declaration")
    parser.add_argument("--outputs", required=True, help="JSON outputs declaration")
    parser.add_argument("--config-format", default="toml")
    parser.add_argument("--upload-excludes", default="[]", help="JSON list of exclude patterns")
    parser.add_argument("--mount-path", default=None, help="Lustre/NFS mount path")
    parser.add_argument("--flags", default="[]", help="JSON list of flag names (store_true args)")
    args = parser.parse_args()

    # job_id is injected by the platform SDK via env var (single source of truth
    # — matches Job.id returned to the agent). Fall back to a UUID for the rare
    # case where the runner is invoked without an SDK-managed env (e.g. manual
    # smoke runs in a shell).
    job_id = os.environ.get("TAO_JOB_ID") or str(uuid.uuid4())

    if args.mode == "args":
        run_args(
            args_dict=json.loads(args.specs),
            job_id=job_id,
            command=args.command,
            inputs=json.loads(args.inputs),
            outputs=json.loads(args.outputs),
            upload_excludes=json.loads(args.upload_excludes),
            mount_path=args.mount_path,
            flags=json.loads(args.flags),
        )
    elif args.mode == "passthrough":
        run_passthrough(
            inputs=json.loads(args.inputs),
            command=args.command,
            outputs=json.loads(args.outputs),
            job_id=job_id,
            upload_excludes=json.loads(args.upload_excludes),
            mount_path=args.mount_path,
        )
    else:
        run(
            specs=json.loads(args.specs),
            job_id=job_id,
            command=args.command,
            inputs=json.loads(args.inputs),
            outputs=json.loads(args.outputs),
            config_format=args.config_format,
            upload_excludes=json.loads(args.upload_excludes),
            mount_path=args.mount_path,
        )


if __name__ == "__main__":
    main()
