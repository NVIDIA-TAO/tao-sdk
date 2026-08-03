# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint resolution helpers shared by SDK and AutoML flows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHECKPOINT_FILE_EXTENSIONS = (
    ".pth",
    ".tlt",
    ".hdf5",
    ".pt",
    ".ckpt",
    ".engine",
    ".onnx",
    ".trt",
)


@dataclass(frozen=True)
class CheckpointCandidate:
    """A candidate checkpoint path with parsed training-position metadata."""

    path: str
    epoch: int | None = None
    step: int | None = None
    is_dir: bool = False
    is_latest: bool = False
    is_best: bool = False
    mtime: float = 0.0


def _parse_number(patterns: Iterable[str], text: str) -> int | None:
    values: list[int] = []
    for pattern in patterns:
        values.extend(
            int(match.group(1))
            for match in re.finditer(pattern, text, re.IGNORECASE)
        )
    return max(values) if values else None


def checkpoint_epoch(path: str | os.PathLike[str]) -> int | None:
    """Extract an epoch-like number from a checkpoint path."""

    text = str(path).replace("\\", "/")
    base = Path(text).name
    epoch = _parse_number(
        (
            r"(?:^|[/_.-])checkpoint[_-]?epoch[_-]?0*(\d+)(?:\D|$)",
            r"(?:^|[/_.-])model[_-]?epoch[_-]?0*(\d+)(?:\D|$)",
            r"(?:^|[/_.-])model[_-]?best[_-]?0*(\d+)(?:\D|$)",
            r"(?:^|[/_.-])epoch[_-]?0*(\d+)(?:\D|$)",
            r"(?:^|[/_.-])ep[_-]?0*(\d+)(?:\D|$)",
        ),
        text,
    )
    if epoch is not None:
        return epoch
    if re.fullmatch(r"0*\d+(?:\.[^.]+)?", base):
        return int(base.split(".", 1)[0])
    return None


def checkpoint_step(path: str | os.PathLike[str]) -> int | None:
    """Extract a step/iteration-like number from a checkpoint path."""

    text = str(path).replace("\\", "/")
    return _parse_number(
        (
            r"(?:^|[/_.-])step[_-]?0*(\d+)(?:\D|$)",
            r"(?:^|[/_.-])iter[_-]?0*(\d+)(?:\D|$)",
            r"(?:^|[/_.-])iteration[_-]?0*(\d+)(?:\D|$)",
        ),
        text,
    )


def build_checkpoint_candidate(
    path: str | os.PathLike[str],
    *,
    is_dir: bool | None = None,
    mtime: float | None = None,
) -> CheckpointCandidate:
    """Build a parsed checkpoint candidate from a path string."""

    value = str(path)
    local = Path(value)
    if is_dir is None:
        try:
            is_dir = local.is_dir()
        except OSError:
            is_dir = False
    if mtime is None:
        try:
            mtime = local.stat().st_mtime
        except OSError:
            mtime = 0.0
    lower = value.lower()
    return CheckpointCandidate(
        path=value,
        epoch=checkpoint_epoch(value),
        step=checkpoint_step(value),
        is_dir=bool(is_dir),
        is_latest="latest" in lower,
        is_best="best" in lower,
        mtime=float(mtime or 0.0),
    )


def iter_local_checkpoint_candidates(
    experiment_dir: str | os.PathLike[str],
    *,
    include_dirs: bool = True,
) -> list[CheckpointCandidate]:
    """Recursively collect checkpoint candidates from a local experiment dir."""

    root_path = Path(experiment_dir)
    if not root_path.exists():
        return []

    candidates: list[CheckpointCandidate] = []

    def usable(path: Path) -> bool:
        norm = path.as_posix()
        return "/inputs/" not in norm and "/ptm/" not in norm

    for root, dirs, files in os.walk(root_path):
        current = Path(root)
        dirs[:] = [d for d in dirs if d not in {"inputs", "ptm", "__pycache__"}]
        if not usable(current):
            continue

        if include_dirs and current != root_path:
            name = current.name.lower()
            if name.startswith(("epoch_", "epoch-", "step_", "step-")):
                try:
                    if any(current.iterdir()):
                        candidates.append(
                            build_checkpoint_candidate(current, is_dir=True)
                        )
                except OSError:
                    pass

        for filename in files:
            file_path = current / filename
            if not usable(file_path):
                continue
            if filename.lower().endswith(CHECKPOINT_FILE_EXTENSIONS):
                candidates.append(build_checkpoint_candidate(file_path, is_dir=False))

    return candidates


def _epoch_match_rank(candidate: CheckpointCandidate, epoch: int | None) -> int:
    if epoch is None:
        return 0
    if candidate.epoch == epoch:
        return 3
    # Many TAO trainers write zero-indexed epoch filenames after an epoch
    # completes. Treat epoch_000 as the checkpoint for completed epoch 1.
    if epoch > 0 and candidate.epoch == epoch - 1:
        return 2
    return -1


def _step_match_rank(candidate: CheckpointCandidate, step: int | None) -> int:
    if step is None:
        return 0
    return 2 if candidate.step == step else -1


def select_checkpoint_path(
    candidates: Iterable[CheckpointCandidate | str],
    *,
    model_name: str | None = None,
    epoch: int | None = None,
    step: int | None = None,
    action: str = "resume",
    prefer_directory: bool = False,
    allow_latest: bool = False,
) -> str | None:
    """Select the checkpoint path that matches the requested action/budget.

    Resume callers should pass the stopped epoch/step. In that mode, generic
    ``latest`` files are not selected unless they also contain matching
    epoch/step metadata or ``allow_latest`` is explicit.
    """

    parsed = [
        item if isinstance(item, CheckpointCandidate) else build_checkpoint_candidate(item)
        for item in candidates
    ]
    if not parsed:
        return None

    requires_position = epoch is not None or step is not None
    scored: list[tuple[tuple[int, int, int, int, int, int, int, float], str]] = []

    for candidate in parsed:
        epoch_rank = _epoch_match_rank(candidate, epoch)
        step_rank = _step_match_rank(candidate, step)
        if epoch_rank < 0 or step_rank < 0:
            continue
        if requires_position and candidate.is_latest and epoch_rank == 0 and step_rank == 0:
            if not allow_latest:
                continue

        norm = candidate.path.replace("\\", "/")
        in_checkpoints = "/checkpoints/" in norm or norm.endswith("/checkpoints")
        in_safetensors = "/safetensors/" in norm or norm.endswith("/safetensors")

        path_kind = 0
        if candidate.is_dir:
            path_kind = 4 if prefer_directory else 2
        elif not prefer_directory:
            path_kind = 4

        action_rank = 0
        lower_norm = norm.lower()
        if action in {"resume", "train"}:
            # NVDINOv2 writes teacher/student weights next to the Lightning
            # trainer checkpoint. Resume needs the full trainer checkpoint.
            if model_name and model_name.replace("-", "_") == "nvdinov2":
                if "model_epoch" in lower_norm:
                    action_rank += 3
                elif "teacher_epoch" in lower_norm or "student_epoch" in lower_norm:
                    action_rank -= 3
        if action in {"eval", "evaluate", "inference", "infer", "export", "best"}:
            action_rank += 4 if candidate.is_best else 0
            if model_name and model_name.replace("-", "_") == "nvdinov2":
                action_rank += 3 if "student_epoch" in lower_norm else 0

        location_rank = 2 if in_checkpoints else 1 if in_safetensors else 0
        named_rank = 0 if candidate.is_latest else 1
        budget_rank = candidate.step if step is not None and candidate.step is not None else -1
        if budget_rank < 0:
            budget_rank = candidate.epoch if candidate.epoch is not None else -1

        scored.append((
            (
                epoch_rank,
                step_rank,
                action_rank,
                path_kind,
                location_rank,
                named_rank,
                budget_rank,
                candidate.mtime,
            ),
            candidate.path,
        ))

    if not scored:
        if requires_position or not allow_latest:
            return None
        latest = [item for item in parsed if item.is_latest]
        if latest:
            return max(latest, key=lambda item: item.mtime).path
        return max(parsed, key=lambda item: item.mtime).path

    return max(scored, key=lambda item: item[0])[1]


def get_checkpoint_path(
    *,
    model_name: str,
    experiment_dir: str | os.PathLike[str],
    epoch: int | None = None,
    step: int | None = None,
    action: str = "resume",
    prefer_directory: bool = False,
    allow_latest: bool = False,
) -> str | None:
    """Resolve a model/action checkpoint path from a local experiment folder."""

    candidates = iter_local_checkpoint_candidates(
        experiment_dir,
        include_dirs=prefer_directory or action == "resume",
    )
    return select_checkpoint_path(
        candidates,
        model_name=model_name,
        epoch=epoch,
        step=step,
        action=action,
        prefer_directory=prefer_directory,
        allow_latest=allow_latest,
    )
