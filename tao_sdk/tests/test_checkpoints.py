"""Tests for shared checkpoint selection helpers."""

from tao_sdk.checkpoints import (
    build_checkpoint_candidate,
    checkpoint_epoch,
    select_checkpoint_path,
)


def test_checkpoint_epoch_parses_bounded_best_filename():
    """Lightning's ``{epoch}`` token is zero-indexed in best checkpoints."""

    assert checkpoint_epoch("/results/train/model_best_000.pth") == 0
    assert checkpoint_epoch("/results/train/model-best-012.ckpt") == 12


def test_resume_selects_bounded_best_for_completed_epoch():
    """ASHA epoch 1 must resolve the physical zero-indexed best checkpoint."""

    best = build_checkpoint_candidate(
        "/results/parent/train/model_best_000.pth",
        mtime=1.0,
    )
    latest = build_checkpoint_candidate(
        "/results/parent/train/model_latest.pth",
        mtime=2.0,
    )

    assert select_checkpoint_path(
        [latest, best],
        model_name="visual-changenet",
        epoch=1,
        action="resume",
    ) == best.path
