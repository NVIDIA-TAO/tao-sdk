# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_sdk.job_store."""

import json
import pytest

from tao_sdk.job_store import JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def store_with_train_job(store):
    """Store with a completed train job and checkpoint."""
    store.create_job(
        "train-1", "cosmos-rl", "train", "ws-1",
        results_dir="s3://bucket/results/train-1",
        local_results_dir="/results/train-1",
        base_checkpoint="nvidia/Cosmos-Reason2-8B",
    )
    store.update_job_status("train-1", "Complete")
    store.add_checkpoint("train-1", "/results/train-1/safetensors/epoch_1", epoch=1)
    return store


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------

class TestJobCRUD:

    def test_create_and_get(self, store):
        store.create_job("j-1", "clip", "train", "ws-1", backend_job_id="tao-job-j-1")
        job = store.get_job("j-1")
        assert job is not None
        assert job["job_id"] == "j-1"
        assert job["network_arch"] == "clip"
        assert job["action"] == "train"
        assert job["status"] == "Pending"

    def test_get_nonexistent(self, store):
        assert store.get_job("nonexistent") is None

    def test_update_status(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.update_job_status("j-1", "Running")
        assert store.get_job("j-1")["status"] == "Running"

    def test_update_status_with_epoch(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.update_job_status("j-1", "Running", last_epoch=5)
        job = store.get_job("j-1")
        assert job["status"] == "Running"
        assert job["last_epoch"] == 5

    def test_list_jobs(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.create_job("j-2", "cosmos-rl", "train", "ws-1")
        store.create_job("j-3", "clip", "eval", "ws-1")
        assert len(store.list_jobs()) == 3

    def test_list_jobs_filter_status(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.create_job("j-2", "clip", "train", "ws-1")
        store.update_job_status("j-2", "Complete")
        assert len(store.list_jobs(status="Pending")) == 1
        assert len(store.list_jobs(status="Complete")) == 1

    def test_list_jobs_filter_network(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.create_job("j-2", "cosmos-rl", "train", "ws-1")
        results = store.list_jobs(network_arch="clip")
        assert len(results) == 1
        assert results[0]["job_id"] == "j-1"

    def test_create_job_stores_specs(self, store):
        specs = {"train": {"num_epochs": 10}, "policy": {"model_max_length": 40960}}
        store.create_job("j-1", "clip", "train", "ws-1", specs=specs)
        job = store.get_job("j-1")
        stored_specs = json.loads(job["specs"])
        assert stored_specs["train"]["num_epochs"] == 10

    def test_create_job_insert_or_replace(self, store):
        store.create_job("j-1", "clip", "train", "ws-1", status="Pending")
        store.create_job("j-1", "clip", "train", "ws-1",
                         backend_job_id="tao-job-j-1", status="Pending")
        job = store.get_job("j-1")
        assert job["backend_job_id"] == "tao-job-j-1"


# ---------------------------------------------------------------------------
# Checkpoint tracking
# ---------------------------------------------------------------------------

class TestCheckpoints:

    def test_add_and_get(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.add_checkpoint("j-1", "/results/j-1/epoch_5", epoch=5)
        store.add_checkpoint("j-1", "/results/j-1/epoch_10", epoch=10)
        cps = store.get_checkpoints("j-1")
        assert len(cps) == 2
        assert cps[0]["epoch"] == 10  # newest first
        assert cps[1]["epoch"] == 5

    def test_latest_checkpoint(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.add_checkpoint("j-1", "/results/j-1/epoch_5", epoch=5)
        store.add_checkpoint("j-1", "/results/j-1/epoch_10", epoch=10)
        latest = store.get_latest_checkpoint("j-1")
        assert latest["epoch"] == 10
        assert latest["path"] == "/results/j-1/epoch_10"

    def test_latest_checkpoint_empty(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        assert store.get_latest_checkpoint("j-1") is None

    def test_checkpoint_format(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.add_checkpoint("j-1", "/path/model.pth", epoch=1, format="pth")
        cp = store.get_latest_checkpoint("j-1")
        assert cp["format"] == "pth"

    def test_checkpoint_path_prefers_requested_epoch_over_latest(self, store):
        store.create_job("j-1", "classification-pyt", "train", "ws-1")
        store.add_checkpoint("j-1", "/results/j-1/train/classifier_model_latest.pth", epoch=1)
        store.add_checkpoint("j-1", "/results/j-1/train/model_epoch_001.pth", epoch=1)
        assert (
            store.get_checkpoint_path("j-1", model_name="classification-pyt", epoch=1)
            == "/results/j-1/train/model_epoch_001.pth"
        )

    def test_checkpoint_path_matches_zero_indexed_epoch_file(self, store):
        store.create_job("j-1", "depth-net-stereo", "train", "ws-1")
        store.add_checkpoint("j-1", "/results/j-1/train/dn_model_latest.pth", epoch=1)
        store.add_checkpoint("j-1", "/results/j-1/train/model_epoch_000_step_00014.pth", epoch=1)
        assert (
            store.get_checkpoint_path("j-1", model_name="depth-net-stereo", epoch=1)
            == "/results/j-1/train/model_epoch_000_step_00014.pth"
        )

    def test_checkpoint_path_returns_none_for_requested_epoch_without_match(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        store.add_checkpoint("j-1", "/results/j-1/train/model_latest.pth", epoch=3)
        assert store.get_checkpoint_path("j-1", model_name="clip", epoch=1) is None


# ---------------------------------------------------------------------------
# Job chain queries
# ---------------------------------------------------------------------------

class TestJobChain:

    def test_get_parent_job(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("eval-1", "cosmos-rl", "eval", "ws-1", parent_job_id="train-1")
        parent = store.get_parent_job("eval-1")
        assert parent is not None
        assert parent["job_id"] == "train-1"

    def test_get_parent_job_none(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        assert store.get_parent_job("j-1") is None

    def test_get_parent_model_path(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("eval-1", "cosmos-rl", "eval", "ws-1", parent_job_id="train-1")
        path = store.get_parent_model_path("eval-1")
        assert path == "/results/train-1/safetensors/epoch_1"

    def test_get_parent_model_path_no_parent(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        assert store.get_parent_model_path("j-1") == ""

    def test_get_parent_results_dir(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("eval-1", "cosmos-rl", "eval", "ws-1", parent_job_id="train-1")
        assert store.get_parent_results_dir("eval-1") == "s3://bucket/results/train-1"


# ---------------------------------------------------------------------------
# resolve_spec_param (inference function registry)
# ---------------------------------------------------------------------------

class TestResolveSpecParam:

    def test_output_dir(self, store):
        store.create_job("j-1", "clip", "train", "ws-1",
                         results_dir="s3://bucket/results/j-1")
        assert store.resolve_spec_param("j-1", "output_dir") == "s3://bucket/results/j-1"

    def test_automl_output_dir(self, store):
        store.create_job("j-1", "clip", "train", "ws-1",
                         results_dir="s3://bucket/results/j-1")
        assert store.resolve_spec_param("j-1", "automl_output_dir") == "s3://bucket/results/j-1"

    def test_resume_model(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("j-2", "cosmos-rl", "train", "ws-1", parent_job_id="train-1")
        assert store.resolve_spec_param("j-2", "resume_model") == "/results/train-1/safetensors/epoch_1"

    def test_resume_model_no_parent(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        assert store.resolve_spec_param("j-1", "resume_model") is None

    def test_parent_model(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("eval-1", "cosmos-rl", "eval", "ws-1", parent_job_id="train-1")
        assert store.resolve_spec_param("eval-1", "parent_model") == "/results/train-1/safetensors/epoch_1"

    def test_resume_model_bool_true(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("j-2", "cosmos-rl", "train", "ws-1", parent_job_id="train-1")
        assert store.resolve_spec_param("j-2", "resume_model_bool") == "true"

    def test_resume_model_bool_false(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        assert store.resolve_spec_param("j-1", "resume_model_bool") == "false"

    def test_parent_model_folder(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("eval-1", "cosmos-rl", "eval", "ws-1", parent_job_id="train-1")
        assert store.resolve_spec_param("eval-1", "parent_model_folder") == "s3://bucket/results/train-1"

    def test_ptm_if_no_resume_model_no_parent(self, store):
        store.create_job("j-1", "cosmos-rl", "train", "ws-1",
                         base_checkpoint="nvidia/Cosmos-Reason2-8B")
        assert store.resolve_spec_param("j-1", "ptm_if_no_resume_model") == "nvidia/Cosmos-Reason2-8B"

    def test_ptm_if_no_resume_model_has_parent(self, store_with_train_job):
        store = store_with_train_job
        store.create_job("j-2", "cosmos-rl", "train", "ws-1",
                         parent_job_id="train-1", base_checkpoint="nvidia/Cosmos-Reason2-8B")
        assert store.resolve_spec_param("j-2", "ptm_if_no_resume_model") is None

    def test_unsupported_function(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        assert store.resolve_spec_param("j-1", "nonexistent_function") is None

    def test_output_dir_nonexistent_job(self, store):
        assert store.resolve_spec_param("nonexistent", "output_dir") is None


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

class TestMigration:

    def test_import_from_state_file(self, store, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "active_jobs": [
                {
                    "job_id": "j-1",
                    "network_arch": "clip",
                    "action": "train",
                    "workspace_id": "ws-1",
                    "backend_type": "brev",
                    "status": "Running",
                    "backend_job_id": "tao-job-j-1",
                    "created_at": "2026-03-25T00:00:00+00:00",
                }
            ]
        }))
        count = store.import_from_state_file(str(state_file))
        assert count == 1
        job = store.get_job("j-1")
        assert job["status"] == "Running"
        assert job["network_arch"] == "clip"

    def test_import_skips_existing(self, store, tmp_path):
        store.create_job("j-1", "clip", "train", "ws-1", status="Complete")
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "active_jobs": [{"job_id": "j-1", "network_arch": "clip",
                            "action": "train", "workspace_id": "ws-1",
                            "status": "Running"}]
        }))
        count = store.import_from_state_file(str(state_file))
        assert count == 0
        assert store.get_job("j-1")["status"] == "Complete"  # not overwritten

    def test_import_missing_file(self, store):
        assert store.import_from_state_file("/nonexistent/state.json") == 0

    def test_import_invalid_json(self, store, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert store.import_from_state_file(str(bad)) == 0


# ---------------------------------------------------------------------------
# Plan step tracking
# ---------------------------------------------------------------------------

class TestPlanStepTracking:

    def test_create_job_with_plan_id_and_step_id(self, store):
        store.create_job(
            "j-1", "cosmos-rl", "evaluate", "ws-1",
            results_dir="s3://bucket/results/j-1",
            plan_id="plan-abc", step_id=1,
        )
        job = store.get_job("j-1")
        assert job["plan_id"] == "plan-abc"
        assert job["step_id"] == 1

    def test_get_job_by_plan_step(self, store):
        store.create_job(
            "j-1", "cosmos-rl", "evaluate", "ws-1",
            results_dir="s3://bucket/results/j-1",
            plan_id="plan-abc", step_id=1,
        )
        store.create_job(
            "j-2", "cosmos-rl", "train", "ws-1",
            results_dir="s3://bucket/results/j-2",
            plan_id="plan-abc", step_id=2,
        )
        job = store.get_job_by_plan_step("plan-abc", 1)
        assert job is not None
        assert job["job_id"] == "j-1"
        assert job["results_dir"] == "s3://bucket/results/j-1"

    def test_get_job_by_plan_step_returns_latest(self, store):
        """If a step is re-run, return the most recent job."""
        store.create_job(
            "j-1", "cosmos-rl", "evaluate", "ws-1",
            results_dir="s3://bucket/results/j-1",
            plan_id="plan-abc", step_id=1,
            created_at="2026-01-01T00:00:00+00:00",
        )
        store.create_job(
            "j-1-retry", "cosmos-rl", "evaluate", "ws-1",
            results_dir="s3://bucket/results/j-1-retry",
            plan_id="plan-abc", step_id=1,
            created_at="2026-01-02T00:00:00+00:00",
        )
        job = store.get_job_by_plan_step("plan-abc", 1)
        assert job["job_id"] == "j-1-retry"

    def test_get_job_by_plan_step_not_found(self, store):
        assert store.get_job_by_plan_step("nonexistent", 1) is None

    def test_plan_id_defaults_empty(self, store):
        store.create_job("j-1", "clip", "train", "ws-1")
        job = store.get_job("j-1")
        assert job["plan_id"] == ""
        assert job["step_id"] == 0


class TestMigrateColumns:

    def test_migrate_adds_missing_columns(self, tmp_path):
        """Migration adds plan_id/step_id to an older DB without them."""
        import sqlite3
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        # Create schema WITHOUT plan_id/step_id (simulates old DB)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                network_arch TEXT NOT NULL,
                action TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                backend_type TEXT DEFAULT 'brev',
                backend_job_id TEXT DEFAULT '',
                status TEXT DEFAULT 'Pending',
                parent_job_id TEXT,
                image TEXT DEFAULT '',
                specs TEXT DEFAULT '{}',
                results_dir TEXT DEFAULT '',
                local_results_dir TEXT DEFAULT '',
                base_checkpoint TEXT DEFAULT '',
                last_epoch INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                path TEXT NOT NULL,
                epoch INTEGER DEFAULT 0,
                format TEXT DEFAULT 'safetensors',
                created_at TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            ("old-j", "clip", "train", "ws-1", "brev", "", "Complete",
             None, "", "{}", "s3://b/r/old-j", "", "",
             "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        # Opening with JobStore should migrate
        store = JobStore(db_path=db_path)
        job = store.get_job("old-j")
        assert job["plan_id"] == ""
        assert job["step_id"] == 0
        # New jobs can use the columns
        store.create_job(
            "new-j", "clip", "train", "ws-1",
            plan_id="p-1", step_id=3,
        )
        assert store.get_job("new-j")["plan_id"] == "p-1"
