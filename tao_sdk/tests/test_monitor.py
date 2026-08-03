# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_sdk.monitor."""

import time

import pytest
from unittest.mock import MagicMock

from tao_sdk.job_store import JobStore
from tao_sdk.models import Job
from tao_sdk.monitor import JobMonitor


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "state.json")


@pytest.fixture
def store(tmp_path):
    return JobStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def monitor(state_file, store):
    m = JobMonitor(poll_interval=1, epoch_milestone_interval=10,
                   state_file=state_file, store=store)
    yield m
    m.stop()


@pytest.fixture
def sample_job():
    return Job(id="j-1", network_arch="nvdinov2", action="train", workspace_id="ws-1",
               backend_type="brev", backend_job_id="tao-job-j-1")


class TestJobMonitor:

    def test_watch_and_unwatch(self, monitor, sample_job, mock_handler):
        monitor.watch(sample_job, {}, mock_handler)
        assert "j-1" in monitor._watched
        monitor.unwatch("j-1")
        assert "j-1" not in monitor._watched

    def test_watch_registers_job(self, monitor, sample_job, mock_handler):
        monitor.watch(sample_job, {}, mock_handler)
        assert "j-1" in monitor._watched

    def test_watch_short_circuits_existing_durable_terminal_row(
        self, monitor, sample_job, mock_handler, store
    ):
        store.create_job(
            "j-1", "nvdinov2", "train", "ws-1",
            backend_type="brev", status="Canceled",
        )

        monitor.watch(sample_job, {}, mock_handler)

        assert sample_job.status == "Canceled"
        assert "j-1" not in monitor._watched
        mock_handler.get_tao_job_status.assert_not_called()

    def test_poll_short_circuits_terminal_row_won_after_watch(
        self, monitor, sample_job, mock_handler, store
    ):
        monitor.watch(sample_job, {}, mock_handler)
        entry = monitor._watched["j-1"]
        store.update_job_status("j-1", "Complete")

        monitor._check_one("j-1", entry)

        assert sample_job.status == "Complete"
        assert "j-1" not in monitor._watched
        mock_handler.get_tao_job_status.assert_not_called()

    def test_start_stop(self, monitor):
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        monitor.stop()
        assert not monitor._thread.is_alive()

    def test_status_transition_fires_on_complete(self, monitor, sample_job, mock_handler, state_file):
        on_complete = MagicMock()
        mock_handler.get_tao_job_status.return_value = None
        mock_handler.normalize_status.side_effect = lambda raw: "Complete" if raw is None else raw

        monitor.watch(sample_job, {"on_complete": on_complete}, mock_handler)

        # Manually trigger a poll cycle
        entry = monitor._watched["j-1"]
        monitor._check_one("j-1", entry)

        on_complete.assert_called_once_with(sample_job)
        assert sample_job.status == "Complete"

    def test_status_transition_fires_on_error(self, monitor, sample_job, mock_handler):
        on_error = MagicMock()
        mock_handler.get_tao_job_status.return_value = "Error"
        mock_handler.normalize_status.side_effect = lambda raw: raw

        monitor.watch(sample_job, {"on_error": on_error}, mock_handler)

        entry = monitor._watched["j-1"]
        monitor._check_one("j-1", entry)

        on_error.assert_called_once_with(sample_job)
        assert sample_job.status == "Error"

    def test_terminal_status_unwatches(self, monitor, sample_job, mock_handler):
        mock_handler.get_tao_job_status.return_value = None
        mock_handler.normalize_status.side_effect = lambda raw: "Complete" if raw is None else raw

        monitor.watch(sample_job, {}, mock_handler)
        entry = monitor._watched["j-1"]
        monitor._check_one("j-1", entry)

        assert "j-1" not in monitor._watched

    def test_no_transition_no_hook(self, monitor, sample_job, mock_handler):
        on_complete = MagicMock()
        mock_handler.get_tao_job_status.return_value = "Pending"
        mock_handler.normalize_status.side_effect = lambda raw: raw

        monitor.watch(sample_job, {"on_complete": on_complete}, mock_handler)
        entry = monitor._watched["j-1"]
        monitor._check_one("j-1", entry)

        on_complete.assert_not_called()
        assert sample_job.status == "Pending"

    def test_epoch_milestone_detection(self, monitor, sample_job, mock_handler):
        on_milestone = MagicMock()
        mock_handler.get_tao_job_status.return_value = "Running"
        mock_handler.normalize_status.side_effect = lambda raw: raw
        mock_handler.get_job_logs.return_value = "Epoch 10/50 - loss: 0.3"

        monitor.watch(sample_job, {"on_epoch_milestone": on_milestone}, mock_handler)
        entry = monitor._watched["j-1"]

        # First poll: transition Pending -> Running
        monitor._check_one("j-1", entry)
        # Second poll: stays Running, should check epoch
        monitor._check_one("j-1", entry)

        on_milestone.assert_called_once_with(sample_job, 10, 50)

    def test_epoch_milestone_not_fired_twice(self, monitor, sample_job, mock_handler):
        on_milestone = MagicMock()
        mock_handler.get_tao_job_status.return_value = "Running"
        mock_handler.normalize_status.side_effect = lambda raw: raw
        mock_handler.get_job_logs.return_value = "Epoch 10/50 - loss: 0.3"

        monitor.watch(sample_job, {"on_epoch_milestone": on_milestone}, mock_handler)
        entry = monitor._watched["j-1"]

        # Transition to Running
        monitor._check_one("j-1", entry)
        # First milestone
        monitor._check_one("j-1", entry)
        # Same epoch again — should not fire
        monitor._check_one("j-1", entry)

        assert on_milestone.call_count == 1

    def test_bad_hook_does_not_crash(self, monitor, sample_job, mock_handler):
        def bad_hook(job):
            raise ValueError("hook exploded")

        mock_handler.get_tao_job_status.return_value = None
        mock_handler.normalize_status.side_effect = lambda raw: "Complete" if raw is None else raw

        monitor.watch(sample_job, {"on_complete": bad_hook}, mock_handler)
        entry = monitor._watched["j-1"]

        # Should not raise
        monitor._check_one("j-1", entry)
        assert sample_job.status == "Complete"

    def test_handler_exception_does_not_crash(self, monitor, sample_job, mock_handler):
        mock_handler.get_tao_job_status.side_effect = ConnectionError("network down")

        monitor.watch(sample_job, {}, mock_handler)
        entry = monitor._watched["j-1"]

        # Should not raise
        monitor._check_one("j-1", entry)
        assert sample_job.status == "Pending"  # unchanged

    def test_poll_loop_integration(self, monitor, sample_job, mock_handler):
        """Verify the background thread actually polls."""
        call_count = {"n": 0}

        def counting_status(job_id):
            call_count["n"] += 1
            return "Pending"

        mock_handler.get_tao_job_status.side_effect = counting_status
        mock_handler.normalize_status.side_effect = lambda raw: raw

        monitor.watch(sample_job, {}, mock_handler)
        monitor.start()
        time.sleep(2.5)  # poll_interval=1, should get ~2 polls
        monitor.stop()

        assert call_count["n"] >= 2

    def test_store_updated_on_transition(self, monitor, sample_job, mock_handler, store):
        mock_handler.get_tao_job_status.return_value = "Running"
        mock_handler.normalize_status.side_effect = lambda raw: raw

        store.create_job("j-1", "nvdinov2", "train", "ws-1")

        monitor.watch(sample_job, {}, mock_handler)
        entry = monitor._watched["j-1"]
        monitor._check_one("j-1", entry)

        job = store.get_job("j-1")
        assert job["status"] == "Running"
