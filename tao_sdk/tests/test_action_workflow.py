# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for action workflow status persistence."""

from types import SimpleNamespace

from tao_sdk.action_workflow import ActionWorkflow


def test_sync_from_sdk_updates_status_and_clears_terminal_active_job(tmp_path):
    workflow = ActionWorkflow(tmp_path, "dino_eval", "20260427_120000")
    job = SimpleNamespace(
        id="job-1",
        backend_job_id="tao-job-job-1",
        network_arch="dino",
        action="evaluate",
        workspace_id="ws",
        status="Pending",
        created_at="now",
    )
    workflow.write_submission(job=job, specs={"evaluate": {}}, script_runner={})

    status = SimpleNamespace(
        job_id="job-1",
        status="Error",
        message="backend failed",
        updated_at="later",
    )
    sdk = SimpleNamespace(
        get_job_status=lambda job_id: status,
        get_job_logs=lambda job_id, tail=200: "log tail",
        get_failure_analysis=lambda job_id: {"reason": "bad checkpoint"},
    )

    returned = workflow.sync_from_sdk(sdk, "job-1")

    assert returned is status
    assert '"status": "Error"' in workflow.status_path.read_text(encoding="utf-8")
    assert workflow.active_jobs_path.read_text(encoding="utf-8") == "[]\n"
    assert "backend failed" in workflow.events_path.read_text(encoding="utf-8")
    assert workflow.logs_path.read_text(encoding="utf-8") == "log tail"
