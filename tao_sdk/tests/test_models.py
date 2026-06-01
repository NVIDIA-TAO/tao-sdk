# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_sdk.models."""

import pytest
from tao_sdk.models import Checkpoint, CredentialError, Job, JobStatus


class TestJob:

    def test_create_with_required_fields(self):
        job = Job(id="j-1", network_arch="nvdinov2", action="train", workspace_id="ws-1")
        assert job.id == "j-1"
        assert job.status == "Pending"
        assert job.backend_type == ""
        assert job.backend_job_id == ""
        assert job.created_at  # auto-populated

    def test_to_dict_roundtrip(self):
        job = Job(
            id="j-2", network_arch="nvdinov2", action="eval",
            workspace_id="ws-1", backend_type="brev",
            backend_job_id="tao-job-j-2", status="Running",
        )
        d = job.to_dict()
        assert d["job_id"] == "j-2"
        assert d["backend_type"] == "brev"
        assert d["status"] == "Running"
        assert "network_arch" in d

    def test_status_values(self):
        for status in ("Pending", "Running", "Complete", "Error", "Canceled"):
            job = Job(id="j", network_arch="n", action="a", workspace_id="w", status=status)
            assert job.status == status


class TestJobStatus:

    def test_create(self):
        js = JobStatus(job_id="j-1", status="Running", message="all good")
        assert js.job_id == "j-1"
        assert js.status == "Running"
        assert js.updated_at  # auto-populated


class TestCheckpoint:

    def test_create(self):
        ckpt = Checkpoint(path="s3://bucket/epoch_10.pth", epoch=10,
                          metric_name="mAP", metric_value=0.85)
        assert ckpt.epoch == 10
        assert ckpt.metric_value == 0.85


class TestCredentialError:

    def test_is_exception(self):
        with pytest.raises(CredentialError, match="BREV_API_TOKEN"):
            raise CredentialError("Missing BREV_API_TOKEN")
