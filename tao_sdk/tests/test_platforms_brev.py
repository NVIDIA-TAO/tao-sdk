# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev container-identity regression tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


_CONTAINER_ID = "a" * 64
_REPLACEMENT_ID = "b" * 64


def _restored_handler(cli, *, container_id=_CONTAINER_ID):
    from tao_sdk.platforms.brev.handler import BrevHandler

    handler = BrevHandler(cli=cli)
    runtime = {
        "instance_id": "instance-1",
        "pid": "1234",
        "launch_uncertain": False,
    }
    if container_id:
        runtime["container_id"] = container_id
    assert handler.restore_job_runtime("job-1", runtime)
    return handler


def test_brev_status_refuses_same_name_replacement():
    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=0,
        stdout=f"CONTAINER_ID:{_CONTAINER_ID}\nIDENTITY_MISMATCH\n",
        stderr="",
    )
    handler = _restored_handler(cli)

    assert handler.get_tao_job_status("job-1") == "Pending"

    command = cli.exec.call_args.args[1]
    assert f"expected_id={_CONTAINER_ID}" in command
    assert 'docker inspect -f' in command
    assert '"$expected_id"' in command
    assert '"$actual_id" != "$expected_id"' in command
    assert '"$actual_name" != /tao-job-job-1' in command
    assert '"$job_label" != job-1' in command
    assert '"$backend_label" != brev' in command
    assert handler.get_job_runtime_identity("job-1")["container_id"] == (
        _CONTAINER_ID
    )


def test_brev_cancel_refuses_same_name_replacement():
    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=1,
        stdout="IDENTITY_MISMATCH\n",
        stderr="",
    )
    handler = _restored_handler(cli)

    assert not handler.cancel_job("job-1")

    command = cli.exec.call_args.args[1]
    assert f"expected_id={_CONTAINER_ID}" in command
    assert 'docker stop --time 10 "$expected_id"' in command
    assert 'docker rm -f "$expected_id"' in command
    assert "docker stop --time 10 tao-job-job-1" not in command
    assert handler.get_job_runtime_identity("job-1")["container_id"] == (
        _CONTAINER_ID
    )


def test_brev_status_learns_cidfile_identity_for_durable_persistence():
    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=0,
        stdout=f"CONTAINER_ID:{_CONTAINER_ID}\nEXIT_DONE\n",
        stderr="",
    )
    handler = _restored_handler(cli, container_id="")

    assert handler.get_tao_job_status("job-1") is None
    assert handler.get_job_runtime_identity("job-1") == {
        "instance_id": "instance-1",
        "pid": "1234",
        "launch_uncertain": False,
        "container_id": _CONTAINER_ID,
    }
    assert "/tmp/tao-job-job-1.cid" in cli.exec.call_args.args[1]


def test_brev_create_captures_full_cidfile_identity():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.get_instance.return_value = object()
    cli.exec.return_value = SimpleNamespace(
        returncode=0,
        stdout=f"1234 {_CONTAINER_ID}\n",
        stderr="",
    )
    handler = BrevHandler(cli=cli)

    runtime = handler.create_job(
        image="example.invalid/tao:test",
        network="",
        action="",
        cloud_metadata={},
        specs={},
        job_id="job-1",
        docker_env_vars={},
        resource_shape=None,
        dedicated_node_group=None,
        num_nodes=1,
        entrypoint={"command": "docker run image"},
        instance_id="instance-1",
    )

    assert runtime["container_id"] == _CONTAINER_ID
    assert handler.get_job_runtime_identity("job-1")["container_id"] == (
        _CONTAINER_ID
    )
    launch_command = cli.exec.call_args.args[1]
    assert "rm -f -- /tmp/tao-job-job-1.exit" in launch_command
    assert "/tmp/tao-job-job-1.cid" in launch_command
    assert "container_id=$(cat -- /tmp/tao-job-job-1.cid" in launch_command


def test_brev_sdk_labels_container_and_persists_returned_id():
    from tao_sdk.platforms.brev.sdk import BrevSDK

    handler = MagicMock()
    handler.get_container_name.return_value = "tao-job-job-1"
    handler.get_container_id_path.return_value = "/tmp/tao-job-job-1.cid"
    handler.get_exit_status_path.return_value = "/tmp/tao-job-job-1.exit"
    handler.get_job_name.return_value = "job-1"
    handler.create_job.return_value = {
        "instance_id": "instance-1",
        "pid": "1234",
        "container_id": _CONTAINER_ID,
        "launch_uncertain": False,
    }
    sdk = object.__new__(BrevSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    with patch("tao_sdk.platforms.brev.sdk.uuid.uuid4", return_value="job-1"):
        sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=1,
        )

    invocation = handler.create_job.call_args.kwargs["entrypoint"]["command"]
    assert "--cidfile /tmp/tao-job-job-1.cid" in invocation
    assert "--label tao-sdk-job-id=job-1" in invocation
    assert "--label tao-sdk-backend=brev" in invocation
    persisted = sdk._store.update_job_specs.call_args.args[1]
    assert persisted["_brev_runtime"]["container_id"] == _CONTAINER_ID


def test_brev_runtime_rejects_nonexact_container_id():
    from tao_sdk.platforms.brev.handler import BrevHandler

    handler = BrevHandler(cli=MagicMock())
    assert not handler.restore_job_runtime(
        "job-1",
        {
            "instance_id": "instance-1",
            "pid": "1234",
            "container_id": _REPLACEMENT_ID[:12],
            "launch_uncertain": False,
        },
    )
