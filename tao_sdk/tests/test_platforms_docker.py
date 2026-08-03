# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Docker platform handler.

Validates package wiring, init paths, and a few invariants. Heavier tests of
the docker.py handler internals (subprocess + docker client interactions) live
in the legacy test file imported alongside the handler when needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_handler_class_exposed_at_package_level():
    from tao_sdk.platforms.docker import DockerHandler, DockerSDK
    assert DockerHandler.__module__ == "tao_sdk.platforms.docker.handler"
    assert DockerSDK.__module__ == "tao_sdk.platforms.docker.sdk"


def test_handler_inherits_from_base():
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.base import BaseHandler
    assert issubclass(DockerHandler, BaseHandler)


def test_handler_get_job_name_is_docker_safe():
    from tao_sdk.platforms.docker.handler import DockerHandler
    handler = DockerHandler.__new__(DockerHandler)
    name = handler.get_job_name("3F8F-2D24/abc-XYZ")
    assert name.startswith("tao-job-")
    assert len(name) <= 128
    assert all(c.isalnum() or c in "_.-" for c in name)


def test_registry_entry_point_resolves_docker():
    """Entry point under tao_sdk.platforms group should resolve DockerHandler."""
    from importlib.metadata import entry_points
    eps = list(entry_points(group="tao_sdk.platforms"))
    docker_eps = [e for e in eps if e.name == "docker"]
    assert docker_eps, f"docker entry point missing; have: {[e.name for e in eps]}"
    cls = docker_eps[0].load()
    assert cls.__name__ == "DockerHandler"


def test_sdk_init_raises_when_docker_missing(monkeypatch):
    """If the docker Python package isn't importable, DockerSDK init raises CredentialError."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    from tao_sdk.models import CredentialError
    # Force the handler's _import_docker() to fail by patching importlib.
    with patch("tao_sdk.platforms.docker.handler.DockerHandler._import_docker", side_effect=ImportError("no docker")):
        with pytest.raises(CredentialError):
            DockerSDK()


# ---------------------------------------------------------------------------
# _results_root_prefix — what gets injected as TAO_RESULTS_ROOT
# ---------------------------------------------------------------------------

def test_docker_results_root_prefix_is_results_when_bind_mount_present():
    """A /results bind mount signals 'I have local persistence' — prefix is
    /results so the runner writes there directly, no cloud upload."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    from tao_sdk.platforms.docker.handler import DockerHandler
    sdk = object.__new__(DockerSDK)
    sdk._handler = object.__new__(DockerHandler)
    mounts = [{"host_path": "/host/out", "container_path": "/results"}]
    assert sdk._results_root_prefix(mounts=mounts) == "/results"


def test_docker_results_root_prefix_is_none_without_mount():
    """No /results bind mount → no TAO_RESULTS_ROOT injection. script_runner
    falls back to S3 (if S3_BUCKET_NAME) or container-ephemeral with warning."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    from tao_sdk.platforms.docker.handler import DockerHandler
    sdk = object.__new__(DockerSDK)
    sdk._handler = object.__new__(DockerHandler)
    assert sdk._results_root_prefix() is None


def test_docker_results_root_prefix_ignores_other_mounts():
    """Mounts at non-/results paths don't qualify."""
    from tao_sdk.platforms.docker.sdk import DockerSDK
    from tao_sdk.platforms.docker.handler import DockerHandler
    sdk = object.__new__(DockerSDK)
    sdk._handler = object.__new__(DockerHandler)
    mounts = [{"host_path": "/host/data", "container_path": "/data"}]
    assert sdk._results_root_prefix(mounts=mounts) is None


@pytest.mark.parametrize(
    "mount",
    (
        {"source": "/host/out", "target": "/results"},
        {"from_path": "/host/out", "mount_path": "/results"},
        "/host/out:/results:rw",
    ),
)
def test_docker_results_root_prefix_accepts_supported_mount_shapes(mount):
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._handler = object.__new__(DockerHandler)

    assert sdk._results_root_prefix(mounts=[mount]) == "/results"


def test_docker_results_root_prefix_rejects_read_only_results_mount():
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._handler = object.__new__(DockerHandler)

    assert sdk._results_root_prefix(
        mounts=[{"host_path": "/host/out", "container_path": "/results", "mode": "ro"}]
    ) is None


# ---------------------------------------------------------------------------
# Docker network prereq
# ---------------------------------------------------------------------------

def _network_test_handler():
    from tao_sdk.platforms.docker.handler import DockerHandler

    class NotFound(Exception):
        pass

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_network = "tao_default"
    handler._docker = SimpleNamespace(errors=SimpleNamespace(NotFound=NotFound))
    handler._client = MagicMock()
    handler._job_messages = {}
    handler._jobs = {}
    return handler, NotFound


def test_docker_handler_creates_missing_network():
    handler, NotFound = _network_test_handler()
    handler._client.networks.get.side_effect = NotFound("missing")

    handler._ensure_network("job-1")

    handler._client.networks.create.assert_called_once_with("tao_default")
    assert handler.get_job_message("job-1") == "Creating Docker network tao_default"


def test_docker_handler_does_not_create_existing_network():
    handler, _ = _network_test_handler()
    handler._client.networks.get.return_value = object()

    handler._ensure_network("job-1")

    handler._client.networks.create.assert_not_called()


# ---------------------------------------------------------------------------
# Local bind-mount ownership
# ---------------------------------------------------------------------------

def _job_test_handler(docker_host=""):
    from tao_sdk.platforms.docker.handler import DockerHandler

    class NotFound(Exception):
        pass

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = docker_host
    handler.docker_network = "tao_default"
    handler.auto_remove = False
    handler._docker = SimpleNamespace(errors=SimpleNamespace(NotFound=NotFound))
    handler._client = MagicMock()
    handler._client.info.return_value = {"SecurityOptions": []}
    handler._client.containers.run.return_value = SimpleNamespace(id="container-1")
    handler._jobs = {}
    handler._job_messages = {}
    handler._ensure_image = MagicMock()
    handler._remove_existing_container = MagicMock()
    handler._ensure_network = MagicMock()
    handler._save_state = MagicMock()
    handler._update_job_status = MagicMock()
    return handler


def _create_test_job(handler, mounts=None, **kwargs):
    handler.create_job(
        image="example.invalid/tao:test",
        network="",
        action="train",
        job_id="job-1",
        entrypoint={"command": "true"},
        mounts=mounts,
        num_gpus=0,
        **kwargs,
    )
    return handler._client.containers.run.call_args.kwargs


def test_local_writable_bind_auto_maps_invoking_user_and_precreates_source(
    tmp_path, monkeypatch,
):
    handler = _job_test_handler()
    output = tmp_path / "new" / "results"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)
    monkeypatch.setattr(
        "tao_sdk.platforms.docker.handler.os.getgroups",
        lambda: [4321, 987, 6543],
    )

    run_kwargs = _create_test_job(
        handler,
        mounts=[{"host_path": str(output), "container_path": "/results"}],
    )

    assert output.is_dir()
    assert run_kwargs["user"] == "1234:4321"
    assert run_kwargs["environment"]["USER"] == "1234"
    assert run_kwargs["environment"]["LOGNAME"] == "1234"
    assert run_kwargs["environment"]["HOME"] == (
        "/results/.tao-runtime/job-1/home"
    )
    assert run_kwargs["environment"]["XDG_CACHE_HOME"] == (
        "/results/.tao-runtime/job-1/home/.cache"
    )
    assert run_kwargs["environment"]["HF_HOME"].endswith("/.cache/huggingface")
    assert run_kwargs["environment"]["TORCH_HOME"].endswith("/.cache/torch")
    assert run_kwargs["environment"]["TORCHINDUCTOR_CACHE_DIR"].endswith(
        "/.cache/torchinductor"
    )
    assert run_kwargs["group_add"] == ["987", "6543"]
    assert (
        output
        / ".tao-runtime"
        / "job-1"
        / "home"
        / ".cache"
        / "huggingface"
    ).is_dir()


@pytest.mark.parametrize(
    "daemon_info",
    (
        {"Rootless": True, "SecurityOptions": []},
        {"SecurityOptions": ["name=rootless"]},
        {"SecurityOptions": ["name=userns"]},
        {
            "SecurityOptions": [
                "name=seccomp,profile=builtin",
                "name=userns-remap",
            ]
        },
    ),
)
def test_local_writable_bind_fails_closed_for_user_namespace_daemon(
    tmp_path, monkeypatch, daemon_info,
):
    handler = _job_test_handler()
    handler._client.info.return_value = daemon_info
    output = tmp_path / "results"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)

    with pytest.raises(RuntimeError, match="rootless/userns-remap"):
        _create_test_job(
            handler,
            mounts=[{"host_path": str(output), "container_path": "/results"}],
        )

    assert not output.exists()
    handler._client.containers.run.assert_not_called()


def test_local_writable_bind_fails_closed_when_daemon_security_is_unavailable(
    tmp_path, monkeypatch,
):
    handler = _job_test_handler()
    handler._client.info.side_effect = OSError("daemon unavailable")
    output = tmp_path / "results"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)

    with pytest.raises(RuntimeError, match="Could not inspect Docker daemon"):
        _create_test_job(
            handler,
            mounts=[{"host_path": str(output), "container_path": "/results"}],
        )

    assert not output.exists()
    handler._client.containers.run.assert_not_called()


def test_user_namespace_daemon_allows_reviewed_external_ownership_opt_out(
    tmp_path,
):
    handler = _job_test_handler()
    handler._client.info.return_value = {"SecurityOptions": ["name=rootless"]}
    output = tmp_path / "results"

    run_kwargs = _create_test_job(
        handler,
        mounts=[{"host_path": str(output), "container_path": "/results"}],
        run_as_user=False,
    )

    assert "user" not in run_kwargs


def test_non_root_mapping_preserves_explicit_environment(tmp_path, monkeypatch):
    handler = _job_test_handler()
    output = tmp_path / "results"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)

    run_kwargs = _create_test_job(
        handler,
        mounts=[{"host_path": str(output), "container_path": "/results"}],
        docker_env_vars={"HOME": "/workspace/home", "HF_HOME": "/hf-cache"},
    )

    assert run_kwargs["environment"]["HOME"] == "/workspace/home"
    assert run_kwargs["environment"]["HF_HOME"] == "/hf-cache"


def test_explicit_different_uid_uses_container_local_runtime_home(
    tmp_path, monkeypatch,
):
    handler = _job_test_handler()
    output = tmp_path / "results"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)

    run_kwargs = _create_test_job(
        handler,
        mounts=[{"host_path": str(output), "container_path": "/results"}],
        container_user="2000:3000",
    )

    assert run_kwargs["environment"]["HOME"] == "/tmp/tao-home-2000"
    assert run_kwargs["environment"]["XDG_CACHE_HOME"] == (
        "/tmp/tao-home-2000/.cache"
    )
    assert not (output / ".tao-runtime").exists()


def test_runtime_home_symlink_cannot_escape_results_bind(tmp_path, monkeypatch):
    handler = _job_test_handler()
    output = tmp_path / "results"
    outside = tmp_path / "outside"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)
    output.mkdir()
    outside.mkdir()
    (output / ".tao-runtime").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the /results bind source"):
        _create_test_job(
            handler,
            mounts=[{"host_path": str(output), "container_path": "/results"}],
        )

    handler._client.containers.run.assert_not_called()


def test_run_as_user_false_preserves_image_user_for_root_required_image(tmp_path):
    handler = _job_test_handler()
    output = tmp_path / "results"

    run_kwargs = _create_test_job(
        handler,
        mounts=[{"host_path": str(output), "container_path": "/results"}],
        run_as_user=False,
    )

    assert output.is_dir()
    assert "user" not in run_kwargs
    assert "HOME" not in run_kwargs["environment"]
    assert "XDG_CACHE_HOME" not in run_kwargs["environment"]


def test_local_writable_bind_refuses_automatic_root_mapping(tmp_path, monkeypatch):
    handler = _job_test_handler()
    output = tmp_path / "results"
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 0)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 0)

    with pytest.raises(RuntimeError, match="would create root-owned results"):
        _create_test_job(
            handler,
            mounts=[{"host_path": str(output), "container_path": "/results"}],
        )

    assert not output.exists()
    handler._client.containers.run.assert_not_called()


def test_writable_results_rejects_explicit_root_container_user(tmp_path):
    handler = _job_test_handler()
    output = tmp_path / "results"

    with pytest.raises(ValueError, match="cannot use an explicit root"):
        _create_test_job(
            handler,
            mounts=[{"host_path": str(output), "container_path": "/results"}],
            container_user="0:0",
        )

    assert not output.exists()
    handler._client.containers.run.assert_not_called()


def test_run_as_user_true_maps_local_job_without_bind_mount(monkeypatch):
    handler = _job_test_handler()
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1234)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 4321)

    run_kwargs = _create_test_job(handler, run_as_user=True)

    assert run_kwargs["user"] == "1234:4321"


def test_read_only_bind_does_not_trigger_user_mapping_or_directory_creation(tmp_path):
    handler = _job_test_handler()
    source = tmp_path / "missing-input"

    run_kwargs = _create_test_job(
        handler,
        mounts=[{
            "host_path": str(source),
            "container_path": "/data",
            "read_only": True,
        }],
    )

    assert not source.exists()
    assert "user" not in run_kwargs


def test_writable_input_bind_does_not_change_image_user_in_auto_mode(tmp_path):
    handler = _job_test_handler()
    source = tmp_path / "data"

    run_kwargs = _create_test_job(
        handler,
        mounts=[{"host_path": str(source), "container_path": "/data"}],
    )

    assert source.is_dir()
    assert "user" not in run_kwargs


def test_named_volume_does_not_trigger_host_user_mapping():
    handler = _job_test_handler()

    run_kwargs = _create_test_job(
        handler,
        mounts=[{
            "type": "volume",
            "source": "tao-results",
            "target": "/results",
        }],
    )

    assert "user" not in run_kwargs
    assert run_kwargs["volumes"] == {
        "tao-results": {"bind": "/results", "mode": "rw"},
    }


def test_remote_docker_auto_mode_requires_user_for_writable_results(tmp_path):
    handler = _job_test_handler("ssh://gpu.example")
    remote_source = tmp_path / "remote-results"

    with pytest.raises(ValueError, match="provide container_user explicitly"):
        _create_test_job(
            handler,
            mounts=[{
                "host_path": str(remote_source),
                "container_path": "/results",
            }],
        )

    assert not remote_source.exists()
    handler._client.containers.run.assert_not_called()


def test_remote_docker_explicit_root_opt_out_is_preserved(tmp_path):
    handler = _job_test_handler("ssh://gpu.example")

    run_kwargs = _create_test_job(
        handler,
        mounts=[{
            "host_path": str(tmp_path / "remote-results"),
            "container_path": "/results",
        }],
        run_as_user=False,
    )

    assert "user" not in run_kwargs
    assert "HOME" not in run_kwargs["environment"]


def test_remote_docker_explicit_run_as_user_requires_container_user(tmp_path):
    handler = _job_test_handler("ssh://gpu.example")

    with pytest.raises(ValueError, match="provide container_user explicitly"):
        _create_test_job(
            handler,
            mounts=[{
                "host_path": str(tmp_path / "remote-results"),
                "container_path": "/results",
            }],
            run_as_user=True,
        )

    handler._client.containers.run.assert_not_called()


def test_explicit_container_user_overrides_policy_for_remote_docker(tmp_path):
    handler = _job_test_handler("ssh://gpu.example")

    run_kwargs = _create_test_job(
        handler,
        mounts=[{
            "host_path": str(tmp_path / "remote-results"),
            "container_path": "/results",
        }],
        run_as_user=False,
        container_user="2000:3000",
    )

    assert run_kwargs["user"] == "2000:3000"
    assert run_kwargs["environment"]["USER"] == "2000"
    assert run_kwargs["environment"]["HOME"] == "/tmp/tao-home-2000"
    assert "group_add" not in run_kwargs


def test_docker_sdk_forwards_user_mapping_policy():
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._handler = MagicMock()
    sdk._handler.get_job_name.return_value = "tao-job-job-1"
    sdk._handler.get_backend_identity.return_value = "docker-1"
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
            run_as_user=True,
            container_user="1234:4321",
        )

    create_kwargs = sdk._handler.create_job.call_args.kwargs
    assert create_kwargs["run_as_user"] is True
    assert create_kwargs["container_user"] == "1234:4321"


# ---------------------------------------------------------------------------
# Ambiguous Docker create outcomes
# ---------------------------------------------------------------------------

def _stateful_job_test_handler():
    from tao_sdk.platforms.docker.handler import DockerHandler

    handler = _job_test_handler()
    handler._update_job_status = DockerHandler._update_job_status.__get__(handler)
    handler.get_backend_identity = MagicMock(return_value="docker-sha256:test")
    return handler


def _sdk_with_handler(handler, store=None):
    from tao_sdk.job_store import JobStore
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    sdk._store = store or JobStore(db_path=":memory:")
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    return sdk


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    ((0, "Complete"), (7, "Error")),
)
def test_docker_handler_delete_preserves_natural_terminal_result(
    exit_code, expected_status,
):
    handler = _stateful_job_test_handler()
    container = MagicMock()
    container.status = "exited"
    container.attrs = {"State": {"ExitCode": exit_code}}
    handler._client.containers.get.return_value = container
    container.id = "container-1"
    container.name = "tao-job-job-1"
    container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "status": "Running",
    }

    assert handler.delete_job("job-1")

    container.stop.assert_not_called()
    container.remove.assert_called_once_with(force=True)
    assert handler._jobs["job-1"]["status"] == expected_status
    assert handler._jobs["job-1"]["exit_code"] == exit_code


def test_docker_handler_absent_container_preserves_recorded_completion():
    handler = _stateful_job_test_handler()
    not_found = handler._docker.errors.NotFound
    handler._client.containers.get.side_effect = not_found("already removed")
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "status": "Complete",
    }

    assert handler.delete_job("job-1")
    assert handler._jobs["job-1"]["status"] == "Complete"


@pytest.mark.parametrize(
    ("exit_code", "expected_raw", "expected_status"),
    ((0, None, "Complete"), (23, "Error", "Error")),
)
def test_docker_handler_recovers_auto_removed_container_exit_event(
    exit_code, expected_raw, expected_status,
):
    handler = _stateful_job_test_handler()
    not_found = handler._docker.errors.NotFound
    handler._client.containers.get.side_effect = not_found("auto removed")
    handler._client.events.return_value = [{
        "Action": "die",
        "Actor": {
            "ID": "container-1",
            "Attributes": {
                "tao-sdk-job-id": "job-1",
                "tao-sdk-backend": "local-docker",
                "exitCode": str(exit_code),
            },
        },
    }]
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.get_tao_job_status("job-1") == expected_raw
    assert handler._jobs["job-1"]["status"] == expected_status
    assert handler._jobs["job-1"]["exit_code"] == exit_code


def test_docker_handler_removing_without_exit_evidence_is_nonterminal():
    handler = _stateful_job_test_handler()
    container = MagicMock()
    container.id = "container-1"
    container.name = "tao-job-job-1"
    container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    container.status = "removing"
    handler._client.containers.get.return_value = container
    handler._client.events.return_value = []
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.get_tao_job_status("job-1") == "Pending"
    assert handler._jobs["job-1"]["status"] == "Running"


def test_docker_handler_removing_uses_exact_die_event_not_cancellation_guess():
    handler = _stateful_job_test_handler()
    container = MagicMock()
    container.id = "container-1"
    container.name = "tao-job-job-1"
    container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    container.status = "removing"
    handler._client.containers.get.return_value = container
    handler._client.events.return_value = [{
        "Action": "die",
        "Actor": {
            "ID": "container-1",
            "Attributes": {
                "tao-sdk-job-id": "job-1",
                "tao-sdk-backend": "local-docker",
                "exitCode": "17",
            },
        },
    }]
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.get_tao_job_status("job-1") == "Error"
    assert handler._jobs["job-1"]["status"] == "Error"
    assert handler._jobs["job-1"]["exit_code"] == 17


def test_docker_handler_stop_auto_remove_race_preserves_cancellation_intent():
    handler = _stateful_job_test_handler()
    not_found = handler._docker.errors.NotFound
    container = MagicMock()
    container.id = "container-1"
    container.name = "tao-job-job-1"
    container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    container.status = "running"
    container.remove.side_effect = not_found("auto removed after stop")
    handler._client.containers.get.side_effect = [
        container,
        not_found("bound ID absent"),
        not_found("name absent"),
    ]
    handler._client.events.return_value = [{
        "Action": "die",
        "Actor": {
            "ID": "container-1",
            "Attributes": {
                "tao-sdk-job-id": "job-1",
                "tao-sdk-backend": "local-docker",
                "exitCode": "137",
            },
        },
    }]
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.delete_job("job-1") is True
    container.stop.assert_called_once_with(timeout=10)
    assert handler._jobs["job-1"]["status"] == "Canceled"
    assert handler._jobs["job-1"]["exit_code"] == 137


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    ((0, "Complete"), (19, "Error")),
)
def test_docker_handler_terminal_remove_race_preserves_natural_result(
    exit_code, expected_status,
):
    handler = _stateful_job_test_handler()
    not_found = handler._docker.errors.NotFound
    container = MagicMock()
    container.id = "container-1"
    container.name = "tao-job-job-1"
    container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    container.status = "exited"
    container.attrs = {"State": {"ExitCode": exit_code}}
    container.remove.side_effect = not_found("removed concurrently")
    handler._client.containers.get.side_effect = [
        container,
        not_found("bound ID absent"),
        not_found("name absent"),
    ]
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.delete_job("job-1") is True
    container.stop.assert_not_called()
    assert handler._jobs["job-1"]["status"] == expected_status
    assert handler._jobs["job-1"]["exit_code"] == exit_code
    handler._client.events.assert_not_called()


def test_docker_handler_refuses_replacement_with_same_container_name():
    handler = _stateful_job_test_handler()
    not_found = handler._docker.errors.NotFound
    replacement = MagicMock()
    replacement.id = "replacement-container"
    replacement.name = "tao-job-job-1"
    replacement.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    handler._client.containers.get.side_effect = [
        not_found("recorded container removed"),
        replacement,
        not_found("recorded container removed"),
        replacement,
    ]
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.get_tao_job_status("job-1") == "Pending"
    assert not handler.cancel_job("job-1")
    replacement.stop.assert_not_called()
    replacement.remove.assert_not_called()


def test_docker_handler_refuses_bound_container_with_wrong_labels():
    handler = _stateful_job_test_handler()
    replacement = MagicMock()
    replacement.id = "container-1"
    replacement.name = "tao-job-job-1"
    replacement.labels = {
        "tao-sdk-job-id": "some-other-job",
        "tao-sdk-backend": "local-docker",
    }
    handler._client.containers.get.return_value = replacement
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
        "submission_attempted": True,
        "status": "Running",
    }

    assert handler.get_tao_job_status("job-1") == "Pending"
    assert not handler.cancel_job("job-1")
    replacement.stop.assert_not_called()
    replacement.remove.assert_not_called()


def test_docker_handler_absence_does_not_resolve_ambiguous_create():
    handler = _stateful_job_test_handler()
    not_found = handler._docker.errors.NotFound
    handler._client.containers.run.side_effect = TimeoutError("read timed out")
    handler._client.containers.get.side_effect = not_found("not visible yet")

    with pytest.raises(TimeoutError, match="read timed out"):
        _create_test_job(handler)

    state = handler._jobs["job-1"]
    assert state["container_name"] == "tao-job-job-1"
    assert state["submission_attempted"] is True
    assert state["launch_uncertain"] is True
    assert state["status"] == "Pending"

    assert handler.cancel_job("job-1") is False
    assert handler.get_tao_job_status("job-1") == "Pending"
    assert handler._jobs["job-1"]["launch_uncertain"] is True


def test_docker_sdk_tracks_late_container_after_create_timeout():
    from tao_sdk.artifacts import artifact_metadata

    handler = _stateful_job_test_handler()
    sdk = _sdk_with_handler(handler)
    not_found = handler._docker.errors.NotFound

    runtime_at_run = []

    def time_out_after_request(**_kwargs):
        runtime_at_run.append(
            artifact_metadata(sdk._store.get_job("job-1"))["runtime_identity"]
        )
        raise TimeoutError("read timed out")

    handler._client.containers.run.side_effect = time_out_after_request
    handler._client.containers.get.side_effect = not_found("not visible yet")

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        job = sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
        )

    stored = sdk._store.get_job("job-1")
    assert job.id == "job-1"
    assert job.backend_job_id == "tao-job-job-1"
    assert stored["backend_job_id"] == "tao-job-job-1"
    assert stored["status"] == "Pending"
    assert runtime_at_run == [{
        "container_name": "tao-job-job-1",
        "container_id": "",
        "launch_uncertain": True,
    }]
    # One deterministic-name lookup captures any natural terminal outcome and
    # a second attempts cancellation. Unlabeled fallback names are not trusted.
    assert handler._client.containers.get.call_count == 2
    assert sdk.get_job_status(job.id).status == "Pending"
    assert sdk._store.get_job(job.id)["status"] == "Pending"

    late_container = MagicMock()
    late_container.id = "late-container"
    late_container.name = "tao-job-job-1"
    late_container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    late_container.status = "running"
    handler._client.containers.get.side_effect = None
    handler._client.containers.get.return_value = late_container

    assert sdk.get_job_status(job.id).status == "Running"
    assert sdk._store.get_job(job.id)["status"] == "Running"
    assert handler._jobs[job.id]["container_id"] == "late-container"
    assert handler._jobs[job.id]["launch_uncertain"] is False
    assert artifact_metadata(
        sdk._store.get_job(job.id)
    )["runtime_identity"] == {
        "container_name": "tao-job-job-1",
        "container_id": "late-container",
        "launch_uncertain": False,
    }


def test_docker_sdk_persists_confirmed_container_identity():
    from tao_sdk.artifacts import artifact_metadata

    handler = _stateful_job_test_handler()
    sdk = _sdk_with_handler(handler)

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
        )

    assert artifact_metadata(
        sdk._store.get_job("job-1")
    )["runtime_identity"] == {
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
    }


def test_docker_runtime_identity_is_restored_from_job_store_after_restart():
    from tao_sdk.artifacts import artifact_metadata

    handler = _stateful_job_test_handler()
    sdk = _sdk_with_handler(handler)
    not_found = handler._docker.errors.NotFound
    handler._client.containers.run.side_effect = TimeoutError("read timed out")
    handler._client.containers.get.side_effect = not_found("not visible yet")

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
        )

    recovered_handler = _stateful_job_test_handler()
    recovered_handler._jobs = {
        "job-1": {
            "job_id": "job-1",
            "container_name": "stale-cache-name",
            "container_id": "stale-cache-id",
            "launch_uncertain": False,
            "status": "Error",
        }
    }
    recovered_not_found = recovered_handler._docker.errors.NotFound
    recovered_handler._client.containers.get.side_effect = recovered_not_found(
        "not visible yet"
    )
    recovered = _sdk_with_handler(recovered_handler, store=sdk._store)

    assert recovered.get_job_status("job-1").status == "Pending"
    assert recovered_handler._jobs["job-1"]["container_name"] == "tao-job-job-1"
    assert recovered_handler._jobs["job-1"]["container_id"] == ""
    assert recovered_handler._jobs["job-1"]["launch_uncertain"] is True
    assert recovered_handler._jobs["job-1"]["status"] == "Pending"

    late_container = MagicMock()
    late_container.id = "late-container"
    late_container.name = "tao-job-job-1"
    late_container.labels = {
        "tao-sdk-job-id": "job-1",
        "tao-sdk-backend": "local-docker",
    }
    late_container.status = "running"
    recovered_handler._client.containers.get.side_effect = None
    recovered_handler._client.containers.get.return_value = late_container

    assert recovered.get_job_status("job-1").status == "Running"
    assert artifact_metadata(
        sdk._store.get_job("job-1")
    )["runtime_identity"] == {
        "container_name": "tao-job-job-1",
        "container_id": "late-container",
        "launch_uncertain": False,
    }


def test_docker_interrupted_ambiguous_create_exposes_durable_identity():
    from tao_sdk.artifacts import artifact_metadata

    handler = _stateful_job_test_handler()
    sdk = _sdk_with_handler(handler)
    not_found = handler._docker.errors.NotFound
    interrupt = KeyboardInterrupt("interrupted")
    handler._client.containers.run.side_effect = interrupt
    handler._client.containers.get.side_effect = not_found("not visible yet")

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(KeyboardInterrupt) as caught:
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    assert caught.value is interrupt
    assert caught.value.tao_job_id == "job-1"
    assert caught.value.job_id == "job-1"
    assert caught.value.backend_job_id == "tao-job-job-1"
    assert caught.value.launch_uncertain is True
    assert caught.value.runtime_identity == {
        "container_name": "tao-job-job-1",
        "container_id": "",
        "launch_uncertain": True,
    }
    assert caught.value.durable_job_identity == {
        "job_id": "job-1",
        "backend_job_id": "tao-job-job-1",
        "container_name": "tao-job-job-1",
        "container_id": "",
        "launch_uncertain": True,
    }
    assert artifact_metadata(
        sdk._store.get_job("job-1")
    )["runtime_identity"]["launch_uncertain"] is True


def test_docker_interrupted_confirmed_rollback_is_terminal_and_identified():
    handler = _stateful_job_test_handler()
    interrupt = KeyboardInterrupt("interrupted after create")
    handler.create_job = MagicMock(side_effect=interrupt)
    handler.cancel_job = MagicMock(return_value=True)
    handler.get_job_runtime_identity = MagicMock(return_value={
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
    })
    sdk = _sdk_with_handler(handler)

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(KeyboardInterrupt) as caught:
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    assert caught.value.tao_job_id == "job-1"
    assert caught.value.tao_launch_uncertain is False
    assert sdk._store.get_job("job-1")["status"] == "Canceled"


@pytest.mark.parametrize(
    "launch_exception",
    (
        RuntimeError("Docker client failed after the container exited"),
        KeyboardInterrupt("interrupted after the container exited"),
    ),
    ids=("exception", "base_exception"),
)
def test_docker_sdk_launch_cleanup_preserves_natural_completion(
    launch_exception,
):
    handler = _stateful_job_test_handler()
    handler.create_job = MagicMock(side_effect=launch_exception)
    handler.cancel_job = MagicMock(return_value=True)
    handler.get_tao_job_status = MagicMock(return_value=None)
    sdk = _sdk_with_handler(handler)

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(type(launch_exception)):
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    assert sdk._store.get_job("job-1")["status"] == "Complete"


def test_docker_sdk_still_surfaces_prelaunch_failure():
    from tao_sdk.artifacts import artifact_metadata

    handler = _stateful_job_test_handler()
    sdk = _sdk_with_handler(handler)
    not_found = handler._docker.errors.NotFound
    handler._ensure_image.side_effect = ValueError("invalid image reference")
    handler._client.containers.get.side_effect = not_found("never created")

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(ValueError, match="invalid image reference"):
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    assert sdk._store.get_job("job-1")["status"] == "Error"
    assert artifact_metadata(
        sdk._store.get_job("job-1")
    )["runtime_identity"]["launch_uncertain"] is False
    assert handler._jobs["job-1"].get("launch_uncertain") is not True
    handler._client.containers.run.assert_not_called()


def test_docker_sdk_refuses_name_collision_without_touching_replacement():
    from tao_sdk.platforms.docker.handler import (
        DockerContainerIdentityError,
        DockerHandler,
    )

    handler = _stateful_job_test_handler()
    handler._remove_existing_container = (
        DockerHandler._remove_existing_container.__get__(handler)
    )
    unrelated = MagicMock()
    unrelated.id = "unrelated-container"
    unrelated.name = "tao-job-job-1"
    unrelated.labels = {
        "tao-sdk-job-id": "some-other-job",
        "tao-sdk-backend": "local-docker",
    }
    handler._client.containers.get.return_value = unrelated
    sdk = _sdk_with_handler(handler)

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(DockerContainerIdentityError, match="refusing to replace"):
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    unrelated.stop.assert_not_called()
    unrelated.remove.assert_not_called()
    handler._client.containers.run.assert_not_called()
    assert sdk._store.get_job("job-1")["status"] == "Error"
