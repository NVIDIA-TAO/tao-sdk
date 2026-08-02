# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for job-scoped AutoML artifact cleanup support."""

from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tao_sdk.artifacts import (
    artifact_specs,
    delete_local_job_results,
    delete_remote_job_results,
    job_scoped_remote_path,
    s3_credential_fingerprint,
)


def test_artifact_cleanup_protocol_exports_retention_preflight():
    from tao_sdk import ArtifactCleanupSDK as RootCapability
    from tao_sdk.platforms import ArtifactCleanupSDK as PlatformCapability

    assert RootCapability is PlatformCapability
    assert hasattr(RootCapability, "validate_artifact_retention")
    assert hasattr(RootCapability, "delete_job_artifacts")


@pytest.mark.parametrize("job_id", ("", ".", "..", "../job", "a/b", "a\\b"))
def test_remote_job_path_rejects_unsafe_job_ids(job_id):
    with pytest.raises(ValueError):
        job_scoped_remote_path("s3://bucket/results/job", job_id, "s3")


def test_remote_job_path_requires_exact_results_prefix():
    assert (
        job_scoped_remote_path("s3://bucket/results/job-1", "job-1", "s3")
        == "bucket/results/job-1"
    )
    with pytest.raises(ValueError):
        job_scoped_remote_path("s3://bucket/results/job-2", "job-1", "s3")
    with pytest.raises(ValueError):
        job_scoped_remote_path("s3://bucket/job-1", "job-1", "s3")


def test_delete_remote_job_results_is_recursive_and_idempotent():
    fs = MagicMock()
    fs.exists.side_effect = [True, False, False]

    assert delete_remote_job_results(
        fs, "s3://bucket/results/job-1", "job-1", "s3"
    )
    fs.rm.assert_called_once_with("bucket/results/job-1", recursive=True)

    fs.rm.reset_mock()
    assert delete_remote_job_results(
        fs, "s3://bucket/results/job-1", "job-1", "s3"
    )
    fs.rm.assert_not_called()


def test_delete_remote_job_results_refuses_silent_partial_deletion():
    fs = MagicMock()
    fs.exists.return_value = True

    assert not delete_remote_job_results(
        fs,
        "s3://bucket/results/job-1",
        "job-1",
        "s3",
        verify_attempts=3,
        verify_interval=0,
    )
    fs.rm.assert_called_once_with("bucket/results/job-1", recursive=True)
    assert fs.exists.call_count == 4


def test_delete_local_job_results_removes_only_job_directory(tmp_path):
    results_root = tmp_path / "results"
    target = results_root / "job-1"
    sibling = results_root / "job-2"
    target.mkdir(parents=True)
    sibling.mkdir()
    (target / "checkpoint.pt").write_bytes(b"checkpoint")

    assert delete_local_job_results(target, "job-1", expected_path=target)
    assert not target.exists()
    assert sibling.is_dir()
    assert delete_local_job_results(target, "job-1", expected_path=target)


def test_delete_local_job_results_unlinks_symlink_without_following_it(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    target = tmp_path / "job-1"
    target.symlink_to(outside, target_is_directory=True)

    assert delete_local_job_results(target, "job-1", expected_path=target)
    assert not os.path.lexists(target)
    assert (outside / "keep.txt").is_file()


def test_delete_local_job_results_refuses_leaf_symlink_swap(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    target = tmp_path / "job-1"
    target.mkdir()
    (target / "checkpoint.pt").write_bytes(b"checkpoint")
    real_rmtree = __import__("shutil").rmtree

    def swap_before_delete(path, *, dir_fd):
        os.rename(
            path,
            "original-job-1",
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
        )
        os.symlink(outside, path, dir_fd=dir_fd, target_is_directory=True)
        return real_rmtree(path, dir_fd=dir_fd)

    swap_before_delete.avoids_symlink_attacks = True
    monkeypatch.setattr("tao_sdk.artifacts.shutil.rmtree", swap_before_delete)

    with pytest.raises(OSError):
        delete_local_job_results(target, "job-1", expected_path=target)
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_delete_local_job_results_verifies_postcondition(tmp_path, monkeypatch):
    target = tmp_path / "job-1"
    target.mkdir()
    (target / "checkpoint.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "tao_sdk.artifacts.shutil.rmtree",
        MagicMock(avoids_symlink_attacks=True),
    )

    assert not delete_local_job_results(target, "job-1", expected_path=target)
    assert target.is_dir()


def test_delete_local_job_results_refuses_parent_directory_swap(
    tmp_path, monkeypatch,
):
    results_root = tmp_path / "results"
    target = results_root / "job-1"
    target.mkdir(parents=True)
    (target / "checkpoint.pt").write_bytes(b"checkpoint")
    displaced_root = tmp_path / "displaced-results"
    real_rmtree = __import__("shutil").rmtree

    def swap_parent_after_delete(path, *, dir_fd):
        real_rmtree(path, dir_fd=dir_fd)
        results_root.rename(displaced_root)
        target.mkdir(parents=True)

    swap_parent_after_delete.avoids_symlink_attacks = True
    monkeypatch.setattr(
        "tao_sdk.artifacts.shutil.rmtree", swap_parent_after_delete
    )

    assert not delete_local_job_results(target, "job-1", expected_path=target)
    assert target.is_dir()
    assert displaced_root.is_dir()


def test_docker_sdk_derives_and_deletes_local_bind_results(tmp_path):
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = ""
    handler._client = MagicMock()
    handler._client.info.return_value = {"ID": "daemon-1"}
    handler.restore_job_runtime = MagicMock(return_value=True)
    handler.get_tao_job_status = MagicMock(return_value=None)
    handler.normalize_status = MagicMock(return_value="Complete")
    handler.delete_job = MagicMock(return_value=True)
    backend_identity = handler.get_backend_identity()
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    sdk._store = MagicMock()

    local_results = sdk._local_results_dir(
        "job-1",
        [{"host_path": str(tmp_path), "container_path": "/results"}],
    )
    target = tmp_path / "job-1"
    target.mkdir()
    (target / "checkpoint.pt").write_bytes(b"checkpoint")
    runtime_target = tmp_path / ".tao-runtime" / "job-1"
    runtime_target.mkdir(parents=True)
    (runtime_target / "cache.bin").write_bytes(b"cache")
    runtime_sibling = tmp_path / ".tao-runtime" / "job-2"
    runtime_sibling.mkdir()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "docker",
        "status": "Complete",
        "local_results_dir": local_results,
        "results_dir": local_results,
        "specs": json.dumps(artifact_specs(
            "local_bind",
            str(tmp_path),
            backend_identity=backend_identity,
            runtime_identity={
                "container_name": "tao-job-job-1",
                "container_id": "container-1",
                "launch_uncertain": False,
            },
        )),
    }

    assert local_results == str(target)
    assert sdk.get_job_results_dir("job-1") == str(target)
    assert sdk.delete_job_artifacts("job-1")
    assert not target.exists()
    assert not runtime_target.exists()
    assert runtime_sibling.is_dir()
    handler.delete_job.assert_called_once_with("job-1")


def test_docker_create_job_persists_local_results_directory(tmp_path):
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = ""
    handler._client = MagicMock()
    handler._client.info.return_value = {"ID": "daemon-1"}
    handler.create_job = MagicMock()
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        job = sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
            mounts=[{"host_path": str(tmp_path), "container_path": "/results"}],
        )

    expected = str(tmp_path / "job-1")
    assert job.results_dir == expected
    assert job.local_results_dir == expected
    assert sdk._store.create_job.call_args.kwargs["local_results_dir"] == expected
    assert sdk._store.create_job.call_count == 1


def test_docker_backend_identity_includes_endpoint_and_daemon_id():
    from tao_sdk.platforms.docker.handler import DockerHandler

    default = DockerHandler.__new__(DockerHandler)
    default.docker_host = ""
    default._client = MagicMock()
    default._client.info.return_value = {"ID": "daemon-1"}

    explicit = DockerHandler.__new__(DockerHandler)
    explicit.docker_host = "unix:///var/run/docker.sock"
    explicit._client = MagicMock()
    explicit._client.info.return_value = {"ID": "daemon-1"}

    replacement = DockerHandler.__new__(DockerHandler)
    replacement.docker_host = "/var/run/docker.sock"
    replacement._client = MagicMock()
    replacement._client.info.return_value = {"ID": "daemon-2"}

    assert default.get_backend_identity() == explicit.get_backend_identity()
    assert default.get_backend_identity() != replacement.get_backend_identity()


def test_docker_sdk_refuses_a_job_from_a_replaced_daemon(tmp_path):
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._handler = MagicMock()
    sdk._handler.get_backend_identity.return_value = "docker-new"
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    target = tmp_path / "job-1"
    target.mkdir()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "backend_job_id": "tao-job-job-1",
        "local_results_dir": str(target),
        "results_dir": str(target),
        "specs": json.dumps(artifact_specs(
            "local_bind",
            str(tmp_path),
            backend_identity="docker-old",
        )),
    }

    assert sdk.get_job_status("job-1").status == "Unknown"
    assert not sdk.cancel_job("job-1")
    assert not sdk.delete_job_artifacts("job-1")
    sdk._handler.get_tao_job_status.assert_not_called()
    sdk._handler.cancel_job.assert_not_called()
    assert target.is_dir()


def _configure_terminal_docker_cleanup(sdk) -> dict:
    runtime = {
        "container_name": "tao-job-job-1",
        "container_id": "container-1",
        "launch_uncertain": False,
    }
    sdk._handler.restore_job_runtime.return_value = True
    sdk._handler.get_tao_job_status.return_value = None
    sdk._handler.normalize_status.return_value = "Complete"
    sdk._handler.delete_job.return_value = True
    return runtime


def _configure_terminal_kubernetes_cleanup(sdk) -> dict:
    runtime = {
        "job_name": "tao-job-job-1",
        "job_uid": "uid-1",
        "namespace": "default",
        "num_nodes": 1,
        "service_name": "",
        "service_uid": "",
        "launch_uncertain": False,
    }
    sdk._handler.restore_job_runtime.return_value = True
    sdk._handler.delete_terminal_job.return_value = True
    return runtime


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_s3_platforms_delete_only_the_recorded_job_prefix(
    sdk_path, class_name, monkeypatch
):
    import importlib

    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    fs = MagicMock()
    fs.exists.side_effect = [True, False]
    sdk._get_fs = MagicMock(return_value=fs)
    sdk._store = MagicMock()
    monkeypatch.setenv("ACCESS_KEY", "job-owner")
    monkeypatch.setenv("SECRET_KEY", "current-secret")
    backend_identity = ""
    runtime_identity = None
    if class_name in {"DockerSDK", "KubernetesSDK"}:
        sdk._handler = MagicMock()
        sdk._handler.get_backend_identity.return_value = "backend-1"
        backend_identity = "backend-1"
    if class_name == "DockerSDK":
        runtime_identity = _configure_terminal_docker_cleanup(sdk)
    elif class_name == "KubernetesSDK":
        runtime_identity = _configure_terminal_kubernetes_cleanup(sdk)
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "results_dir": "s3://bucket/results/job-1",
        "local_results_dir": "",
        "specs": json.dumps(artifact_specs(
            "s3",
            "s3://bucket/results/job-1",
            endpoint_url="https://minio.test",
            credential_fingerprint=s3_credential_fingerprint("job-owner"),
            backend_identity=backend_identity,
            runtime_identity=runtime_identity,
        )),
    }

    assert sdk.delete_job_artifacts("job-1")
    sdk._get_fs.assert_called_once_with(
        "https://minio.test",
        access_key="job-owner",
        secret_key="current-secret",
    )
    fs.rm.assert_called_once_with("bucket/results/job-1", recursive=True)
    sdk._store.delete_checkpoints.assert_called_once_with("job-1")


def test_docker_artifact_cleanup_requires_terminal_container_removal(monkeypatch):
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._handler = MagicMock()
    sdk._handler.get_backend_identity.return_value = "backend-1"
    runtime = _configure_terminal_docker_cleanup(sdk)
    sdk._handler.delete_job.return_value = False
    sdk._store = MagicMock()
    fs = MagicMock()
    sdk._get_fs = MagicMock(return_value=fs)
    monkeypatch.setenv("ACCESS_KEY", "job-owner")
    monkeypatch.setenv("SECRET_KEY", "job-secret")
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "results_dir": "s3://bucket/results/job-1",
        "local_results_dir": "",
        "specs": json.dumps(artifact_specs(
            "s3",
            "s3://bucket/results/job-1",
            credential_fingerprint=s3_credential_fingerprint("job-owner"),
            backend_identity="backend-1",
            runtime_identity=runtime,
        )),
    }

    assert not sdk.delete_job_artifacts("job-1")
    fs.rm.assert_not_called()
    sdk._store.delete_checkpoints.assert_not_called()


def test_kubernetes_artifact_cleanup_requires_exact_job_and_pod_removal(
    monkeypatch,
):
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

    sdk = object.__new__(KubernetesSDK)
    sdk._handler = MagicMock()
    sdk._handler.get_backend_identity.return_value = "backend-1"
    runtime = _configure_terminal_kubernetes_cleanup(sdk)
    sdk._handler.delete_terminal_job.return_value = False
    sdk._store = MagicMock()
    fs = MagicMock()
    sdk._get_fs = MagicMock(return_value=fs)
    monkeypatch.setenv("ACCESS_KEY", "job-owner")
    monkeypatch.setenv("SECRET_KEY", "job-secret")
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "results_dir": "s3://bucket/results/job-1",
        "local_results_dir": "",
        "specs": json.dumps(artifact_specs(
            "s3",
            "s3://bucket/results/job-1",
            credential_fingerprint=s3_credential_fingerprint("job-owner"),
            backend_identity="backend-1",
            runtime_identity=runtime,
        )),
    }

    assert not sdk.delete_job_artifacts("job-1")
    fs.rm.assert_not_called()
    sdk._store.delete_checkpoints.assert_not_called()


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_s3_cleanup_reuses_per_job_call_credentials_without_persisting_secrets(
    sdk_path, class_name, monkeypatch
):
    import importlib

    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    fs = MagicMock()
    fs.exists.side_effect = [True, False]
    sdk._get_fs = MagicMock(return_value=fs)
    sdk._store = MagicMock()
    sdk._artifact_credentials = {"job-1": ("per-job-owner", "per-job-secret")}
    monkeypatch.setenv("ACCESS_KEY", "unrelated-ambient-owner")
    monkeypatch.setenv("SECRET_KEY", "unrelated-ambient-secret")
    backend_identity = ""
    runtime_identity = None
    if class_name in {"DockerSDK", "KubernetesSDK"}:
        sdk._handler = MagicMock()
        sdk._handler.get_backend_identity.return_value = "backend-1"
        backend_identity = "backend-1"
    if class_name == "DockerSDK":
        runtime_identity = _configure_terminal_docker_cleanup(sdk)
    elif class_name == "KubernetesSDK":
        runtime_identity = _configure_terminal_kubernetes_cleanup(sdk)
    specs = artifact_specs(
        "s3",
        "s3://bucket/results/job-1",
        endpoint_url="https://minio.test",
        credential_fingerprint=s3_credential_fingerprint("per-job-owner"),
        backend_identity=backend_identity,
        runtime_identity=runtime_identity,
    )
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "results_dir": "s3://bucket/results/job-1",
        "local_results_dir": "",
        "specs": json.dumps(specs),
    }

    assert "per-job-secret" not in json.dumps(specs)
    assert sdk.delete_job_artifacts("job-1")
    sdk._get_fs.assert_called_once_with(
        "https://minio.test",
        access_key="per-job-owner",
        secret_key="per-job-secret",
    )
    assert "job-1" not in sdk._artifact_credentials


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_s3_retention_preflight_requires_explicit_reproducible_principal(
    sdk_path, class_name, monkeypatch
):
    import importlib

    for key in (
        "ACCESS_KEY", "SECRET_KEY", "S3_BUCKET_NAME",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    if class_name == "DockerSDK":
        sdk._handler = MagicMock()
        sdk._handler._build_volumes.return_value = None

    with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
        sdk.validate_artifact_retention(
            env_vars={"S3_BUCKET_NAME": "bucket"},
            irrelevant_platform_option=True,
        )
    with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
        sdk.validate_artifact_retention(env_vars={
            "S3_BUCKET_NAME": "bucket",
            "AWS_ACCESS_KEY_ID": "owner",
        })
    assert sdk.validate_artifact_retention(env_vars={
        "S3_BUCKET_NAME": "bucket",
        "AWS_ACCESS_KEY_ID": "owner",
        "AWS_SECRET_ACCESS_KEY": "secret",
    })
    if class_name == "DockerSDK":
        with pytest.raises(ValueError, match="run_as_user=False"):
            sdk.validate_artifact_retention(
                env_vars={
                    "S3_BUCKET_NAME": "bucket",
                    "AWS_ACCESS_KEY_ID": "owner",
                    "AWS_SECRET_ACCESS_KEY": "secret",
                },
                run_as_user=False,
            )


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_s3_retention_preflight_credentials_survive_sdk_reinstantiation(
    sdk_path, class_name, monkeypatch
):
    import importlib
    import tao_sdk.artifacts as artifact_helpers

    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    owner = f"resume-owner-{class_name}"
    secret = f"resume-secret-{class_name}"
    preflight_sdk = object.__new__(sdk_class)
    if class_name == "DockerSDK":
        preflight_sdk._handler = MagicMock()
        preflight_sdk._handler._build_volumes.return_value = None
    recovery_config = {
        "S3_BUCKET_NAME": "bucket",
        "AWS_ACCESS_KEY_ID": owner,
        "AWS_SECRET_ACCESS_KEY": secret,
        "AWS_ENDPOINT_URL": "HTTPS://MINIO.TEST:443/",
    }
    assert preflight_sdk.validate_artifact_retention(env_vars=recovery_config)

    # Simulate a process restart, then replay the durable run configuration's
    # explicit credentials through preflight before reconstructing the SDK.
    with artifact_helpers._PROCESS_S3_CREDENTIALS_LOCK:
        artifact_helpers._PROCESS_S3_CREDENTIALS.clear()
    recovery_preflight = object.__new__(sdk_class)
    if class_name == "DockerSDK":
        recovery_preflight._handler = MagicMock()
        recovery_preflight._handler._build_volumes.return_value = None
    assert recovery_preflight.validate_artifact_retention(env_vars=recovery_config)

    # Simulate the AutoML runner rebuilding its SDK object in the same process.
    resumed_sdk = object.__new__(sdk_class)
    resumed_sdk._artifact_credentials = {}
    resumed_sdk._store = MagicMock()
    fs = MagicMock()
    fs.exists.side_effect = [True, False]
    resumed_sdk._get_fs = MagicMock(return_value=fs)
    backend_identity = ""
    runtime_identity = None
    if class_name in {"DockerSDK", "KubernetesSDK"}:
        resumed_sdk._handler = MagicMock()
        resumed_sdk._handler.get_backend_identity.return_value = "backend-1"
        backend_identity = "backend-1"
    if class_name == "DockerSDK":
        runtime_identity = _configure_terminal_docker_cleanup(resumed_sdk)
    elif class_name == "KubernetesSDK":
        runtime_identity = _configure_terminal_kubernetes_cleanup(resumed_sdk)
    specs = artifact_specs(
        "s3",
        "s3://bucket/results/job-1",
        endpoint_url="HTTPS://MINIO.TEST:443/",
        credential_fingerprint=s3_credential_fingerprint(owner),
        backend_identity=backend_identity,
        runtime_identity=runtime_identity,
    )
    resumed_sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "results_dir": "s3://bucket/results/job-1",
        "local_results_dir": "",
        "specs": json.dumps(specs),
    }
    for key in (
        "ACCESS_KEY", "SECRET_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    assert secret not in json.dumps(specs)
    assert resumed_sdk.delete_job_artifacts("job-1")
    resumed_sdk._get_fs.assert_called_once_with(
        "HTTPS://MINIO.TEST:443/", access_key=owner, secret_key=secret
    )


def test_process_s3_cache_isolated_by_normalized_endpoint(monkeypatch):
    from tao_sdk.platforms.brev.sdk import BrevSDK

    access_key = "shared-access-key"
    first = object.__new__(BrevSDK)
    second = object.__new__(BrevSDK)
    assert first.validate_artifact_retention(env_vars={
        "S3_BUCKET_NAME": "bucket-a",
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": "secret-a",
        "AWS_ENDPOINT_URL": "https://STORE-A.test:443/",
    })
    assert second.validate_artifact_retention(env_vars={
        "S3_BUCKET_NAME": "bucket-b",
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": "secret-b",
        "AWS_ENDPOINT_URL": "https://store-b.test",
    })

    resumed = object.__new__(BrevSDK)
    resumed._artifact_credentials = {}
    resumed._store = MagicMock()
    first_fs = MagicMock()
    second_fs = MagicMock()
    first_fs.exists.return_value = False
    second_fs.exists.return_value = False
    resumed._get_fs = MagicMock(side_effect=[first_fs, second_fs])
    rows = {
        "job-a": {
            "job_id": "job-a",
            "status": "Complete",
            "results_dir": "s3://bucket-a/results/job-a",
            "specs": json.dumps(artifact_specs(
                "s3",
                "s3://bucket-a/results/job-a",
                endpoint_url="HTTPS://store-a.test/",
                credential_fingerprint=s3_credential_fingerprint(access_key),
            )),
        },
        "job-b": {
            "job_id": "job-b",
            "status": "Complete",
            "results_dir": "s3://bucket-b/results/job-b",
            "specs": json.dumps(artifact_specs(
                "s3",
                "s3://bucket-b/results/job-b",
                endpoint_url="https://STORE-B.TEST:443/",
                credential_fingerprint=s3_credential_fingerprint(access_key),
            )),
        },
    }
    resumed._store.get_job.side_effect = rows.get
    for key in (
        "ACCESS_KEY", "SECRET_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    assert resumed.delete_job_artifacts("job-a")
    assert resumed.delete_job_artifacts("job-b")
    assert resumed._get_fs.call_args_list == [
        call(
            "HTTPS://store-a.test/",
            access_key=access_key,
            secret_key="secret-a",
        ),
        call(
            "https://STORE-B.TEST:443/",
            access_key=access_key,
            secret_key="secret-b",
        ),
    ]


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_retention_preflight_rejects_ephemeral_outputs(
    sdk_path, class_name, monkeypatch
):
    import importlib

    for key in ("S3_BUCKET_NAME", "ACCESS_KEY", "SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    if class_name == "DockerSDK":
        sdk._handler = MagicMock()
        sdk._handler._build_volumes.return_value = None

    with pytest.raises(ValueError, match="durable reclaimable"):
        sdk.validate_artifact_retention()


def test_docker_retention_preflight_accepts_only_invoking_user_local_bind(
    tmp_path, monkeypatch
):
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = ""
    handler._client = MagicMock()
    handler._client.info.return_value = {"SecurityOptions": []}
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    mounts = [{"host_path": str(tmp_path), "container_path": "/results"}]
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getuid", lambda: 1004)
    monkeypatch.setattr("tao_sdk.platforms.docker.handler.os.getgid", lambda: 1025)

    assert sdk.validate_artifact_retention(mounts=mounts)
    assert sdk.validate_artifact_retention(
        mounts=mounts,
        container_user="1004:1025",
    )
    with pytest.raises(ValueError, match="run_as_user=False"):
        sdk.validate_artifact_retention(mounts=mounts, run_as_user=False)
    with pytest.raises(ValueError, match="must exactly match"):
        sdk.validate_artifact_retention(
            mounts=mounts,
            container_user="2000:3000",
        )


def test_docker_retention_preflight_rejects_unmanaged_result_routes(tmp_path):
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = ""
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    with pytest.raises(ValueError, match="named volume"):
        sdk.validate_artifact_retention(
            mounts=[{"source": "named-results", "target": "/results"}],
        )

    handler.docker_host = "tcp://remote.example:2376"
    with pytest.raises(ValueError, match="remote bind"):
        sdk.validate_artifact_retention(
            mounts=[{
                "host_path": str(tmp_path),
                "container_path": "/results",
            }],
        )


def test_kubernetes_retention_preflight_rejects_untracked_results_volume():
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

    sdk = object.__new__(KubernetesSDK)
    with pytest.raises(ValueError, match="Kubernetes /results volume"):
        sdk.validate_artifact_retention(
            volume_mounts=[{"name": "results", "mountPath": "/results"}],
        )


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_s3_cleanup_refuses_a_different_or_missing_principal(
    sdk_path, class_name, monkeypatch
):
    import importlib

    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    sdk._get_fs = MagicMock()
    sdk._store = MagicMock()
    backend_identity = ""
    if class_name in {"DockerSDK", "KubernetesSDK"}:
        sdk._handler = MagicMock()
        sdk._handler.get_backend_identity.return_value = "backend-1"
        backend_identity = "backend-1"
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "status": "Complete",
        "results_dir": "s3://bucket/results/job-1",
        "local_results_dir": "",
        "specs": json.dumps(artifact_specs(
            "s3",
            "s3://bucket/results/job-1",
            endpoint_url="https://minio.test",
            credential_fingerprint=s3_credential_fingerprint("original-owner"),
            backend_identity=backend_identity,
        )),
    }
    monkeypatch.setenv("ACCESS_KEY", "different-owner")
    monkeypatch.setenv("SECRET_KEY", "different-secret")

    assert not sdk.delete_job_artifacts("job-1")
    sdk._get_fs.assert_not_called()
    sdk._store.delete_checkpoints.assert_not_called()


def test_slurm_sdk_rejects_a_mismatched_recorded_path():
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    sdk = object.__new__(SlurmSDK)
    handler = SlurmHandler.__new__(SlurmHandler)
    handler.base_results_dir = "/lustre/users/test"
    handler.get_backend_identity = MagicMock(return_value="slurm-1")
    handler._ssh = MagicMock(
        return_value=SimpleNamespace(
            returncode=1, stdout="", stderr="recorded root mismatch",
        )
    )
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "slurm",
        "status": "Complete",
        "results_dir": "lustre:///lustre/other/results/job-1",
        "specs": json.dumps(artifact_specs(
            "lustre",
            "lustre:///lustre/other/results/job-1",
            backend_identity="slurm-1",
        )),
    }

    assert not sdk.delete_job_artifacts("job-1")
    assert handler._ssh.call_count == 1


def test_virtualenv_sdk_rejects_a_tampered_results_path(tmp_path):
    from tao_sdk.platforms.virtualenv.sdk import VirtualEnvSDK

    sdk = object.__new__(VirtualEnvSDK)
    sdk.jobs_dir = tmp_path / "jobs"
    sdk._lock = threading.RLock()
    sdk._store = MagicMock()
    outside = tmp_path / "outside"
    outside.mkdir()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "local_results_dir": str(outside),
    }

    assert not sdk.delete_job_artifacts("job-1")
    assert outside.is_dir()


def test_slurm_handler_deletes_only_scoped_job_results():
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.base_results_dir = "/lustre/users/test"
    handler._ssh = MagicMock(
        return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
    )

    assert handler.delete_job_artifacts(
        "job-1", expected_root="/lustre/users/test/results"
    )
    command = handler._ssh.call_args.args[0]
    assert "realpath -m -- /lustre/users/test/results" in command
    assert "cd -P -- /lustre/users/test/results" in command
    assert '"$(pwd -P)" = /lustre/users/test/results' in command
    assert command.endswith("rm -rf -- ./job-1")

    handler._ssh.reset_mock()
    assert not handler.delete_job_artifacts(
        "../job-1", expected_root="/lustre/users/test/results"
    )
    handler._ssh.assert_not_called()


def test_remote_job_path_rejects_parent_traversal():
    with pytest.raises(ValueError, match="parent traversal"):
        job_scoped_remote_path(
            "s3://bucket/safe/../results/job-1", "job-1", "s3"
        )


def test_local_delete_requires_exact_trusted_path(tmp_path):
    target = tmp_path / "job-1"
    target.mkdir()
    outside = tmp_path / "outside" / "job-1"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="does not match"):
        delete_local_job_results(
            outside, "job-1", expected_path=target
        )
    assert outside.is_dir()


def test_docker_refuses_artifact_deletion_for_running_job(tmp_path):
    from tao_sdk.platforms.docker.sdk import DockerSDK

    sdk = object.__new__(DockerSDK)
    sdk._store = MagicMock()
    target = tmp_path / "job-1"
    target.mkdir()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "docker",
        "status": "Running",
        "results_dir": str(target),
        "local_results_dir": str(target),
        "specs": json.dumps(artifact_specs("local_bind", str(tmp_path))),
    }

    assert not sdk.delete_job_artifacts("job-1")
    assert target.is_dir()


def test_docker_preserves_named_results_volume_with_unsupported_cleanup():
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = ""
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler

    mounts = [{"source": "named-results", "target": "/results"}]
    assert sdk._artifact_route("job-1", mounts) == (
        "named_volume",
        "docker-volume://named-results/job-1",
    )
    assert sdk._results_root_prefix(mounts) == "/results"


def test_docker_remote_bind_is_not_misidentified_as_s3():
    from tao_sdk.platforms.docker.handler import DockerHandler
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = DockerHandler.__new__(DockerHandler)
    handler.docker_host = "tcp://remote.example:2376"
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    kind, path = sdk._artifact_route(
        "job-1", [{"host_path": "/remote/results", "container_path": "/results"}]
    )

    assert kind == "remote_bind"
    assert path == "docker-bind:///remote/results/job-1"


def test_docker_submission_persists_effective_per_job_storage(monkeypatch):
    from tao_sdk.platforms.docker.sdk import DockerSDK

    handler = MagicMock()
    handler._build_volumes.return_value = None
    handler._has_writable_results_bind.return_value = False
    handler.get_job_name.return_value = "tao-job-job-1"
    sdk = object.__new__(DockerSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    monkeypatch.setenv("S3_BUCKET_NAME", "trusted-bucket")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://trusted.example")
    monkeypatch.setenv("ACCESS_KEY", "trusted-access")
    monkeypatch.setenv("SECRET_KEY", "trusted-secret")

    with patch("tao_sdk.platforms.docker.sdk.uuid.uuid4", return_value="job-1"):
        sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
            env_vars={
                "S3_BUCKET_NAME": "wrong-bucket",
                "AWS_ENDPOINT_URL": "https://wrong.example",
                "AWS_ACCESS_KEY_ID": "wrong-access",
                "AWS_SECRET_ACCESS_KEY": "wrong-secret",
            },
        )

    submitted = handler.create_job.call_args.kwargs
    assert submitted["docker_env_vars"]["S3_BUCKET_NAME"] == "wrong-bucket"
    assert submitted["docker_env_vars"]["AWS_ENDPOINT_URL"] == "https://wrong.example"
    assert submitted["docker_env_vars"]["AWS_ACCESS_KEY_ID"] == "wrong-access"
    assert submitted["docker_env_vars"]["AWS_SECRET_ACCESS_KEY"] == "wrong-secret"
    final_store = sdk._store.create_job.call_args.kwargs
    assert final_store["results_dir"] == "s3://wrong-bucket/results/job-1"
    assert final_store["specs"]["_tao_artifacts"]["endpoint_url"] == "https://wrong.example"
    assert final_store["specs"]["_tao_artifacts"]["credential_fingerprint"] == (
        s3_credential_fingerprint("wrong-access")
    )
    assert "wrong-secret" not in json.dumps(final_store["specs"])


def test_kubernetes_cancel_waits_until_writer_pods_are_gone(monkeypatch):
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler

    handler = KubernetesHandler.__new__(KubernetesHandler)
    handler._client_module = SimpleNamespace(
        V1Preconditions=lambda **kwargs: kwargs,
        V1DeleteOptions=lambda **kwargs: kwargs,
    )
    handler._batch = MagicMock()
    handler._core = MagicMock()
    handler._jobs = {
        "job-1": {
            "job_id": "job-1",
            "job_name": "tao-job-job-1",
            "job_uid": "uid-1",
            "namespace": "default",
            "num_nodes": 1,
            "service_name": "",
            "service_uid": "",
            "launch_uncertain": False,
        }
    }
    live_job = SimpleNamespace(
        metadata=SimpleNamespace(
            uid="uid-1", labels={"tao-sdk-job-id": "job-1"},
        ),
        status=SimpleNamespace(
            succeeded=0, failed=0, active=1, conditions=[],
        ),
    )
    not_found = RuntimeError("not found")
    not_found.status = 404
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        live_job,
        not_found,
    ]
    owned_pod = SimpleNamespace(metadata=SimpleNamespace(
        labels={
            "job-name": "tao-job-job-1",
            "tao-sdk-job-id": "job-1",
        },
        owner_references=[SimpleNamespace(
            uid="uid-1",
            kind="Job",
            name="tao-job-job-1",
            controller=True,
        )],
    ))
    handler._core.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[owned_pod]),
        SimpleNamespace(items=[]),
    ]
    monkeypatch.setattr("tao_sdk.platforms.kubernetes.handler.time.sleep", lambda _: None)

    assert handler.cancel_job("job-1")
    assert handler._core.list_namespaced_pod.call_count == 2


def test_slurm_cancel_waits_for_terminal_scheduler_state(monkeypatch):
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.login_hostnames = ["login-1"]
    handler.login_hostname = "login-1"
    handler.current_hostname_index = 0
    handler._get_job_metadata = MagicMock(
        return_value={"slurm_job_id": "42", "hostname": "login-1"}
    )
    handler.get_slurm_job_status = MagicMock(
        side_effect=["RUNNING", "RUNNING", "CANCELLED"]
    )
    handler._ssh = MagicMock(
        return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    handler._set_job_message = MagicMock()
    monkeypatch.setattr("tao_sdk.platforms.slurm.handler.time.sleep", lambda _: None)

    assert handler.cancel_job("job-1")
    assert handler.get_slurm_job_status.call_count == 3


def test_slurm_backend_identity_uses_cluster_name_and_ignores_host_order():
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    first = SlurmHandler.__new__(SlurmHandler)
    first.login_user = "scientist"
    first.login_hostnames = ["login-b.example", "LOGIN-A.EXAMPLE."]
    first._ssh = MagicMock(return_value=SimpleNamespace(
        returncode=0,
        stdout="ClusterName = tao-cluster\n",
        stderr="",
    ))
    second = SlurmHandler.__new__(SlurmHandler)
    second.login_user = "scientist"
    second.login_hostnames = ["login-a.example", "login-b.example"]
    second._ssh = first._ssh

    assert first.get_backend_identity() == second.get_backend_identity()

    # The identity is cached for the life of a handler on a given login host
    # (per-poll refetching stalled monitoring on flaky networks), so a changed
    # ClusterName is observed by a new handler, not by an existing one.
    third = SlurmHandler.__new__(SlurmHandler)
    third.login_user = "scientist"
    third.login_hostnames = ["login-a.example", "login-b.example"]
    third._ssh = MagicMock(return_value=SimpleNamespace(
        returncode=0,
        stdout="ClusterName = replacement-cluster\n",
        stderr="",
    ))
    assert first.get_backend_identity() != third.get_backend_identity()


def test_slurm_backend_identity_falls_back_to_accounting_cluster_record():
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.login_user = "scientist"
    handler._ssh = MagicMock(side_effect=[
        SimpleNamespace(returncode=0, stdout="", stderr=""),
        SimpleNamespace(returncode=0, stdout="tao-cluster\n", stderr=""),
    ])

    identity = handler.get_backend_identity()

    assert identity.startswith("slurm-sha256:")
    assert handler._ssh.call_args_list == [
        call("scontrol show config", check=False, timeout=30),
        call(
            "sacctmgr -nP show cluster format=Cluster",
            check=False,
            timeout=30,
        ),
    ]


def test_slurm_backend_identity_rejects_ambiguous_accounting_clusters():
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.login_user = "scientist"
    handler._ssh = MagicMock(side_effect=[
        SimpleNamespace(returncode=1, stdout="", stderr="unavailable"),
        SimpleNamespace(
            returncode=0,
            stdout="tao-cluster-a\ntao-cluster-b\n",
            stderr="",
        ),
    ])

    assert handler.get_backend_identity() == ""


def _slurm_sdk_with_create_error(error):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = MagicMock()
    handler.base_results_dir = "/lustre/results"
    handler.get_canonical_results_root.return_value = "/lustre/results"
    handler.get_backend_identity.return_value = "slurm-1"
    handler.get_job_name.return_value = "tao-job-job-1"
    handler.resolve_resource_shape.return_value = ("gpu.1", None)
    handler.create_job.side_effect = error
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    return sdk


def test_slurm_sdk_surfaces_definitive_scheduler_rejection():
    error = RuntimeError("bad account")
    error.writer_quiescent = True
    sdk = _slurm_sdk_with_create_error(error)

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(RuntimeError, match="bad account"):
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=1,
            )

    sdk._handler.cancel_job.assert_not_called()
    sdk._store.update_job_status.assert_called_once_with("job-1", "Error")


def test_slurm_sdk_preserves_transport_ambiguous_submission():
    sdk = _slurm_sdk_with_create_error(TimeoutError("ssh timed out"))
    sdk._handler.cancel_job.return_value = False

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        job = sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=1,
        )

    assert job.id == "job-1"
    assert sdk._jobs["job-1"] is job
    sdk._store.update_job_status.assert_not_called()


def test_brev_cancel_stops_exact_container_before_success():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=0, stdout="STOPPED_OBSERVED\n", stderr=""
    )
    handler = BrevHandler(cli=cli)
    handler._job_instances["job-1"] = "instance-1"
    handler._job_pids["job-1"] = "1234"
    handler._job_container_ids["job-1"] = "a" * 64

    assert handler.cancel_job("job-1")
    command = cli.exec.call_args.args[1]
    assert 'docker stop --time 10 "$expected_id"' in command
    assert 'docker rm -f "$expected_id"' in command
    assert "docker stop --time 10 tao-job-job-1" not in command
    assert "kill 1234" in command
    assert "job-1" not in handler._job_instances


def test_brev_cancel_does_not_treat_daemon_failure_as_stopped():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=1, stdout="", stderr="UNKNOWN\n"
    )
    handler = BrevHandler(cli=cli)
    handler._job_instances["job-1"] = "instance-1"

    assert not handler.cancel_job("job-1")
    assert handler._job_instances["job-1"] == "instance-1"


def test_brev_status_transport_failure_remains_nonterminal():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=1, stdout="", stderr="ssh unavailable"
    )
    handler = BrevHandler(cli=cli)
    handler._job_instances["job-1"] = "instance-1"

    assert handler.get_tao_job_status("job-1") == "Running"


@pytest.mark.parametrize(
    ("marker", "expected"),
    (
        ("CONTAINER_DONE", "Complete"),
        ("CONTAINER_FAILED", "Error"),
        ("EXIT_DONE", "Complete"),
        ("EXIT_FAILED:19", "Error"),
    ),
)
def test_brev_status_preserves_natural_container_exit(marker, expected):
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=0, stdout=f"{marker}\n", stderr=""
    )
    handler = BrevHandler(cli=cli)
    handler._job_instances["job-1"] = "instance-1"

    raw = handler.get_tao_job_status("job-1")

    assert handler.normalize_status(raw) == expected


def test_brev_log_only_disappearance_is_not_assumed_successful():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=0, stdout="LOG_ONLY\n", stderr=""
    )
    handler = BrevHandler(cli=cli)
    handler._job_instances["job-1"] = "instance-1"
    handler._job_pids["job-1"] = "1234"

    assert handler.get_tao_job_status("job-1") == "Pending"


@pytest.mark.parametrize(
    ("marker", "expected"),
    (("COMPLETE_RECORDED", "Complete"), ("ERROR_RECORDED", "Error")),
)
def test_brev_cancel_preserves_recorded_auto_removed_exit(marker, expected):
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.return_value = SimpleNamespace(
        returncode=0, stdout=f"{marker}\n", stderr=""
    )
    handler = BrevHandler(cli=cli)
    handler._job_instances["job-1"] = "instance-1"

    assert handler.cancel_job("job-1")
    assert handler.normalize_status(
        handler.get_tao_job_status("job-1")
    ) == expected


def test_slurm_cancel_requires_durable_scheduler_identity():
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.login_hostnames = ["login-1"]
    handler.login_hostname = "login-1"
    handler.current_hostname_index = 0
    handler._set_job_message = MagicMock()
    handler._get_job_metadata = MagicMock(return_value=None)
    assert not handler.cancel_job("job-1")

    handler._get_job_metadata.return_value = {
        "slurm_job_id": "42",
        "hostname": "login-1",
    }
    handler.get_slurm_job_status = MagicMock(return_value="NOT_FOUND")
    assert not handler.cancel_job("job-1")


def test_docker_status_transport_failure_remains_nonterminal():
    from tao_sdk.platforms.docker.handler import DockerHandler

    handler = DockerHandler.__new__(DockerHandler)
    handler._docker = SimpleNamespace(
        errors=SimpleNamespace(NotFound=type("NotFound", (Exception,), {}))
    )
    handler._jobs = {"job-1": {"status": "Running"}}
    handler._job_messages = {}
    handler._get_container = MagicMock(side_effect=RuntimeError("daemon unavailable"))

    assert handler.get_tao_job_status("job-1") == "Running"
    assert handler._jobs["job-1"]["status"] == "Running"


@pytest.mark.parametrize(
    "sdk_path,class_name",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK"),
    ),
)
def test_recorded_default_s3_route_ignores_later_ambient_endpoint(
    sdk_path, class_name, monkeypatch
):
    import importlib

    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://changed.example")
    with patch("fsspec.filesystem") as filesystem:
        sdk._get_fs("")

    options = filesystem.call_args.kwargs
    assert "client_kwargs" not in options
    assert options["config_kwargs"] == {
        "ignore_configured_endpoint_urls": True
    }


def test_kubernetes_multinode_partial_completion_is_nonterminal():
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler

    handler = KubernetesHandler.__new__(KubernetesHandler)
    handler._jobs = {
        "job-1": {
            "job_name": "tao-job-job-1",
            "namespace": "default",
            "num_nodes": 3,
        }
    }
    handler._batch = MagicMock()
    handler._core = MagicMock()
    status = SimpleNamespace(
        succeeded=1,
        failed=0,
        active=2,
        conditions=[],
    )
    handler._batch.read_namespaced_job.return_value = SimpleNamespace(status=status)

    assert handler.get_tao_job_status("job-1") == "Running"

    status.succeeded = 0
    status.failed = 1
    assert handler.get_tao_job_status("job-1") == "Running"


def test_kubernetes_terminal_condition_waits_for_pod_quiescence():
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler

    handler = KubernetesHandler.__new__(KubernetesHandler)
    handler._jobs = {
        "job-1": {
            "job_name": "tao-job-job-1",
            "namespace": "default",
            "num_nodes": 2,
        }
    }
    handler._batch = MagicMock()
    handler._core = MagicMock()
    handler._batch.read_namespaced_job.return_value = SimpleNamespace(
        status=SimpleNamespace(
            succeeded=1,
            failed=0,
            active=1,
            conditions=[SimpleNamespace(type="Complete", status="True")],
        )
    )
    handler._core.list_namespaced_pod.side_effect = [
        SimpleNamespace(
            items=[SimpleNamespace(status=SimpleNamespace(phase="Running"))]
        ),
        SimpleNamespace(
            items=[SimpleNamespace(status=SimpleNamespace(phase="Succeeded"))]
        ),
    ]

    assert handler.get_tao_job_status("job-1") == "Running"
    assert handler.get_tao_job_status("job-1") == "Complete"


def test_kubernetes_runtime_is_restored_after_sdk_restart():
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

    handler = KubernetesHandler.__new__(KubernetesHandler)
    handler._namespace = "default"
    handler._jobs = {}
    handler._client_module = SimpleNamespace(
        V1Preconditions=lambda **kwargs: kwargs,
        V1DeleteOptions=lambda **kwargs: kwargs,
    )
    handler._batch = MagicMock()
    handler._core = MagicMock()
    handler.get_backend_identity = MagicMock(return_value="cluster-1")
    live_job = SimpleNamespace(
        metadata=SimpleNamespace(
            uid="uid-1",
            labels={"tao-sdk-job-id": "job-1"},
        ),
        status=SimpleNamespace(
            succeeded=0,
            failed=0,
            active=1,
            conditions=[],
        ),
    )
    not_found = RuntimeError("not found")
    not_found.status = 404
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        live_job,
        live_job,
        live_job,
        live_job,
        not_found,
    ]
    runtime = handler.get_runtime_identity(
        "job-1", namespace="training", num_nodes=1, job_uid="uid-1"
    )

    sdk = object.__new__(KubernetesSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "kubernetes",
        "status": "Running",
        "backend_job_id": "tao-job-job-1",
        "results_dir": "s3://bucket/results/job-1",
        "specs": json.dumps(artifact_specs(
            "s3",
            "s3://bucket/results/job-1",
            backend_identity="cluster-1",
            runtime_identity=runtime,
        )),
    }

    assert sdk.get_job_status("job-1").status == "Running"
    assert handler._jobs["job-1"]["namespace"] == "training"
    assert handler._jobs["job-1"]["job_uid"] == "uid-1"

    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    assert sdk.cancel_job("job-1")
    handler._batch.delete_namespaced_job.assert_called_once_with(
        name="tao-job-job-1",
        namespace="training",
        body={
            "propagation_policy": "Foreground",
            "preconditions": {"uid": "uid-1"},
        },
    )


def test_kubernetes_ordinary_job_404_remains_ambiguous():
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler

    handler = KubernetesHandler.__new__(KubernetesHandler)
    handler._jobs = {
        "job-1": {
            "job_id": "job-1",
            "job_name": "tao-job-job-1",
            "namespace": "default",
            "num_nodes": 1,
        }
    }
    handler._batch = MagicMock()
    handler._core = MagicMock()
    not_found = RuntimeError("not found")
    not_found.status = 404
    handler._batch.read_namespaced_job.side_effect = not_found

    assert handler.get_tao_job_status("job-1") == "Pending"
    handler._core.list_namespaced_pod.assert_not_called()


def _kubernetes_sdk_with_create_error(error, *, cancel_result=False):
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

    runtime = {
        "job_name": "tao-job-job-1",
        "job_uid": "",
        "namespace": "default",
        "num_nodes": 1,
        "service_name": "",
        "service_uid": "",
        "launch_uncertain": True,
    }
    handler = MagicMock()
    handler.get_backend_identity.return_value = "cluster-1"
    handler.get_runtime_identity.return_value = runtime
    handler.get_job_name.return_value = "tao-job-job-1"
    handler.create_job.side_effect = error
    handler.restore_job_runtime.return_value = True
    handler.cancel_job.return_value = cancel_result
    sdk = object.__new__(KubernetesSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    return sdk


def test_kubernetes_sdk_persists_transport_ambiguous_create():
    sdk = _kubernetes_sdk_with_create_error(TimeoutError("API timed out"))

    with patch(
        "tao_sdk.platforms.kubernetes.sdk.uuid.uuid4", return_value="job-1"
    ):
        job = sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=0,
        )

    assert job.id == "job-1"
    persisted = sdk._store.update_job_specs.call_args.args[1]
    assert persisted["_tao_artifacts"]["runtime_identity"]["launch_uncertain"] is True
    sdk._store.update_job_status.assert_not_called()


def test_kubernetes_sdk_surfaces_definitive_create_rejection():
    rejection = RuntimeError("invalid resource")
    rejection.writer_quiescent = True
    sdk = _kubernetes_sdk_with_create_error(rejection)

    with patch(
        "tao_sdk.platforms.kubernetes.sdk.uuid.uuid4", return_value="job-1"
    ):
        with pytest.raises(RuntimeError, match="invalid resource"):
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    sdk._handler.cancel_job.assert_not_called()
    sdk._store.update_job_status.assert_called_once_with("job-1", "Error")


def test_kubernetes_interrupted_confirmed_rollback_is_terminal_and_identified():
    interrupt = KeyboardInterrupt("interrupted after create")
    sdk = _kubernetes_sdk_with_create_error(interrupt, cancel_result=True)

    with patch(
        "tao_sdk.platforms.kubernetes.sdk.uuid.uuid4", return_value="job-1"
    ):
        with pytest.raises(KeyboardInterrupt) as caught:
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=0,
            )

    assert caught.value.tao_job_id == "job-1"
    assert caught.value.tao_launch_uncertain is False
    sdk._store.update_job_status.assert_called_once_with("job-1", "Canceled")


def test_kubernetes_confirmed_cancellation_survives_backend_deletion():
    from tao_sdk.models import Job
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

    sdk = object.__new__(KubernetesSDK)
    sdk._handler = MagicMock()
    sdk._handler.get_tao_job_status.side_effect = AssertionError(
        "deleted backend must not overwrite a durable cancellation"
    )
    sdk._store = MagicMock()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "kubernetes",
        "status": "Canceled",
    }
    sdk._jobs = {
        "job-1": Job(
            id="job-1",
            network_arch="",
            action="",
            workspace_id="kubernetes",
            status="Canceled",
        )
    }
    sdk._backend_identity_matches = MagicMock(return_value=True)

    assert sdk.get_job_status("job-1").status == "Canceled"
    sdk._handler.get_tao_job_status.assert_not_called()


def test_brev_missing_runtime_identity_stays_nonterminal():
    from tao_sdk.platforms.brev.handler import BrevHandler

    handler = BrevHandler(cli=MagicMock())
    assert handler.get_tao_job_status("job-1") == "Pending"


def test_brev_launch_requires_success_and_numeric_launcher_pid():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.get_instance.return_value = object()
    cli.exec.side_effect = [
        SimpleNamespace(returncode=1, stdout="", stderr="remote exec failed"),
        SimpleNamespace(returncode=0, stdout="STOPPED_OBSERVED\n", stderr=""),
    ]
    handler = BrevHandler(cli=cli)

    with pytest.raises(RuntimeError, match="valid launcher PID"):
        handler.create_job(
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
            entrypoint={"command": "docker run --name tao-job-job-1 image"},
            instance_id="instance-1",
        )

    assert "job-1" not in handler._job_instances


def test_brev_prelaunch_provisioning_failure_is_definitively_quiescent():
    from tao_sdk.platforms.brev.cli import BrevCLIError
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.get_instance.return_value = None
    cli.create_instance.side_effect = BrevCLIError("quota rejected")
    handler = BrevHandler(cli=cli)

    with pytest.raises(BrevCLIError, match="quota rejected") as raised:
        handler.create_job(
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
            entrypoint={"command": "docker run --name tao-job-job-1 image"},
            instance_id="instance-1",
        )

    assert raised.value.writer_quiescent is True
    cli.exec.assert_not_called()


def test_brev_transport_ambiguous_absence_returns_recoverable_identity():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.get_instance.return_value = object()
    cli.exec.side_effect = [
        TimeoutError("remote exec timed out"),
        SimpleNamespace(returncode=0, stdout="ABSENT\n", stderr=""),
    ]
    handler = BrevHandler(cli=cli)

    result = handler.create_job(
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
        entrypoint={"command": "docker run --name tao-job-job-1 image"},
        instance_id="instance-1",
    )

    assert result["launch_uncertain"] is True
    assert handler._job_instances["job-1"] == "instance-1"
    assert "job-1" in handler._job_launch_uncertain


def test_brev_uncertain_log_is_not_terminal_and_absence_cannot_cancel():
    from tao_sdk.platforms.brev.handler import BrevHandler

    cli = MagicMock()
    cli.exec.side_effect = [
        SimpleNamespace(returncode=0, stdout="LOG_ONLY\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="ABSENT\n", stderr=""),
    ]
    handler = BrevHandler(cli=cli)
    assert handler.restore_job_runtime(
        "job-1",
        {
            "instance_id": "instance-1",
            "pid": "",
            "launch_uncertain": True,
        },
    )

    assert handler.get_tao_job_status("job-1") == "Pending"
    assert not handler.cancel_job("job-1")
    assert "job-1" in handler._job_launch_uncertain


def test_brev_sdk_persists_ambiguous_runtime_identity():
    from tao_sdk.platforms.brev.sdk import BrevSDK

    handler = MagicMock()
    handler.get_container_name.return_value = "tao-job-job-1"
    handler.get_job_name.return_value = "job-1"
    handler.create_job.return_value = {
        "instance_id": "tao-job-1",
        "pid": "",
        "launch_uncertain": True,
    }
    sdk = object.__new__(BrevSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    with patch("tao_sdk.platforms.brev.sdk.uuid.uuid4", return_value="job-1"):
        job = sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=1,
        )

    assert job.id == "job-1"
    persisted = sdk._store.update_job_specs.call_args.args[1]
    assert persisted["_brev_runtime"] == {
        "instance_id": "tao-job-1",
        "pid": "",
        "launch_uncertain": True,
    }
    command = handler.create_job.call_args.kwargs["entrypoint"]["command"]
    assert "tao_exit_code=$?" in command
    assert "/tmp/tao-job-job-1.exit" in command
    assert "mv -f" in command


def test_brev_interrupted_confirmed_rollback_is_terminal_and_identified():
    from tao_sdk.platforms.brev.sdk import BrevSDK

    interrupt = KeyboardInterrupt("interrupted after create")
    handler = MagicMock()
    handler.get_container_name.return_value = "tao-job-job-1"
    handler.get_job_name.return_value = "job-1"
    handler.create_job.side_effect = interrupt
    handler.cancel_job.return_value = True
    handler._job_instances = {}
    sdk = object.__new__(BrevSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    with patch("tao_sdk.platforms.brev.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(KeyboardInterrupt) as caught:
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=1,
            )

    assert caught.value.tao_job_id == "job-1"
    assert caught.value.tao_launch_uncertain is False
    sdk._store.update_job_status.assert_called_once_with("job-1", "Canceled")


def test_brev_confirmed_cancellation_survives_backend_deletion():
    from tao_sdk.models import Job
    from tao_sdk.platforms.brev.sdk import BrevSDK

    sdk = object.__new__(BrevSDK)
    sdk._handler = MagicMock()
    sdk._handler.get_tao_job_status.side_effect = AssertionError(
        "deleted backend must not overwrite a durable cancellation"
    )
    sdk._store = MagicMock()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "brev",
        "status": "Canceled",
    }
    sdk._jobs = {
        "job-1": Job(
            id="job-1",
            network_arch="",
            action="",
            workspace_id="",
            status="Canceled",
        )
    }

    assert sdk.get_job_status("job-1").status == "Canceled"
    sdk._handler.get_tao_job_status.assert_not_called()


def test_slurm_interrupted_confirmed_rollback_is_terminal_and_identified():
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    interrupt = KeyboardInterrupt("interrupted after create")
    handler = MagicMock()
    handler.get_canonical_results_root.return_value = "/lustre/results"
    handler.get_backend_identity.return_value = "cluster-1"
    handler.get_job_name.return_value = "slurm-job-1"
    handler.resolve_resource_shape.return_value = (None, None)
    handler.create_job.side_effect = interrupt
    handler.cancel_job.return_value = True
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(KeyboardInterrupt) as caught:
            sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=1,
            )

    assert caught.value.tao_job_id == "job-1"
    assert caught.value.tao_launch_uncertain is False
    sdk._store.update_job_status.assert_called_once_with("job-1", "Canceled")


@pytest.mark.parametrize(
    "sdk_path,class_name,workspace_id",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK", "local"),
        ("tao_sdk.platforms.slurm.sdk", "SlurmSDK", "slurm"),
    ),
)
def test_confirmed_cancellation_is_not_overwritten_by_missing_backend(
    sdk_path, class_name, workspace_id,
):
    import importlib

    from tao_sdk.models import Job

    sdk_class = getattr(importlib.import_module(sdk_path), class_name)
    sdk = object.__new__(sdk_class)
    sdk._handler = MagicMock()
    sdk._handler.get_tao_job_status.side_effect = AssertionError(
        "deleted backend must not overwrite a durable cancellation"
    )
    sdk._store = MagicMock()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "docker" if workspace_id == "local" else "slurm",
        "status": "Canceled",
    }
    sdk._jobs = {
        "job-1": Job(
            id="job-1",
            network_arch="",
            action="",
            workspace_id=workspace_id,
            status="Canceled",
        )
    }
    sdk._backend_identity_matches = MagicMock(return_value=True)

    assert sdk.get_job_status("job-1").status == "Canceled"
    sdk._handler.get_tao_job_status.assert_not_called()


def test_slurm_cleanup_rejects_changed_canonical_results_root():
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.base_results_dir = "/configured/base"
    handler._ssh = MagicMock(
        return_value=SimpleNamespace(
            returncode=1, stdout="", stderr="recorded root mismatch",
        )
    )

    assert not handler.delete_job_artifacts(
        "job-1", expected_root="/original/target"
    )
    assert handler._ssh.call_count == 1
