# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Kubernetes platform handler.

Mocks the kubernetes Python client end-to-end. No real cluster required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _patch_k8s_client_module():
    """Build a fake kubernetes.client module with the V1* classes used by the handler."""
    fake_client = SimpleNamespace(
        V1EnvVar=lambda name, value: {"name": name, "value": value},
        V1ResourceRequirements=lambda limits: {"limits": limits},
        V1Container=lambda **kw: kw,
        V1LocalObjectReference=lambda name: {"name": name},
        V1PodSpec=lambda **kw: kw,
        V1PodTemplateSpec=lambda **kw: kw,
        V1JobSpec=lambda **kw: kw,
        V1ObjectMeta=lambda **kw: kw,
        V1Job=lambda **kw: kw,
        V1Service=lambda **kw: kw,
        V1ServiceSpec=lambda **kw: kw,
        V1Preconditions=lambda **kw: kw,
        V1DeleteOptions=lambda **kw: kw,
        BatchV1Api=MagicMock,
        CoreV1Api=MagicMock,
    )
    return fake_client


def _mock_node(gpu_count: int):
    return SimpleNamespace(
        status=SimpleNamespace(allocatable={"nvidia.com/gpu": str(gpu_count)}),
    )


def _build_handler(gpu_capacity: int = 8, kubeconfig_works: bool = True):
    """Construct a KubernetesHandler with all I/O mocked out."""
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler
    from kubernetes.config.config_exception import ConfigException

    fake_client = _patch_k8s_client_module()
    fake_config = MagicMock()
    if not kubeconfig_works:
        fake_config.load_kube_config.side_effect = ConfigException("no kubeconfig")
        fake_config.load_incluster_config.side_effect = ConfigException("not in cluster")
    else:
        fake_config.load_kube_config.return_value = None

    with patch.object(KubernetesHandler, "_ping", lambda self: None):
        handler = KubernetesHandler.__new__(KubernetesHandler)
        handler._creds = {}
        handler._namespace = "default"
        handler._context = None
        handler._kubeconfig_path = None
        handler._client_module = fake_client
        handler._config_module = fake_config
        handler._config_exception = ConfigException
        handler._batch = MagicMock()
        handler._core = MagicMock()
        handler._jobs = {}
        handler._gpu_capacity_cache = gpu_capacity

    handler._core.list_node.return_value = SimpleNamespace(
        items=[_mock_node(gpu_capacity)]
    )
    return handler


def test_get_job_name_dns_safe():
    handler = _build_handler()
    name = handler.get_job_name("3F8F-2D24/abc")
    assert name == "tao-job-3f8f-2d24-abc"
    assert len(name) <= 63
    assert name.islower() or all(c.isdigit() or c in "-" for c in name if not c.isalpha())


def test_create_job_multinode_creates_indexed_job_and_service():
    handler = _build_handler(gpu_capacity=16)
    handler._core.create_namespaced_service.return_value = SimpleNamespace(
        metadata=SimpleNamespace(uid="service-uid-multi")
    )
    handler._batch.create_namespaced_job.return_value = SimpleNamespace(
        metadata=SimpleNamespace(uid="uid-multi")
    )
    handler.create_job(
        image="img:tag",
        network="dino", action="train",
        job_id="job-mn",
        entrypoint={"command": "torchrun --nnodes=$WORLD_SIZE my_train.py"},
        num_gpus=8,
        num_nodes=2,
    )

    # Headless Service was created first.
    svc_call = handler._core.create_namespaced_service.call_args
    assert svc_call is not None
    svc = svc_call.kwargs["body"]
    assert svc["spec"]["cluster_ip"] == "None"
    assert svc["spec"]["selector"] == {"job-name": "tao-job-job-mn"}
    assert svc["spec"]["publish_not_ready_addresses"] is True

    # Job was Indexed with parallelism = completions = num_nodes.
    job_call = handler._batch.create_namespaced_job.call_args
    body = job_call.kwargs["body"]
    spec = body["spec"]
    assert spec["parallelism"] == 2
    assert spec["completions"] == 2
    assert spec["completion_mode"] == "Indexed"
    assert "ttl_seconds_after_finished" not in spec

    # Pod template has subdomain set for stable DNS.
    pod_spec = spec["template"]["spec"]
    assert pod_spec["subdomain"] == "tao-job-job-mn"

    # Container command is wrapped with rendezvous env vars matching
    # the TAO container contract (WORLD_SIZE/NUM_GPU_PER_NODE) plus
    # PyTorch-standard synonyms (NNODES/NPROC_PER_NODE).
    container = pod_spec["containers"][0]
    wrapped_cmd = container["command"][2]
    assert "JOB_COMPLETION_INDEX" in wrapped_cmd
    assert "WORLD_SIZE=2" in wrapped_cmd                # TAO contract
    assert "NUM_GPU_PER_NODE=8" in wrapped_cmd           # TAO contract
    assert "NNODES=2" in wrapped_cmd                     # PyTorch-standard synonym
    assert "NPROC_PER_NODE=8" in wrapped_cmd             # PyTorch-standard synonym
    assert "MASTER_PORT=29500" in wrapped_cmd            # TAO default
    assert "MASTER_ADDR=tao-job-job-mn-0.tao-job-job-mn" in wrapped_cmd
    assert "torchrun --nnodes=$WORLD_SIZE my_train.py" in wrapped_cmd
    assert handler._jobs["job-mn"]["service_uid"] == "service-uid-multi"


def test_create_job_multinode_capacity_check_uses_total():
    """For 2 nodes × 4 GPUs the cluster needs 8 allocatable, not 4."""
    handler = _build_handler(gpu_capacity=4)
    with pytest.raises(RuntimeError, match="× 2 nodes = 8 total"):
        handler.create_job(
            image="img:tag", network="", action="",
            entrypoint={"command": "echo hi"},
            num_gpus=4, num_nodes=2,
        )


def _running_job(uid="uid-1", job_id="job-1"):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            uid=uid,
            labels={"tao-sdk-job-id": job_id},
        ),
        status=SimpleNamespace(
            succeeded=0,
            failed=0,
            active=1,
            conditions=[],
        ),
    )


def _owned_running_pod(uid="uid-1", job_id="job-1"):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            labels={
                "job-name": f"tao-job-{job_id}",
                "tao-sdk-job-id": job_id,
            },
            owner_references=[SimpleNamespace(
                uid=uid,
                kind="Job",
                name=f"tao-job-{job_id}",
                controller=True,
            )],
        ),
        status=SimpleNamespace(phase="Running"),
    )


def _cancel_runtime(
    handler, job_id="job-1", uid="uid-1", *, num_nodes=1,
    service_uid="",
):
    job_name = f"tao-job-{job_id}"
    handler._jobs[job_id] = {
        "job_id": job_id,
        "job_name": job_name,
        "job_uid": uid,
        "namespace": "default",
        "service_name": job_name if num_nodes > 1 else "",
        "service_uid": service_uid if num_nodes > 1 else "",
        "num_nodes": num_nodes,
        "launch_uncertain": False,
    }


def test_cancel_job_deletes_service_for_multinode():
    handler = _build_handler()
    _cancel_runtime(
        handler, "job-mn", "uid-mn", num_nodes=2,
        service_uid="service-uid-mn",
    )
    live_job = _running_job("uid-mn", "job-mn")
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    service = SimpleNamespace(metadata=SimpleNamespace(
        uid="service-uid-mn",
        labels={"tao-sdk-job-id": "job-mn"},
    ))
    handler._core.read_namespaced_service.side_effect = [
        service,
        _not_found_error(),
    ]

    assert handler.cancel_job("job-mn") is True
    handler._batch.delete_namespaced_job.assert_called_once_with(
        name="tao-job-job-mn",
        namespace="default",
        body={
            "propagation_policy": "Foreground",
            "preconditions": {"uid": "uid-mn"},
        },
    )
    handler._core.delete_namespaced_service.assert_called_once_with(
        name="tao-job-job-mn",
        namespace="default",
        body={"preconditions": {"uid": "service-uid-mn"}},
    )


def test_cancel_job_retains_service_when_job_delete_fails():
    handler = _build_handler()
    _cancel_runtime(
        handler, "job-mn", "uid-mn", num_nodes=2,
        service_uid="service-uid-mn",
    )
    live_job = _running_job("uid-mn", "job-mn")
    handler._batch.read_namespaced_job.side_effect = [live_job, live_job]
    handler._batch.delete_namespaced_job.side_effect = RuntimeError(
        "API unavailable"
    )

    assert handler.cancel_job("job-mn") is False
    handler._core.delete_namespaced_service.assert_not_called()


def test_cancel_job_retains_service_when_pods_do_not_terminate(monkeypatch):
    handler = _build_handler()
    _cancel_runtime(
        handler, "job-mn", "uid-mn", num_nodes=2,
        service_uid="service-uid-mn",
    )
    live_job = _running_job("uid-mn", "job-mn")
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(
        items=[_owned_running_pod("uid-mn", "job-mn")]
    )
    monkeypatch.setattr(
        "tao_sdk.platforms.kubernetes.handler.CANCEL_WAIT_SECONDS", 0
    )

    assert handler.cancel_job("job-mn") is False
    handler._core.delete_namespaced_service.assert_not_called()


def test_cancel_job_skips_service_for_single_pod():
    handler = _build_handler()
    _cancel_runtime(handler)
    live_job = _running_job()
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])

    assert handler.cancel_job("job-1") is True
    handler._core.delete_namespaced_service.assert_not_called()


def test_create_job_requires_entrypoint_command():
    handler = _build_handler()
    with pytest.raises(ValueError, match="entrypoint") as raised:
        handler.create_job(
            image="img:tag", network="", action="",
            entrypoint=None,
        )
    assert raised.value.writer_quiescent is True


def test_create_job_blocks_when_no_gpus_allocatable():
    handler = _build_handler(gpu_capacity=0)
    with pytest.raises(RuntimeError, match="NVIDIA GPU Operator"):
        handler.create_job(
            image="img:tag", network="", action="",
            entrypoint={"command": "echo hi"},
            num_gpus=1,
        )


def test_create_job_submits_job_manifest():
    handler = _build_handler(gpu_capacity=8)
    handler._batch.create_namespaced_job.return_value = SimpleNamespace(
        metadata=SimpleNamespace(uid="uid-123")
    )
    result = handler.create_job(
        image="nvcr.io/foo:tag",
        network="dino", action="train",
        job_id="job-1",
        docker_env_vars={"NGC_KEY": "fake"},
        entrypoint={"command": "echo hi"},
        num_gpus=2,
    )
    assert result["job_name"] == "tao-job-job-1"
    assert result["uid"] == "uid-123"

    call_args = handler._batch.create_namespaced_job.call_args
    body = call_args.kwargs["body"]
    container = body["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "nvcr.io/foo:tag"
    assert container["command"] == ["/bin/bash", "-c", "echo hi"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert {"name": "NGC_KEY", "value": "fake"} in container["env"]
    # Terminal Job state must remain queryable across SDK restarts; automatic
    # TTL deletion previously erased the only backend success/failure evidence.
    assert "ttl_seconds_after_finished" not in body["spec"]


def test_definitive_api_rejection_is_marked_quiescent():
    handler = _build_handler(gpu_capacity=8)
    rejection = RuntimeError("invalid job spec")
    rejection.status = 422
    handler._batch.create_namespaced_job.side_effect = rejection

    with pytest.raises(RuntimeError, match="invalid job spec") as raised:
        handler.create_job(
            image="img:tag",
            network="",
            action="",
            job_id="job-1",
            entrypoint={"command": "echo hi"},
        )

    assert raised.value.writer_quiescent is True


def test_transport_ambiguous_create_is_not_marked_quiescent():
    handler = _build_handler(gpu_capacity=8)
    handler._batch.create_namespaced_job.side_effect = TimeoutError("timed out")

    with pytest.raises(TimeoutError, match="timed out") as raised:
        handler.create_job(
            image="img:tag",
            network="",
            action="",
            job_id="job-1",
            entrypoint={"command": "echo hi"},
        )

    assert not getattr(raised.value, "writer_quiescent", False)


def test_uncertain_create_absence_is_not_cancellation_proof():
    handler = _build_handler()
    not_found = RuntimeError("not found")
    not_found.status = 404
    handler._batch.read_namespaced_job.side_effect = not_found
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "job_name": "tao-job-job-1",
        "job_uid": "",
        "namespace": "default",
        "service_name": "",
        "num_nodes": 1,
        "launch_uncertain": True,
    }

    assert not handler.cancel_job("job-1")
    handler._batch.delete_namespaced_job.assert_not_called()
    handler._core.list_namespaced_pod.assert_not_called()


def test_missing_runtime_status_is_nonterminal():
    handler = _build_handler()
    assert handler.get_tao_job_status("missing") == "Pending"


def test_normalize_status_known_values_pass_through():
    handler = _build_handler()
    for s in ("Pending", "Running", "Complete", "Error", "Canceled"):
        assert handler.normalize_status(s) == s
    assert handler.normalize_status("weird-state") == "Unknown"


def test_get_tao_job_status_maps_k8s_conditions():
    handler = _build_handler()
    handler._jobs["job-1"] = {"job_name": "tao-job-job-1", "namespace": "default"}

    handler._batch.read_namespaced_job.return_value = SimpleNamespace(
        status=SimpleNamespace(succeeded=1, failed=0, active=0)
    )
    assert handler.get_tao_job_status("job-1") == "Complete"

    handler._batch.read_namespaced_job.return_value = SimpleNamespace(
        status=SimpleNamespace(succeeded=0, failed=1, active=0)
    )
    assert handler.get_tao_job_status("job-1") == "Error"

    handler._batch.read_namespaced_job.return_value = SimpleNamespace(
        status=SimpleNamespace(succeeded=0, failed=0, active=1)
    )
    assert handler.get_tao_job_status("job-1") == "Running"


def test_cancel_job_calls_delete_with_foreground():
    handler = _build_handler()
    _cancel_runtime(handler)
    live_job = _running_job()
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])

    assert handler.cancel_job("job-1") is True
    handler._batch.delete_namespaced_job.assert_called_once_with(
        name="tao-job-job-1",
        namespace="default",
        body={
            "propagation_policy": "Foreground",
            "preconditions": {"uid": "uid-1"},
        },
    )


def test_cancel_job_adopts_label_validated_uid_for_delete_precondition():
    handler = _build_handler()
    _cancel_runtime(handler, uid="")
    live_job = _running_job(uid="uid-adopted")
    handler._batch.read_namespaced_job.side_effect = [
        live_job,
        live_job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])

    assert handler.cancel_job("job-1") is True
    assert handler._jobs["job-1"]["job_uid"] == "uid-adopted"
    assert (
        handler._batch.delete_namespaced_job.call_args.kwargs["body"]
        ["preconditions"]["uid"]
        == "uid-adopted"
    )


def test_cancel_job_fails_closed_if_name_is_replaced_after_validation():
    handler = _build_handler()
    _cancel_runtime(handler)
    original = _running_job()
    replacement = _running_job(uid="uid-replacement")
    handler._batch.read_namespaced_job.side_effect = [
        original,
        replacement,
        replacement,
    ]

    assert handler.cancel_job("job-1") is False
    handler._batch.delete_namespaced_job.assert_called_once_with(
        name="tao-job-job-1",
        namespace="default",
        body={
            "propagation_policy": "Foreground",
            "preconditions": {"uid": "uid-1"},
        },
    )
    handler._core.delete_namespaced_service.assert_not_called()


def test_cancel_job_rechecks_initial_404_and_refuses_late_replacement():
    handler = _build_handler()
    _cancel_runtime(handler)
    handler._batch.read_namespaced_job.side_effect = [
        _not_found_error(),
        _running_job(uid="uid-replacement"),
    ]

    assert handler.cancel_job("job-1") is False
    handler._batch.delete_namespaced_job.assert_not_called()
    handler._core.list_namespaced_pod.assert_not_called()
    handler._core.delete_namespaced_service.assert_not_called()


def _terminal_job(uid="uid-1", job_id="job-1"):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            uid=uid,
            labels={"tao-sdk-job-id": job_id},
        ),
        status=SimpleNamespace(
            succeeded=1,
            failed=0,
            active=0,
            conditions=[SimpleNamespace(type="Complete", status="True")],
        ),
    )


def _owned_terminal_pod(uid="uid-1", job_id="job-1"):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            labels={
                "job-name": f"tao-job-{job_id}",
                "tao-sdk-job-id": job_id,
            },
            owner_references=[SimpleNamespace(
                uid=uid,
                kind="Job",
                name=f"tao-job-{job_id}",
                controller=True,
            )],
        ),
        status=SimpleNamespace(phase="Succeeded"),
    )


def _terminal_cleanup_runtime(
    handler, uid="uid-1", *, num_nodes=1, service_uid=""
):
    handler._jobs["job-1"] = {
        "job_id": "job-1",
        "job_name": "tao-job-job-1",
        "job_uid": uid,
        "namespace": "default",
        "service_name": "tao-job-job-1" if num_nodes > 1 else "",
        "service_uid": service_uid if num_nodes > 1 else "",
        "num_nodes": num_nodes,
        "launch_uncertain": False,
    }


def _not_found_error():
    error = RuntimeError("not found")
    error.status = 404
    return error


def test_delete_terminal_job_uses_uid_precondition_and_waits_for_job_and_pods():
    handler = _build_handler()
    _terminal_cleanup_runtime(handler)
    job = _terminal_job()
    pod = _owned_terminal_pod()
    handler._batch.read_namespaced_job.side_effect = [
        job,
        job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[pod]),
        SimpleNamespace(items=[pod]),
        SimpleNamespace(items=[]),
    ]

    assert handler.delete_terminal_job("job-1") is True
    handler._batch.delete_namespaced_job.assert_called_once_with(
        name="tao-job-job-1",
        namespace="default",
        body={
            "propagation_policy": "Foreground",
            "preconditions": {"uid": "uid-1"},
        },
    )
    assert handler._core.list_namespaced_pod.call_count == 3


def test_delete_terminal_job_refuses_same_name_replacement():
    handler = _build_handler()
    _terminal_cleanup_runtime(handler)
    handler._batch.read_namespaced_job.return_value = _terminal_job(uid="uid-2")

    assert handler.delete_terminal_job("job-1") is False
    handler._batch.delete_namespaced_job.assert_not_called()
    handler._core.list_namespaced_pod.assert_not_called()


def test_delete_terminal_job_refuses_pod_not_owned_by_recorded_uid():
    handler = _build_handler()
    _terminal_cleanup_runtime(handler)
    job = _terminal_job()
    unrelated = _owned_terminal_pod(uid="uid-other")
    handler._batch.read_namespaced_job.side_effect = [job, job]
    handler._core.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[unrelated]),
        SimpleNamespace(items=[unrelated]),
    ]

    assert handler.delete_terminal_job("job-1") is False
    handler._batch.delete_namespaced_job.assert_not_called()


def test_delete_terminal_job_fails_closed_if_replacement_appears_while_waiting():
    handler = _build_handler()
    _terminal_cleanup_runtime(handler)
    original = _terminal_job()
    replacement = _terminal_job(uid="uid-2")
    handler._batch.read_namespaced_job.side_effect = [
        original,
        original,
        replacement,
    ]
    handler._core.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[]),
    ]

    assert handler.delete_terminal_job("job-1") is False
    assert handler._batch.delete_namespaced_job.call_count == 1
    assert (
        handler._batch.delete_namespaced_job.call_args.kwargs["body"]
        ["preconditions"]["uid"]
        == "uid-1"
    )
    handler._core.delete_namespaced_service.assert_not_called()


def test_delete_terminal_job_removes_uid_bound_multinode_service_last():
    handler = _build_handler()
    _terminal_cleanup_runtime(
        handler, num_nodes=2, service_uid="service-uid-1"
    )
    job = _terminal_job()
    service = SimpleNamespace(metadata=SimpleNamespace(
        uid="service-uid-1",
        labels={"tao-sdk-job-id": "job-1"},
    ))
    handler._batch.read_namespaced_job.side_effect = [
        job,
        job,
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.side_effect = [
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[]),
        SimpleNamespace(items=[]),
    ]
    handler._core.read_namespaced_service.side_effect = [
        service,
        _not_found_error(),
    ]

    assert handler.delete_terminal_job("job-1") is True
    handler._core.delete_namespaced_service.assert_called_once_with(
        name="tao-job-job-1",
        namespace="default",
        body={"preconditions": {"uid": "service-uid-1"}},
    )


def test_delete_terminal_job_is_restart_safe_after_exact_job_is_absent():
    handler = _build_handler()
    _terminal_cleanup_runtime(handler)
    handler._batch.read_namespaced_job.side_effect = [
        _not_found_error(),
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])

    assert handler.delete_terminal_job("job-1") is True
    handler._batch.delete_namespaced_job.assert_not_called()


def test_delete_terminal_job_retry_removes_remaining_multinode_service():
    handler = _build_handler()
    _terminal_cleanup_runtime(
        handler, num_nodes=2, service_uid="service-uid-1"
    )
    handler._batch.read_namespaced_job.side_effect = [
        _not_found_error(),
        _not_found_error(),
    ]
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[])
    handler._core.read_namespaced_service.side_effect = [
        SimpleNamespace(metadata=SimpleNamespace(
            uid="service-uid-1",
            labels={"tao-sdk-job-id": "job-1"},
        )),
        _not_found_error(),
    ]

    assert handler.delete_terminal_job("job-1") is True
    handler._core.delete_namespaced_service.assert_called_once()


def test_delete_terminal_job_requires_recorded_uid():
    handler = _build_handler()
    _terminal_cleanup_runtime(handler, uid="")

    assert handler.delete_terminal_job("job-1") is False
    handler._batch.read_namespaced_job.assert_not_called()


def test_uncertain_multinode_restart_adopts_and_persists_exact_service_uid():
    from tao_sdk.artifacts import artifact_specs
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK

    handler = _build_handler()
    runtime = handler.get_runtime_identity(
        "job-1",
        namespace="default",
        num_nodes=2,
        launch_uncertain=True,
    )
    assert handler.restore_job_runtime("job-1", runtime)
    handler._batch.read_namespaced_job.return_value = SimpleNamespace(
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
    handler._core.read_namespaced_service.return_value = SimpleNamespace(
        metadata=SimpleNamespace(
            name="tao-job-job-1",
            uid="service-uid-1",
            labels={"tao-sdk-job-id": "job-1"},
        )
    )

    assert handler.get_tao_job_status("job-1") == "Running"
    reconciled = handler.get_job_runtime_identity("job-1")
    assert reconciled["job_uid"] == "uid-1"
    assert reconciled["service_uid"] == "service-uid-1"
    assert reconciled["launch_uncertain"] is False

    sdk = object.__new__(KubernetesSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "specs": artifact_specs(
            "s3",
            "s3://bucket/results/job-1",
            runtime_identity=runtime,
        ),
    }
    sdk._persist_runtime_identity("job-1")

    persisted = sdk._store.update_job_specs.call_args.args[1]
    assert (
        persisted["_tao_artifacts"]["runtime_identity"]["service_uid"]
        == "service-uid-1"
    )


def test_uncertain_service_is_not_adopted_with_conflicting_job_uid():
    handler = _build_handler()
    runtime = handler.get_runtime_identity(
        "job-1",
        namespace="default",
        num_nodes=2,
        job_uid="uid-original",
        launch_uncertain=True,
    )
    assert handler.restore_job_runtime("job-1", runtime)
    handler._batch.read_namespaced_job.return_value = _terminal_job(
        uid="uid-replacement"
    )

    assert handler.get_tao_job_status("job-1") == "Pending"
    handler._core.read_namespaced_service.assert_not_called()
    assert handler._jobs["job-1"]["service_uid"] == ""


def test_get_failure_analysis_oomkilled():
    handler = _build_handler()
    handler._jobs["job-1"] = {"job_name": "tao-job-job-1", "namespace": "default"}
    handler._core.list_namespaced_pod.return_value = SimpleNamespace(items=[
        SimpleNamespace(
            metadata=SimpleNamespace(name="pod-1"),
            spec=SimpleNamespace(node_name="node-x"),
            status=SimpleNamespace(
                phase="Failed",
                container_statuses=[SimpleNamespace(
                    name="tao",
                    state=SimpleNamespace(
                        terminated=SimpleNamespace(
                            exit_code=137, reason="OOMKilled", message="oom"
                        ),
                        waiting=None,
                    ),
                )],
            ),
        )
    ])
    analysis = handler.get_failure_analysis("job-1")
    assert "OOM" in analysis["suggestion"]
    assert analysis["err_class"] == "ERR_PROGRAM"


def test_credential_error_when_no_auth_available():
    """KubernetesHandler raises CredentialError if neither kubeconfig nor in-cluster works."""
    from tao_sdk.platforms.kubernetes.handler import KubernetesHandler
    from tao_sdk.models import CredentialError
    from kubernetes.config.config_exception import ConfigException

    fake_client = _patch_k8s_client_module()
    fake_config = MagicMock()
    fake_config.load_kube_config.side_effect = ConfigException("no kubeconfig")
    fake_config.load_incluster_config.side_effect = ConfigException("not in cluster")

    with patch("tao_sdk.platforms.kubernetes.handler.KubernetesHandler._ping", lambda self: None):
        with patch("kubernetes.client", fake_client), \
             patch("kubernetes.config", fake_config):
            with pytest.raises(CredentialError, match="kubectl"):
                KubernetesHandler(validate=False)


# ---------------------------------------------------------------------------
# _results_root_prefix — what gets injected as TAO_RESULTS_ROOT
# ---------------------------------------------------------------------------

def test_k8s_results_root_prefix_is_results_when_pvc_mounted():
    """A volume mount at /results signals 'I have a PVC for outputs' — prefix
    is /results, runner writes directly, no cloud upload."""
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK
    sdk = object.__new__(KubernetesSDK)
    vms = [{"name": "results", "mountPath": "/results"}]
    assert sdk._results_root_prefix(volume_mounts=vms) == "/results"


def test_k8s_results_root_prefix_is_none_without_pvc():
    """No /results PVC → no TAO_RESULTS_ROOT. script_runner falls back to S3
    (if S3_BUCKET_NAME) or container-ephemeral with warning."""
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK
    sdk = object.__new__(KubernetesSDK)
    assert sdk._results_root_prefix() is None


def test_k8s_results_root_prefix_handles_object_style_volume_mount():
    """V1VolumeMount-shaped objects (with .mount_path) also work."""
    from tao_sdk.platforms.kubernetes.sdk import KubernetesSDK
    sdk = object.__new__(KubernetesSDK)
    obj = type("VM", (), {"mount_path": "/results"})()
    assert sdk._results_root_prefix(volume_mounts=[obj]) == "/results"
