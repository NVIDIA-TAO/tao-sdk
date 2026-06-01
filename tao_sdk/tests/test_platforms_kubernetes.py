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


def test_create_job_multinode_capacity_check_uses_total():
    """For 2 nodes × 4 GPUs the cluster needs 8 allocatable, not 4."""
    handler = _build_handler(gpu_capacity=4)
    with pytest.raises(RuntimeError, match="× 2 nodes = 8 total"):
        handler.create_job(
            image="img:tag", network="", action="",
            entrypoint={"command": "echo hi"},
            num_gpus=4, num_nodes=2,
        )


def test_cancel_job_deletes_service_for_multinode():
    handler = _build_handler()
    handler._jobs["job-mn"] = {
        "job_name": "tao-job-job-mn",
        "namespace": "default",
        "service_name": "tao-job-job-mn",
        "num_nodes": 2,
    }
    assert handler.cancel_job("job-mn") is True
    handler._batch.delete_namespaced_job.assert_called_once()
    handler._core.delete_namespaced_service.assert_called_once_with(
        name="tao-job-job-mn", namespace="default"
    )


def test_cancel_job_skips_service_for_single_pod():
    handler = _build_handler()
    handler._jobs["job-1"] = {
        "job_name": "tao-job-job-1",
        "namespace": "default",
        "service_name": "",
        "num_nodes": 1,
    }
    handler.cancel_job("job-1")
    handler._core.delete_namespaced_service.assert_not_called()


def test_create_job_requires_entrypoint_command():
    handler = _build_handler()
    with pytest.raises(ValueError, match="entrypoint"):
        handler.create_job(
            image="img:tag", network="", action="",
            entrypoint=None,
        )


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
    handler._jobs["job-1"] = {"job_name": "tao-job-job-1", "namespace": "default"}
    assert handler.cancel_job("job-1") is True
    handler._batch.delete_namespaced_job.assert_called_once_with(
        name="tao-job-job-1", namespace="default", propagation_policy="Foreground"
    )


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
