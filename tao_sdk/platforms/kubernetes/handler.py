# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes execution handler for TAO SDK.

Submits jobs as Kubernetes ``Jobs`` (single-pod) using the official
``kubernetes`` Python client. Authenticates via ``~/.kube/config`` (laptop
case) or in-cluster service account (when running inside a pod).

The handler is single-pod only — multi-node distributed training would
require Volcano, the MPI Operator, or the Kubeflow PyTorch operator and
is out of scope. For multi-node, extend later.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from tao_sdk.platforms.base import BaseHandler

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "default"
GPU_RESOURCE_KEY = "nvidia.com/gpu"
GPU_OPERATOR_DOCS = (
    "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html"
)


class KubernetesHandler(BaseHandler):
    """Execution handler for Kubernetes clusters.

    Auth is auto-detected: kubeconfig first (developer laptop), then
    in-cluster service account (running inside a pod). Raises
    ``CredentialError`` if both fail.

    GPU scheduling uses the standard ``nvidia.com/gpu`` resource. Requires the
    NVIDIA GPU Operator (or device plugin) to be installed on the cluster.
    """

    backend_type = "kubernetes"

    def __init__(
        self,
        creds: dict[str, Any] | None = None,
        namespace: str | None = None,
        context: str | None = None,
        kubeconfig_path: str | None = None,
        validate: bool = True,
    ):
        self._creds = creds or {}
        self._namespace = (
            namespace
            or self._creds.get("TAO_K8S_NAMESPACE")
            or DEFAULT_NAMESPACE
        )
        self._context = context or self._creds.get("TAO_K8S_CONTEXT")
        self._kubeconfig_path = kubeconfig_path or self._creds.get("KUBECONFIG")

        try:
            import kubernetes
            from kubernetes import client, config
            from kubernetes.config.config_exception import ConfigException
        except ImportError as e:
            from tao_sdk.models import CredentialError
            raise CredentialError(
                "kubernetes Python client not installed. Run:\n"
                "  pip install nvidia-tao-sdk[kubernetes]"
            ) from e

        self._client_module = client
        self._config_module = config
        self._config_exception = ConfigException

        self._configure_auth()

        self._batch = client.BatchV1Api()
        self._core = client.CoreV1Api()
        self._jobs: dict[str, dict] = {}
        self._gpu_capacity_cache: int | None = None

        if validate:
            self._ping()

        logger.info(
            "KubernetesHandler initialized for namespace=%s context=%s",
            self._namespace,
            self._context or "default",
        )

    @classmethod
    def from_credentials(cls, creds: dict) -> "KubernetesHandler":
        """Construct a KubernetesHandler from SDK credentials/environment."""
        try:
            return cls(creds=creds or {})
        except Exception as e:
            from tao_sdk.models import CredentialError
            if isinstance(e, CredentialError):
                raise
            raise CredentialError(
                "Failed to initialize Kubernetes backend. Ensure kubectl is "
                "configured (~/.kube/config) or the SDK is running inside a "
                f"pod with a service account. Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Auth + cluster validation
    # ------------------------------------------------------------------

    def _configure_auth(self):
        """Try kubeconfig; fall back to in-cluster service account."""
        from tao_sdk.models import CredentialError

        kubeconfig_attempted = self._kubeconfig_path or "~/.kube/config (or $KUBECONFIG)"
        try:
            self._config_module.load_kube_config(
                config_file=self._kubeconfig_path,
                context=self._context,
            )
            logger.debug("Loaded kubeconfig from %s", kubeconfig_attempted)
            return
        except self._config_exception as kc_err:
            logger.debug("Kubeconfig load failed: %s", kc_err)

        try:
            self._config_module.load_incluster_config()
            logger.debug("Loaded in-cluster service account config")
            return
        except self._config_exception as ic_err:
            raise CredentialError(
                "Could not authenticate to a Kubernetes cluster.\n"
                f"  - Kubeconfig: tried {kubeconfig_attempted}; not found or invalid.\n"
                f"  - In-cluster service account: not available ({ic_err}).\n"
                "Configure kubectl (e.g., `aws eks update-kubeconfig`) or set "
                "$KUBECONFIG, or run the SDK from inside a pod."
            )

    def _ping(self):
        """Verify the API server is reachable."""
        try:
            self._core.get_api_resources()
        except Exception as e:
            from tao_sdk.models import CredentialError
            raise CredentialError(
                f"Could not reach the Kubernetes API server: {e}"
            ) from e

    def get_total_gpu_capacity(self, refresh: bool = False) -> int:
        """Sum of nvidia.com/gpu allocatable across all nodes. Cached."""
        if self._gpu_capacity_cache is not None and not refresh:
            return self._gpu_capacity_cache
        try:
            nodes = self._core.list_node().items
        except Exception as e:
            logger.warning("list_node failed: %s", e)
            return 0
        total = 0
        for node in nodes:
            allocatable = (node.status.allocatable or {}) if node.status else {}
            try:
                total += int(allocatable.get(GPU_RESOURCE_KEY, "0"))
            except (TypeError, ValueError):
                continue
        self._gpu_capacity_cache = total
        return total

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def get_job_name(self, job_id: str) -> str:
        """Generate a k8s-safe Job name (DNS-1123: lowercase, dashes, ≤63 chars)."""
        safe = re.sub(r"[^a-z0-9-]+", "-", str(job_id).lower()).strip("-")
        return f"tao-job-{safe}"[:63].rstrip("-")

    def create_job(
        self,
        image: str,
        network: str,
        action: str,
        cloud_metadata: dict | None = None,
        specs: dict | None = None,
        job_id: str = "",
        docker_env_vars: dict | None = None,
        resource_shape: str | None = None,
        dedicated_node_group: str | None = None,
        num_nodes: int = 1,
        entrypoint: dict | None = None,
        mounts: list | None = None,
        **kwargs,
    ) -> dict:
        """Create a Kubernetes Job for an SDK job.

        For ``num_nodes == 1``: a standard ``Job`` with one pod.

        For ``num_nodes > 1``: a headless ``Service`` plus an ``Indexed`` ``Job``
        with ``parallelism=completions=num_nodes``. The user command is wrapped
        with PyTorch-distributed env vars (``NODE_RANK`` from
        ``JOB_COMPLETION_INDEX``, ``WORLD_SIZE``, ``MASTER_PORT``,
        ``MASTER_ADDR=<job-name>-0.<job-name>``). Requires k8s 1.28+ for
        the auto-set pod hostname pattern.
        """
        if not entrypoint or "command" not in entrypoint:
            raise ValueError("entrypoint with 'command' key is required")
        num_nodes = max(int(num_nodes or 1), 1)

        gpu_count = int(kwargs.get("num_gpu") or kwargs.get("num_gpus") or 0)
        if gpu_count > 0:
            available = self.get_total_gpu_capacity()
            required = gpu_count * num_nodes
            if available < required:
                raise RuntimeError(
                    f"Cluster has {available} {GPU_RESOURCE_KEY} allocatable but "
                    f"job requested {gpu_count}/node × {num_nodes} nodes = {required} total. "
                    f"Install or scale the NVIDIA GPU Operator: {GPU_OPERATOR_DOCS}"
                )

        job_id = job_id or str(int(time.time()))
        job_name = self.get_job_name(job_id)
        namespace = kwargs.get("namespace") or self._namespace
        env_vars = dict(docker_env_vars or {})

        labels = {
            "tao-toolkit": "true",
            "tao-sdk-job-id": job_id,
            "tao-sdk-network": network or "",
            "tao-sdk-action": action or "",
        }

        # Multi-node: create the headless Service first so DNS is ready when pods come up.
        service_name = ""
        if num_nodes > 1:
            service_name = job_name  # reuse job name; cleaner DNS
            service = self._build_headless_service(service_name, namespace, labels, job_name)
            try:
                self._core.create_namespaced_service(namespace=namespace, body=service)
                logger.info("Created headless Service %s/%s for multi-node Job", namespace, service_name)
            except Exception as e:
                # Don't pre-clean — let the user investigate stale state.
                raise RuntimeError(
                    f"Failed to create headless Service {namespace}/{service_name} "
                    f"for multi-node Job: {e}"
                ) from e

        body = self._build_job_manifest(
            job_name=job_name,
            image=image,
            command=entrypoint["command"],
            env_vars=env_vars,
            gpu_count=gpu_count,
            num_nodes=num_nodes,
            namespace=namespace,
            image_pull_secret=kwargs.get("image_pull_secret"),
            node_selector=kwargs.get("node_selector"),
            tolerations=kwargs.get("tolerations"),
            labels=labels,
            service_name=service_name,
        )

        logger.info(
            "Submitting k8s Job %s to namespace=%s (nodes=%d, gpu/node=%d)",
            job_name, namespace, num_nodes, gpu_count,
        )
        api_response = self._batch.create_namespaced_job(namespace=namespace, body=body)

        self._jobs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "namespace": namespace,
            "image": image,
            "service_name": service_name,  # empty when single-node
            "num_nodes": num_nodes,
            "created_at": time.time(),
        }
        return {"job_name": job_name, "uid": api_response.metadata.uid}

    def _build_headless_service(
        self,
        service_name: str,
        namespace: str,
        labels: dict,
        job_name: str,
    ):
        """Headless Service for multi-node pod-to-pod DNS rendezvous."""
        client = self._client_module
        return client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=service_name, namespace=namespace, labels=labels,
            ),
            spec=client.V1ServiceSpec(
                cluster_ip="None",                       # headless — DNS returns pod IPs
                selector={"job-name": job_name},          # k8s auto-labels Job pods with this
                publish_not_ready_addresses=True,         # rendezvous before all pods are Ready
            ),
        )

    def _build_job_manifest(
        self,
        job_name: str,
        image: str,
        command: str,
        env_vars: dict,
        gpu_count: int,
        num_nodes: int,
        namespace: str,
        image_pull_secret: str | None,
        node_selector: dict | None,
        tolerations: list | None,
        labels: dict,
        service_name: str = "",
    ):
        """Construct a V1Job for the request — single pod or Indexed Job."""
        client = self._client_module

        # Wrap user command with PyTorch-distributed env vars when multi-node.
        # Single-node: pass the user command through unchanged.
        if num_nodes > 1:
            # Match the env-var contract the TAO PyTorch container reads in
            # nvidia_tao_pytorch/core/entrypoint.py — WORLD_SIZE (= num_nodes,
            # TAO's misnamed convention), NUM_GPU_PER_NODE, NODE_RANK,
            # MASTER_ADDR, MASTER_PORT (default 29500). Also export the
            # PyTorch-standard NNODES/NPROC_PER_NODE for raw torchrun commands.
            # Pod-0's stable DNS via headless Service + Indexed Job pod
            # hostnames (requires k8s 1.28+ for the PodIndexLabel feature).
            wrapped = (
                f"set -euo pipefail\n"
                # TAO container contract:
                f"export WORLD_SIZE={num_nodes}\n"
                f"export NUM_GPU_PER_NODE={gpu_count}\n"
                f"export NODE_RANK=${{JOB_COMPLETION_INDEX:-0}}\n"
                # PyTorch-standard synonyms (for raw torchrun users):
                f"export NNODES={num_nodes}\n"
                f"export NPROC_PER_NODE={gpu_count}\n"
                # Rendezvous endpoint:
                f"export MASTER_PORT=29500\n"
                f"export MASTER_ADDR={job_name}-0.{service_name}\n"
                f'echo "[tao-k8s] node $NODE_RANK / $WORLD_SIZE; MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"\n'
                f"\n"
                f"{command}\n"
            )
        else:
            wrapped = command

        env_list = [client.V1EnvVar(name=k, value=str(v)) for k, v in env_vars.items()]

        resources = None
        if gpu_count > 0:
            resources = client.V1ResourceRequirements(
                limits={GPU_RESOURCE_KEY: str(gpu_count)},
            )

        container = client.V1Container(
            name="tao",
            image=image,
            command=["/bin/bash", "-c", wrapped],
            env=env_list,
            resources=resources,
        )

        pod_spec_kwargs = {
            "containers": [container],
            "restart_policy": "Never",
        }
        if num_nodes > 1 and service_name:
            # Subdomain → pods get FQDN <pod-hostname>.<service>.<ns>.svc.cluster.local.
            # k8s 1.28+ auto-sets pod hostname to <job-name>-<index> for Indexed Jobs.
            pod_spec_kwargs["subdomain"] = service_name
        if image_pull_secret:
            pod_spec_kwargs["image_pull_secrets"] = [
                client.V1LocalObjectReference(name=image_pull_secret),
            ]
        if node_selector:
            pod_spec_kwargs["node_selector"] = node_selector
        if tolerations:
            pod_spec_kwargs["tolerations"] = tolerations

        pod_spec = client.V1PodSpec(**pod_spec_kwargs)
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=pod_spec,
        )

        job_spec_kwargs = {
            "template": pod_template,
            "backoff_limit": 0,                    # failures surface to user, no retry
            "ttl_seconds_after_finished": 3600,    # auto-clean after 1h
        }
        if num_nodes > 1:
            job_spec_kwargs["parallelism"] = num_nodes
            job_spec_kwargs["completions"] = num_nodes
            job_spec_kwargs["completion_mode"] = "Indexed"

        job_spec = client.V1JobSpec(**job_spec_kwargs)
        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name, namespace=namespace, labels=labels),
            spec=job_spec,
        )

    # ------------------------------------------------------------------
    # Status + logs
    # ------------------------------------------------------------------

    def get_tao_job_status(self, job_id: str) -> str | None:
        """Read the k8s Job status; return raw status string for normalize_status."""
        info = self._jobs.get(job_id)
        if not info:
            return None
        try:
            job = self._batch.read_namespaced_job(
                name=info["job_name"], namespace=info["namespace"],
            )
        except Exception as e:
            logger.debug("read_namespaced_job failed for %s: %s", info["job_name"], e)
            return None
        status = job.status
        if status is None:
            return "Pending"
        if status.succeeded:
            return "Complete"
        if status.failed:
            return "Error"
        if status.active:
            return "Running"
        return "Pending"

    def normalize_status(self, raw_status: str | None) -> str:
        if raw_status in ("Pending", "Running", "Complete", "Error", "Canceled"):
            return raw_status
        return "Unknown"

    def get_job_message(self, job_id: str) -> str:
        info = self._jobs.get(job_id)
        if not info:
            return ""
        try:
            job = self._batch.read_namespaced_job(
                name=info["job_name"], namespace=info["namespace"],
            )
        except Exception:
            return ""
        conditions = (job.status.conditions or []) if job.status else []
        if conditions:
            return conditions[-1].message or ""
        return ""

    def get_job_logs(self, job_id: str, tail_lines: int | None = None) -> str | None:
        """Concat logs from all pods owned by the Job."""
        info = self._jobs.get(job_id)
        if not info:
            return None
        try:
            pods = self._core.list_namespaced_pod(
                namespace=info["namespace"],
                label_selector=f"job-name={info['job_name']}",
            ).items
        except Exception as e:
            logger.warning("list_namespaced_pod failed: %s", e)
            return None
        chunks = []
        for pod in pods:
            try:
                log = self._core.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=info["namespace"],
                    tail_lines=tail_lines,
                )
                chunks.append(log)
            except Exception as e:
                logger.debug("read_namespaced_pod_log failed for %s: %s", pod.metadata.name, e)
        return "\n".join(chunks) if chunks else None

    def cancel_job(self, job_id: str) -> bool:
        info = self._jobs.get(job_id)
        if not info:
            return False
        ok = True
        try:
            self._batch.delete_namespaced_job(
                name=info["job_name"],
                namespace=info["namespace"],
                propagation_policy="Foreground",
            )
        except Exception as e:
            logger.warning("delete_namespaced_job failed for %s: %s", info["job_name"], e)
            ok = False

        # Multi-node jobs also have a headless Service to clean up.
        service_name = info.get("service_name") or ""
        if service_name:
            try:
                self._core.delete_namespaced_service(
                    name=service_name, namespace=info["namespace"],
                )
            except Exception as e:
                logger.warning(
                    "delete_namespaced_service failed for %s: %s", service_name, e,
                )
                # Don't fail cancel if Service deletion fails — Job is what matters.
        return ok

    def delete_job(self, job_id: str) -> bool:
        return self.cancel_job(job_id)

    def get_failure_analysis(self, job_id: str) -> dict | None:
        """Map terminated-pod state to a user-readable failure cause."""
        info = self._jobs.get(job_id)
        if not info:
            return None
        try:
            pods = self._core.list_namespaced_pod(
                namespace=info["namespace"],
                label_selector=f"job-name={info['job_name']}",
            ).items
        except Exception as e:
            logger.warning("list_namespaced_pod failed: %s", e)
            return None

        events = []
        suggestion = ""
        err_class = "ERR_PROGRAM"
        for pod in pods:
            statuses = (pod.status.container_statuses or []) if pod.status else []
            for cs in statuses:
                term = cs.state.terminated if cs.state else None
                if term:
                    events.append({
                        "pod": pod.metadata.name,
                        "container": cs.name,
                        "exit_code": term.exit_code,
                        "reason": term.reason or "",
                        "message": term.message or "",
                    })
                    if term.reason == "OOMKilled":
                        suggestion = (
                            "Container OOM-killed. Reduce batch size, lower "
                            "max_length, or request a larger node."
                        )
                    elif term.exit_code == 137:
                        suggestion = "Container received SIGKILL (likely OOM or eviction)."
                    elif term.reason == "Error" and not suggestion:
                        suggestion = "Check container logs for the failure stack."
            wait = (statuses[0].state.waiting if statuses and statuses[0].state else None)
            if wait and wait.reason in ("ImagePullBackOff", "ErrImagePull"):
                err_class = "ERR_INFRA"
                suggestion = (
                    "Image pull failed. Verify the image URI and (for nvcr.io) "
                    "that NGC_KEY is valid and an image-pull-secret is attached."
                )

        if not events:
            return None
        return {
            "err_class": err_class,
            "suggestion": suggestion,
            "job_failure_by_node_event": [
                {
                    "node_event_name": e["reason"] or f"exit-{e['exit_code']}",
                    "message": e["message"] or f"Container {e['container']} exited {e['exit_code']}",
                }
                for e in events
            ],
            "log_streams": [],
        }

    def get_job_replicas(self, job_id: str) -> list[dict]:
        """Pod-level diagnostics for stuck-Pending Jobs."""
        info = self._jobs.get(job_id)
        if not info:
            return []
        try:
            pods = self._core.list_namespaced_pod(
                namespace=info["namespace"],
                label_selector=f"job-name={info['job_name']}",
            ).items
        except Exception:
            return []
        out = []
        for pod in pods:
            statuses = (pod.status.container_statuses or []) if pod.status else []
            wait = statuses[0].state.waiting if statuses and statuses[0].state else None
            readiness_issue = None
            if wait:
                readiness_issue = {
                    "reason": wait.reason or "Waiting",
                    "message": wait.message or "",
                }
            out.append({
                "status": {
                    "node": {
                        "name": pod.spec.node_name or "",
                        "node_group_id": "",
                    },
                    "phase": pod.status.phase if pod.status else "Unknown",
                    "readiness_issue": readiness_issue,
                },
            })
        return out

    def get_available_instances(self, **kwargs) -> dict:
        """Return cluster GPU capacity summary."""
        return {
            "total_nvidia_gpus": self.get_total_gpu_capacity(refresh=True),
        }
