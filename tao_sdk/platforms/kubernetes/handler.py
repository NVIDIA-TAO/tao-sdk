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

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from tao_sdk.platforms.base import BaseHandler

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "default"
GPU_RESOURCE_KEY = "nvidia.com/gpu"
CANCEL_WAIT_SECONDS = 30
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

    def get_backend_identity(self) -> str:
        """Return a non-secret fingerprint of the authenticated cluster."""
        configuration = getattr(
            getattr(self._batch, "api_client", None),
            "configuration",
            None,
        )
        host = str(getattr(configuration, "host", "") or "").strip().rstrip("/")
        if not host:
            logger.warning("Kubernetes client did not expose an API server identity")
            return ""

        ca_digest = ""
        ca_path = str(getattr(configuration, "ssl_ca_cert", "") or "")
        if ca_path:
            try:
                ca_digest = hashlib.sha256(Path(ca_path).read_bytes()).hexdigest()
            except OSError as exc:
                logger.warning("Could not fingerprint Kubernetes CA certificate: %s", exc)

        cluster_uid = ""
        try:
            namespace = self._core.read_namespace(name="kube-system")
            candidate = getattr(getattr(namespace, "metadata", None), "uid", "")
            if isinstance(candidate, (str, int)):
                cluster_uid = str(candidate)
        except Exception as exc:
            logger.debug("Could not read kube-system UID for cluster identity: %s", exc)

        if not cluster_uid and not ca_digest:
            logger.warning(
                "Kubernetes cluster did not provide a stable UID or CA fingerprint"
            )
            return ""
        target = json.dumps(
            {"api_server": host, "ca_sha256": ca_digest, "cluster_uid": cluster_uid},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        return f"kubernetes-sha256:{digest}"

    def get_runtime_identity(
        self,
        job_id: str,
        *,
        namespace: str | None,
        num_nodes: int,
        job_uid: str = "",
        service_uid: str = "",
        launch_uncertain: bool = False,
    ) -> dict:
        """Build the durable coordinates needed to recover a Kubernetes Job."""
        resolved_namespace = namespace or self._namespace
        resolved_nodes = max(int(num_nodes or 1), 1)
        job_name = self.get_job_name(job_id)
        return {
            "job_name": job_name,
            "job_uid": str(job_uid or ""),
            "namespace": resolved_namespace,
            "num_nodes": resolved_nodes,
            "service_name": job_name if resolved_nodes > 1 else "",
            "service_uid": str(service_uid or "") if resolved_nodes > 1 else "",
            "launch_uncertain": bool(launch_uncertain),
        }

    def restore_job_runtime(self, job_id: str, runtime: dict | None) -> bool:
        """Restore validated handler state after an SDK process restart."""
        if not isinstance(runtime, dict):
            return False
        try:
            expected = self.get_runtime_identity(
                job_id,
                namespace=str(runtime.get("namespace", "")),
                num_nodes=int(runtime.get("num_nodes", 1)),
                job_uid=str(runtime.get("job_uid", "")),
                service_uid=str(runtime.get("service_uid", "")),
                launch_uncertain=runtime.get("launch_uncertain", False),
            )
        except (TypeError, ValueError):
            return False
        namespace = expected["namespace"]
        if (
            runtime.get("job_name") != expected["job_name"]
            or runtime.get("namespace") != namespace
            or runtime.get("service_name", "") != expected["service_name"]
            or not isinstance(runtime.get("job_uid", ""), str)
            or not isinstance(runtime.get("service_uid", ""), str)
            or not isinstance(runtime.get("launch_uncertain", False), bool)
            or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace)
            or len(namespace) > 63
        ):
            return False
        self._jobs[job_id] = {"job_id": job_id, **expected}
        return True

    def get_job_runtime_identity(self, job_id: str) -> dict | None:
        """Return the validated durable coordinates currently in the cache."""
        info = self._jobs.get(job_id)
        if not info:
            return None
        return {
            "job_name": str(info.get("job_name", "")),
            "job_uid": str(info.get("job_uid", "")),
            "namespace": str(info.get("namespace", "")),
            "num_nodes": int(info.get("num_nodes", 1) or 1),
            "service_name": str(info.get("service_name", "")),
            "service_uid": str(info.get("service_uid", "")),
            "launch_uncertain": bool(info.get("launch_uncertain", False)),
        }

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
            error = ValueError("entrypoint with 'command' key is required")
            error.writer_quiescent = True
            raise error
        num_nodes = max(int(num_nodes or 1), 1)

        gpu_count = int(kwargs.get("num_gpu") or kwargs.get("num_gpus") or 0)
        if gpu_count > 0:
            available = self.get_total_gpu_capacity()
            required = gpu_count * num_nodes
            if available < required:
                error = RuntimeError(
                    f"Cluster has {available} {GPU_RESOURCE_KEY} allocatable but "
                    f"job requested {gpu_count}/node × {num_nodes} nodes = {required} total. "
                    f"Install or scale the NVIDIA GPU Operator: {GPU_OPERATOR_DOCS}"
                )
                error.writer_quiescent = True
                raise error

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
        service_uid = ""
        if num_nodes > 1:
            service_name = job_name  # reuse job name; cleaner DNS
            service = self._build_headless_service(service_name, namespace, labels, job_name)
            try:
                service_response = self._core.create_namespaced_service(
                    namespace=namespace, body=service
                )
                service_uid = str(
                    getattr(
                        getattr(service_response, "metadata", None), "uid", ""
                    )
                    or ""
                )
                logger.info("Created headless Service %s/%s for multi-node Job", namespace, service_name)
            except Exception as e:
                # The Job create API has not been called, so no training
                # writer can exist even if Service creation completed late.
                error = RuntimeError(
                    f"Failed to create headless Service {namespace}/{service_name} "
                    f"for multi-node Job: {e}"
                )
                error.writer_quiescent = True
                raise error from e

        try:
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
        except Exception as exc:
            exc.writer_quiescent = True
            raise

        logger.info(
            "Submitting k8s Job %s to namespace=%s (nodes=%d, gpu/node=%d)",
            job_name, namespace, num_nodes, gpu_count,
        )
        try:
            api_response = self._batch.create_namespaced_job(
                namespace=namespace, body=body
            )
        except Exception as exc:
            if self._is_definitive_create_rejection(exc):
                # A concrete API rejection proves the Job was not admitted.
                # Conflict is excluded because it can be a late prior create.
                exc.writer_quiescent = True
            raise

        self._jobs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "namespace": namespace,
            "image": image,
            "service_name": service_name,  # empty when single-node
            "service_uid": service_uid,
            "num_nodes": num_nodes,
            "job_uid": str(api_response.metadata.uid or ""),
            "launch_uncertain": False,
            "created_at": time.time(),
        }
        return {
            "job_name": job_name,
            "uid": api_response.metadata.uid,
            "service_uid": service_uid,
        }

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
            return "Pending"
        terminal_status = info.get("terminal_status")
        if terminal_status in {"Complete", "Error", "Canceled"}:
            return terminal_status
        try:
            job = self._batch.read_namespaced_job(
                name=info["job_name"], namespace=info["namespace"],
            )
        except Exception as e:
            logger.debug("read_namespaced_job failed for %s: %s", info["job_name"], e)
            # A TTL controller may remove a successfully completed Job while
            # the SDK is down. Absence proves neither success nor cancellation,
            # so only the explicit cancel path may classify it as quiescent.
            return "Pending"
        if not self._live_job_identity_matches(info, job):
            return "Pending"
        launch_was_uncertain = info.get("launch_uncertain") is True
        observed_uid = str(getattr(getattr(job, "metadata", None), "uid", "") or "")
        if observed_uid:
            info["job_uid"] = observed_uid
        if (
            launch_was_uncertain
            and info.get("service_name")
            and not info.get("service_uid")
            and not self._adopt_uncertain_service_identity(info)
        ):
            logger.warning(
                "Kubernetes Job %s was reconciled but its Service UID is still "
                "uncertain",
                info["job_name"],
            )
            return "Pending"
        info["launch_uncertain"] = False
        status = job.status
        if status is None:
            return "Pending"
        conditions = list(getattr(status, "conditions", None) or [])
        terminal_condition = None
        for condition in conditions:
            if str(getattr(condition, "status", "")).lower() != "true":
                continue
            condition_type = str(getattr(condition, "type", ""))
            if condition_type in {"Complete", "Failed"}:
                terminal_condition = condition_type
                break

        num_nodes = int(info.get("num_nodes", 1) or 1)
        if num_nodes > 1:
            # Per-replica succeeded/failed counters are not terminal for an
            # Indexed Job. Wait for the controller's aggregate condition.
            if terminal_condition and self._job_pods_are_quiescent(info):
                return "Complete" if terminal_condition == "Complete" else "Error"
        else:
            succeeded = getattr(status, "succeeded", 0)
            failed = getattr(status, "failed", 0)
            active = getattr(status, "active", 0)
            succeeded = (
                succeeded if isinstance(succeeded, int) and not isinstance(succeeded, bool)
                else 0
            )
            failed = (
                failed if isinstance(failed, int) and not isinstance(failed, bool)
                else 0
            )
            active = (
                active if isinstance(active, int) and not isinstance(active, bool)
                else 0
            )
            if succeeded > 0 and self._job_pods_are_quiescent(info):
                return "Complete"
            if failed > 0 and self._job_pods_are_quiescent(info):
                return "Error"
        if (
            isinstance(getattr(status, "active", 0), int)
            and not isinstance(getattr(status, "active", 0), bool)
            and getattr(status, "active", 0) > 0
        ):
            return "Running"
        return "Pending"

    def _adopt_uncertain_service_identity(self, info: dict) -> bool:
        """Adopt a crash-window Service only after exact Job reconciliation."""
        if not str(info.get("job_uid", "") or ""):
            return False
        service_name = str(info.get("service_name", "") or "")
        expected_job_id = str(info.get("job_id", "") or "")
        if not service_name or not expected_job_id:
            return False
        try:
            service = self._core.read_namespaced_service(
                name=service_name, namespace=info["namespace"],
            )
        except Exception as exc:
            logger.warning(
                "Could not reconcile Kubernetes Service %s: %s",
                service_name,
                exc,
            )
            return False
        metadata = getattr(service, "metadata", None)
        observed_name = str(getattr(metadata, "name", "") or "")
        observed_uid = str(getattr(metadata, "uid", "") or "")
        labels = getattr(metadata, "labels", None) or {}
        if (
            observed_name != service_name
            or not observed_uid
            or not isinstance(labels, dict)
            or labels.get("tao-sdk-job-id") != expected_job_id
        ):
            logger.warning(
                "Refusing to adopt Kubernetes Service %s without its exact "
                "name, UID, and TAO job label",
                service_name,
            )
            return False
        info["service_uid"] = observed_uid
        return True

    def _job_pods_are_quiescent(self, info: dict) -> bool:
        """Return true only when no pod belonging to a Job can still write."""
        try:
            response = self._core.list_namespaced_pod(
                namespace=info["namespace"],
                label_selector=f"job-name={info['job_name']}",
            )
            pods = list(getattr(response, "items", None) or [])
        except Exception as exc:
            logger.warning(
                "Could not confirm pod quiescence for %s: %s",
                info["job_name"],
                exc,
            )
            return False
        return all(
            str(getattr(getattr(pod, "status", None), "phase", "")).lower()
            not in {"pending", "running", "unknown", ""}
            for pod in pods
        )

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
        recorded_uid = str(info.get("job_uid", "") or "")
        try:
            live_job = self._batch.read_namespaced_job(
                name=info["job_name"], namespace=info["namespace"],
            )
        except Exception as exc:
            if self._is_not_found(exc):
                live_job = None
            else:
                logger.warning(
                    "Could not verify Kubernetes Job %s before cancellation: %s",
                    info["job_name"],
                    exc,
                )
                return False
        if live_job is not None and not self._live_job_identity_matches(info, live_job):
            return False
        if live_job is None and info.get("launch_uncertain") is True:
            logger.warning(
                "Kubernetes Job %s was not observed after an ambiguous create; "
                "its API request may still complete late",
                info["job_name"],
            )
            return False
        if live_job is None and not recorded_uid:
            logger.warning(
                "Refusing cancellation for absent Kubernetes Job %s without a "
                "recorded UID",
                info["job_name"],
            )
            return False
        if live_job is not None:
            observed_uid = str(
                getattr(getattr(live_job, "metadata", None), "uid", "") or ""
            )
            if not observed_uid:
                logger.warning(
                    "Refusing cancellation for Kubernetes Job %s without an "
                    "observable UID",
                    info["job_name"],
                )
                return False
            if not recorded_uid:
                # The TAO label was validated above, so this is a safe adoption
                # of a Job created during the durable create-response window.
                info["job_uid"] = observed_uid
                recorded_uid = observed_uid
            info["launch_uncertain"] = False
            observed_status = self.get_tao_job_status(job_id)
            if observed_status in {"Complete", "Error", "Canceled"}:
                info["terminal_status"] = observed_status
                return True

            try:
                delete_options = self._client_module.V1DeleteOptions(
                    propagation_policy="Foreground",
                    preconditions=self._client_module.V1Preconditions(
                        uid=recorded_uid,
                    ),
                )
                self._batch.delete_namespaced_job(
                    name=info["job_name"],
                    namespace=info["namespace"],
                    body=delete_options,
                )
            except Exception as exc:
                if not self._is_not_found(exc):
                    logger.warning(
                        "Could not delete exact Kubernetes Job %s: %s",
                        info["job_name"],
                        exc,
                    )
                    return False

        # A foreground delete or an initial 404 is only provisional. Re-read
        # the exact Job name and inspect every broadly selected pod so a
        # same-name replacement (including one appearing after the first 404)
        # cannot be mistaken for successful cancellation.
        if not self._wait_for_job_and_owned_pods_absent(info):
            return False

        # Multi-node Services are removed last and only through their recorded
        # UID. A replacement Service is never deleted on this Job's behalf.
        if not self._delete_owned_service_after_terminal_cleanup(info):
            return False

        info["terminal_status"] = "Canceled"
        return True

    def delete_terminal_job(self, job_id: str) -> bool:
        """Delete one exact terminal Job and verify its pods are also absent.

        Like ordinary cancellation, terminal cleanup requires the durable
        Kubernetes UID, uses that UID as an API delete precondition, and
        rejects any same-name replacement or label-selected pod that is not
        owned by the recorded Job.
        """
        info = self._jobs.get(job_id)
        if not info:
            return False
        recorded_uid = str(info.get("job_uid", "") or "")
        if not recorded_uid:
            logger.warning(
                "Refusing terminal cleanup for Kubernetes Job %s without a "
                "recorded UID",
                info.get("job_name", job_id),
            )
            return False

        try:
            live_job = self._batch.read_namespaced_job(
                name=info["job_name"], namespace=info["namespace"],
            )
        except Exception as exc:
            if self._is_not_found(exc):
                if not self._wait_for_job_and_owned_pods_absent(info):
                    return False
                return self._delete_owned_service_after_terminal_cleanup(info)
            logger.warning(
                "Could not verify terminal Kubernetes Job %s: %s",
                info["job_name"],
                exc,
            )
            return False

        if not self._live_job_identity_matches(info, live_job):
            return False
        observed_status = self.get_tao_job_status(job_id)
        if observed_status not in {"Complete", "Error", "Canceled"}:
            logger.warning(
                "Refusing cleanup for nonterminal Kubernetes Job %s (%s)",
                info["job_name"],
                observed_status,
            )
            return False
        try:
            pods = self._list_artifact_cleanup_pods(info)
        except Exception as exc:
            logger.warning(
                "Could not verify pods for Kubernetes Job %s: %s",
                info["job_name"],
                exc,
            )
            return False
        if not self._pods_match_recorded_job(info, pods):
            return False
        if not all(self._pod_is_quiescent(pod) for pod in pods):
            logger.warning(
                "Refusing cleanup while a pod for Kubernetes Job %s may write",
                info["job_name"],
            )
            return False

        try:
            delete_options = self._client_module.V1DeleteOptions(
                propagation_policy="Foreground",
                preconditions=self._client_module.V1Preconditions(
                    uid=recorded_uid,
                ),
            )
            self._batch.delete_namespaced_job(
                name=info["job_name"],
                namespace=info["namespace"],
                body=delete_options,
            )
        except Exception as exc:
            if not self._is_not_found(exc):
                logger.warning(
                    "Could not delete exact Kubernetes Job %s: %s",
                    info["job_name"],
                    exc,
                )
                return False

        if not self._wait_for_job_and_owned_pods_absent(info):
            return False
        if not self._delete_owned_service_after_terminal_cleanup(info):
            return False
        info["terminal_status"] = observed_status
        return True

    def _list_artifact_cleanup_pods(self, info: dict) -> list[Any]:
        """List every exact-name pod; identity validation happens separately."""
        response = self._core.list_namespaced_pod(
            namespace=info["namespace"],
            label_selector=f"job-name={info['job_name']}",
        )
        return list(getattr(response, "items", None) or [])

    @staticmethod
    def _pod_is_quiescent(pod: Any) -> bool:
        return (
            str(getattr(getattr(pod, "status", None), "phase", "")).lower()
            not in {"pending", "running", "unknown", ""}
        )

    @staticmethod
    def _pods_match_recorded_job(info: dict, pods: list[Any]) -> bool:
        """Require every selected pod to carry both exact ownership proofs."""
        recorded_uid = str(info.get("job_uid", "") or "")
        expected_job_id = str(info.get("job_id", "") or "")
        expected_job_name = str(info.get("job_name", "") or "")
        if not recorded_uid or not expected_job_id or not expected_job_name:
            return False
        for pod in pods:
            metadata = getattr(pod, "metadata", None)
            labels = getattr(metadata, "labels", None) or {}
            if (
                not isinstance(labels, dict)
                or labels.get("tao-sdk-job-id") != expected_job_id
                or labels.get("job-name") != expected_job_name
            ):
                logger.warning(
                    "A Kubernetes pod label does not match Job %s",
                    expected_job_name,
                )
                return False
            owner_references = list(
                getattr(metadata, "owner_references", None) or []
            )
            if not any(
                str(getattr(owner, "uid", "") or "") == recorded_uid
                and str(getattr(owner, "kind", "") or "").lower() == "job"
                and str(getattr(owner, "name", "") or "") == expected_job_name
                and getattr(owner, "controller", None) is True
                for owner in owner_references
            ):
                logger.warning(
                    "A selected Kubernetes pod is not owned by recorded Job UID %s",
                    recorded_uid,
                )
                return False
        return True

    def _wait_for_job_and_owned_pods_absent(self, info: dict) -> bool:
        """Wait until both the recorded Job and every selected pod are absent."""
        deadline = time.monotonic() + CANCEL_WAIT_SECONDS
        while True:
            try:
                live_job = self._batch.read_namespaced_job(
                    name=info["job_name"], namespace=info["namespace"],
                )
                if not self._live_job_identity_matches(info, live_job):
                    return False
                job_absent = False
            except Exception as exc:
                if not self._is_not_found(exc):
                    logger.warning(
                        "Could not verify deletion of Kubernetes Job %s: %s",
                        info["job_name"],
                        exc,
                    )
                    return False
                job_absent = True
            try:
                pods = self._list_artifact_cleanup_pods(info)
            except Exception as exc:
                logger.warning(
                    "Could not verify pod deletion for Kubernetes Job %s: %s",
                    info["job_name"],
                    exc,
                )
                return False
            if not self._pods_match_recorded_job(info, pods):
                return False
            if job_absent and not pods:
                return True
            if time.monotonic() >= deadline:
                logger.warning(
                    "Timed out waiting for Kubernetes Job %s and its pods "
                    "to be deleted",
                    info["job_name"],
                )
                return False
            time.sleep(0.5)

    def _delete_owned_service_after_terminal_cleanup(self, info: dict) -> bool:
        """Delete a multi-node Service only after its writers are absent."""
        service_name = str(info.get("service_name", "") or "")
        if not service_name:
            return True
        service_uid = str(info.get("service_uid", "") or "")
        if not service_uid:
            logger.warning(
                "Refusing cleanup for Kubernetes Service %s without a "
                "recorded UID",
                service_name,
            )
            return False
        try:
            service = self._core.read_namespaced_service(
                name=service_name, namespace=info["namespace"],
            )
        except Exception as exc:
            if self._is_not_found(exc):
                return True
            logger.warning(
                "Could not verify Kubernetes Service %s: %s",
                service_name,
                exc,
            )
            return False
        if not self._service_identity_matches(info, service):
            return False
        try:
            delete_options = self._client_module.V1DeleteOptions(
                preconditions=self._client_module.V1Preconditions(uid=service_uid),
            )
            self._core.delete_namespaced_service(
                name=service_name,
                namespace=info["namespace"],
                body=delete_options,
            )
        except Exception as exc:
            if not self._is_not_found(exc):
                logger.warning(
                    "Could not delete exact Kubernetes Service %s: %s",
                    service_name,
                    exc,
                )
                return False

        deadline = time.monotonic() + CANCEL_WAIT_SECONDS
        while True:
            try:
                service = self._core.read_namespaced_service(
                    name=service_name, namespace=info["namespace"],
                )
                if not self._service_identity_matches(info, service):
                    return False
            except Exception as exc:
                if self._is_not_found(exc):
                    return True
                logger.warning(
                    "Could not verify deletion of Kubernetes Service %s: %s",
                    service_name,
                    exc,
                )
                return False
            if time.monotonic() >= deadline:
                logger.warning(
                    "Timed out waiting for Kubernetes Service %s deletion",
                    service_name,
                )
                return False
            time.sleep(0.5)

    @staticmethod
    def _service_identity_matches(info: dict, service: Any) -> bool:
        metadata = getattr(service, "metadata", None)
        observed_uid = str(getattr(metadata, "uid", "") or "")
        labels = getattr(metadata, "labels", None) or {}
        matches = (
            observed_uid == str(info.get("service_uid", "") or "")
            and isinstance(labels, dict)
            and labels.get("tao-sdk-job-id") == str(info.get("job_id", "") or "")
        )
        if not matches:
            logger.warning(
                "Kubernetes Service does not match its durable UID and TAO label"
            )
        return matches

    @staticmethod
    def _is_not_found(exc: BaseException) -> bool:
        return getattr(exc, "status", None) == 404

    @staticmethod
    def _is_definitive_create_rejection(exc: BaseException) -> bool:
        """Whether an API response proves no Kubernetes Job was admitted."""
        return getattr(exc, "status", None) in {
            400, 401, 403, 404, 405, 406, 410, 411, 413, 414, 415, 422,
        }

    @staticmethod
    def _live_job_identity_matches(info: dict, job: Any) -> bool:
        metadata = getattr(job, "metadata", None)
        recorded_uid = str(info.get("job_uid", "") or "")
        observed_uid = str(getattr(metadata, "uid", "") or "")
        labels = getattr(metadata, "labels", None) or {}
        expected_job_id = str(info.get("job_id", "") or "")
        if recorded_uid and observed_uid != recorded_uid:
            logger.warning("Kubernetes Job UID does not match its durable identity")
            return False
        if expected_job_id and labels.get("tao-sdk-job-id") != expected_job_id:
            logger.warning("Kubernetes Job label does not match its durable identity")
            return False
        return True

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
