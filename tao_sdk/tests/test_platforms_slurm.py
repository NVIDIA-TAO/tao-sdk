# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the SLURM platform handler.

Validates package wiring, init paths, and credential handling. Real-cluster
behavior (SSH, sbatch, squeue) needs an actual SLURM cluster — out of scope
for unit tests.
"""

from __future__ import annotations

import pytest


def test_handler_class_exposed_at_package_level():
    from tao_sdk.platforms.slurm import SlurmHandler, SlurmSDK
    assert SlurmHandler.__module__ == "tao_sdk.platforms.slurm.handler"
    assert SlurmSDK.__module__ == "tao_sdk.platforms.slurm.sdk"


def test_handler_inherits_from_base():
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.base import BaseHandler
    assert issubclass(SlurmHandler, BaseHandler)


def test_registry_entry_point_resolves_slurm():
    from importlib.metadata import entry_points
    eps = list(entry_points(group="tao_sdk.platforms"))
    slurm_eps = [e for e in eps if e.name == "slurm"]
    assert slurm_eps
    cls = slurm_eps[0].load()
    assert cls.__name__ == "SlurmHandler"


def test_handler_requires_user(monkeypatch):
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    monkeypatch.delenv("SLURM_USER", raising=False)
    monkeypatch.delenv("SSH_USER", raising=False)
    with pytest.raises(ValueError, match="SLURM_USER"):
        SlurmHandler(slurm_user="", slurm_hostnames=["host"])


def test_handler_requires_hostnames():
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    with pytest.raises(ValueError, match="SLURM_HOSTNAME"):
        SlurmHandler(slurm_user="alice", slurm_hostnames=[])


def test_from_credentials_raises_credential_error_when_missing(monkeypatch):
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.models import CredentialError
    monkeypatch.delenv("SLURM_USER", raising=False)
    monkeypatch.delenv("SSH_USER", raising=False)
    monkeypatch.delenv("SLURM_HOSTNAME", raising=False)
    monkeypatch.delenv("SSH_HOST", raising=False)
    with pytest.raises(CredentialError, match="SLURM"):
        SlurmHandler.from_credentials({})


def test_from_credentials_error_message_no_secrets_json(monkeypatch):
    """The 'secrets.json' wording should be gone from the error message."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.models import CredentialError
    monkeypatch.delenv("SLURM_USER", raising=False)
    monkeypatch.delenv("SSH_USER", raising=False)
    monkeypatch.delenv("SLURM_HOSTNAME", raising=False)
    monkeypatch.delenv("SSH_HOST", raising=False)
    try:
        SlurmHandler.from_credentials({})
    except CredentialError as e:
        assert "secrets.json" not in str(e)
        assert ".env" in str(e)
    else:
        pytest.fail("Expected CredentialError")


def test_results_root_prefix_is_lustre_path():
    """SLURM never uses S3 — Lustre is the persistence layer (avoids
    GPU-idle scheduler kill). The prefix gets injected into the container
    as ``TAO_RESULTS_ROOT``; script_runner appends the job id at runtime."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.slurm.sdk import SlurmSDK
    handler = SlurmHandler(
        slurm_user="alice", slurm_hostnames=["login01"],
        base_results_dir="/lustre/fsw/tao",
    )
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    assert sdk._results_root_prefix() == "/lustre/fsw/tao/results"


def test_results_root_prefix_uses_handler_default_when_not_set():
    """Handler auto-fills base_results_dir from SLURM_BASE_RESULTS_DIR or
    a per-user Lustre template — matches TAO Core's convention. SDK never
    raises; just reads what the handler computed."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    from tao_sdk.platforms.slurm.sdk import SlurmSDK
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    prefix = sdk._results_root_prefix()
    assert prefix.endswith("/results")
    assert handler.base_results_dir in prefix


# ---------------------------------------------------------------------------
# Regression tests: single-node Lightning launch + SQSH conversion safety
# (root-caused from the 2026-07-30 DINO-on-SLURM session)
# ---------------------------------------------------------------------------

def _build_script(handler, num_nodes, num_gpus=4, command="bash -c 'echo hi'",
                  network=None):
    return handler._build_slurm_script(
        job_name="tao-job-x",
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        cpus_per_task=16,
        time_hours=4.0,
        account=None,
        partition="polar",
        log_dir="/lustre/logs",
        mail_user=None,
        image="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
        container_mounts="/lustre",
        use_timeout=True,
        timeout_hours=3.8,
        use_requeue=True,
        use_srun=True,
        command=command,
        docker_env_vars={},
        exclusive=False,
        network=network,
    )


_SINGLE_NODE_UNSET = (
    "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
    "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK WORLD_SIZE NODE_RANK "
    "NUM_GPU_PER_NODE MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE"
)

_RANK_LEVEL_UNSET = (
    "unset SLURM_NTASKS SLURM_NNODES SLURM_PROCID SLURM_LOCALID "
    "SLURM_NTASKS_PER_NODE RANK LOCAL_RANK GROUP_RANK"
)


def test_single_node_wrapper_drops_every_distributed_var():
    """TAO PyTorch launches Lightning networks as plain python; multi-GPU
    depends on Lightning self-spawn, which is disabled by SLURM detection and
    by node-injected rank vars (LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 at container
    start make creates_processes_externally True → permanent MEMBER: 1/N
    hang). The single-node wrapper must drop all of them."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    script = _build_script(handler, num_nodes=1)
    # The override rides inside the container command, not the sbatch env,
    # so srun step naming and %x log paths keep the real job name.
    srun_line = next(l for l in script.splitlines() if "srun " in l)
    assert "export SLURM_JOB_NAME=bash" in srun_line
    assert _SINGLE_NODE_UNSET in srun_line


def test_multi_node_wrapper_unsets_rank_vars_keeps_coordination_exports():
    """Multi-node: each node's Lightning must self-spawn NUM_GPU_PER_NODE
    local ranks (LightningEnvironment reads NODE_RANK/MASTER_* from env), so
    rank-level vars are unset — GROUP_RANK included, as LightningEnvironment
    falls back to it for node_rank — while the five coordination exports the
    TAO entrypoint contract needs are preserved."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    multi = _build_script(handler, num_nodes=2)
    assert "export WORLD_SIZE=2" in multi
    assert "export NUM_GPU_PER_NODE=4" in multi
    assert "export NODE_RANK=$SLURM_NODEID" in multi
    assert "export MASTER_PORT=29500" in multi
    assert "export MASTER_ADDR=" in multi

    srun_line = next(l for l in multi.splitlines() if "srun " in l)
    assert "export SLURM_JOB_NAME=bash" in srun_line
    assert _RANK_LEVEL_UNSET in srun_line
    # The coordination vars must never appear in the multi-node unset list.
    unset_clause = srun_line.split("unset ", 1)[1]
    unset_list = unset_clause.split(";", 1)[0]
    for kept in ("WORLD_SIZE", "NUM_GPU_PER_NODE", "NODE_RANK",
                 "MASTER_ADDR", "MASTER_PORT"):
        assert kept not in unset_list


def test_multi_node_wrapper_rederives_node_rank_per_task():
    """The batch script's `export NODE_RANK=$SLURM_NODEID` executes once, on
    the batch host, and its value (0) propagates to every node's task — both
    nodes then log GLOBAL_RANK: 0..G-1 and rendezvous never completes
    (verified on a live 2-node run). The in-container wrapper must re-derive
    NODE_RANK from slurmstepd's per-task SLURM_NODEID, so the batch-level
    export is no longer the only source."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    multi = _build_script(handler, num_nodes=2)
    srun_line = next(l for l in multi.splitlines() if "srun " in l)
    assert "NODE_RANK=${SLURM_NODEID:?}" in srun_line
    # Re-derivation must happen inside the per-task wrapper, before the
    # unsets run and the command starts.
    assert srun_line.index("NODE_RANK=${SLURM_NODEID") < srun_line.index("unset ")

    # Single-node has no node rank to derive — NODE_RANK is dropped instead.
    single = _build_script(handler, num_nodes=1)
    single_srun = next(l for l in single.splitlines() if "srun " in l)
    assert "SLURM_NODEID:?" not in single_srun


def test_slurm_nodeid_is_never_unset_in_any_path():
    """cosmos-rl consumes SLURM_NODEID itself and the multi-node wrapper
    derives NODE_RANK from it — no generated script may unset it."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    for nodes in (1, 2):
        for network in (None, "cosmos-rl"):
            script = _build_script(handler, num_nodes=nodes, network=network)
            for line in script.splitlines():
                if "unset " in line:
                    unset_list = line.split("unset ", 1)[1].split(";", 1)[0]
                    assert "SLURM_NODEID" not in unset_list


def test_cosmos_rl_scripts_are_untouched_in_both_modes():
    """cosmos-rl runs its own launcher (ntasks=1 is by design) and its
    multi-node contract must stay byte-identical: no wrapper, no unsets, no
    CPU scaling."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    for nodes in (1, 2):
        script = _build_script(handler, num_nodes=nodes, network="cosmos-rl")
        assert "SLURM_JOB_NAME=bash" not in script
        assert "unset " not in script
        assert "--no-container-mount-home --export=ALL" in script
        assert "#SBATCH --cpus-per-task=16" in script


def test_single_node_cpus_per_task_scales_with_gpus(monkeypatch):
    """One task carries all local ranks, so the allocation needs base × GPUs
    CPUs (TAO Core parity: 64 for a 4-GPU job), capped configurably."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    assert "#SBATCH --cpus-per-task=64" in _build_script(
        handler, num_nodes=1, num_gpus=4)
    assert "#SBATCH --cpus-per-task=16" in _build_script(
        handler, num_nodes=1, num_gpus=1)
    # Multi-node keeps the per-task base: ranks spread one task per node.
    assert "#SBATCH --cpus-per-task=16" in _build_script(
        handler, num_nodes=2, num_gpus=4)

    monkeypatch.setenv("SLURM_MAX_CPUS_PER_TASK", "32")
    assert "#SBATCH --cpus-per-task=32" in _build_script(
        handler, num_nodes=1, num_gpus=4)


def test_every_non_cosmos_wrapper_unsets_local_rank():
    """Grep-level regression: LOCAL_RANK is the variable that flips Lightning
    to creates_processes_externally; every generated non-cosmos wrapper must
    unset it, in both modes."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    for nodes in (1, 2):
        script = _build_script(handler, num_nodes=nodes, num_gpus=4)
        srun_line = next(l for l in script.splitlines() if "srun " in l)
        unset_list = srun_line.split("unset ", 1)[1].split(";", 1)[0]
        assert "LOCAL_RANK" in unset_list


def test_single_node_wrapper_never_uses_interactive_job_name():
    """Regression: "interactive" is NOT recognized by the Lightning versions
    pinned in TAO containers (verified broken on a live cluster); only the
    literal "bash" is portable across Lightning versions."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])
    script = _build_script(handler, num_nodes=1)
    assert "interactive" not in script


def test_sqsh_conversion_writes_partial_then_renames(monkeypatch):
    """enroot import must never write the final cache path directly: readers
    treat existence as a valid cache hit, so a crash or concurrent writer at
    the final path serves a corrupt image to Pyxis."""
    from types import SimpleNamespace
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    calls = []

    def fake_ssh(cmd, check=False, timeout=None):
        calls.append(cmd)
        if cmd.startswith("squeue"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "enroot import" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="enroot died")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(handler, "_ssh", fake_ssh)
    ok = handler._convert_to_sqsh_via_srun(
        image="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
        sqsh_path="/lustre/cache/x.sqsh",
        sqsh_cache_dir="/lustre/cache",
    )
    assert ok is False

    srun_cmd = next(c for c in calls if "enroot import" in c)
    assert ".partial" in srun_cmd
    assert "hsqs" in srun_cmd          # SquashFS magic gate before publishing
    assert "mv -f" in srun_cmd          # atomic rename onto the cache path
    assert "--job-name=tao-sqsh-" in srun_cmd
    assert "ENROOT_TEMP_PATH" in srun_cmd  # job-scoped temp, not shared /tmp
    # A failed/dropped SSH channel may leave the srun running remotely; the
    # handler must reap it so it cannot race a future conversion.
    assert any(c.startswith("scancel --jobname=tao-sqsh-") for c in calls)


def test_sqsh_conversion_waits_on_in_flight_writer(monkeypatch):
    """A second session must wait on an in-flight conversion of the same
    image instead of starting a concurrent writer."""
    from types import SimpleNamespace
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    calls = []

    def fake_ssh(cmd, check=False, timeout=None):
        calls.append(cmd)
        if cmd.startswith("squeue"):
            return SimpleNamespace(returncode=0, stdout="RUNNING", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(handler, "_ssh", fake_ssh)
    monkeypatch.setattr(handler, "_check_sqsh_exists", lambda p: True)
    ok = handler._convert_to_sqsh_via_srun(
        image="nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
        sqsh_path="/lustre/cache/x.sqsh",
        sqsh_cache_dir="/lustre/cache",
    )
    assert ok is True
    assert not any("enroot import" in c for c in calls)


def test_check_sqsh_exists_requires_squashfs_magic(monkeypatch):
    """Existence alone is not a valid cache hit: legacy or non-atomic writers
    may have left a partial file at the cache path."""
    from types import SimpleNamespace
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])

    seen = {}

    def fake_ssh(cmd, check=False, timeout=None):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="exists", stderr="")

    monkeypatch.setattr(handler, "_ssh", fake_ssh)
    assert handler._check_sqsh_exists("/lustre/cache/x.sqsh") is True
    assert "hsqs" in seen["cmd"]


def test_requeue_block_unchanged_for_timeout_and_node_failure():
    """Requeue stays scheduler-side (timeout 124 → scontrol requeue; NODE_FAIL
    → #SBATCH --requeue). Resume-on-restart is handled inside the entrypoint
    runtime, so the sbatch script needs no spec patching."""
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])
    script = _build_script(handler, num_nodes=1)
    assert "#SBATCH --requeue" in script
    assert "scontrol requeue $SLURM_JOB_ID" in script


# ---------------------------------------------------------------------------
# Backend-identity caching: the cluster identity is stable per login host,
# but was re-fetched over SSH (30s timeout) on every monitored operation.
# ---------------------------------------------------------------------------

_SCONTROL_CONFIG = "ClusterName = cs-oci-ord\nSlurmctldHost = ctl-01\n"


def _identity_handler(monkeypatch, hostnames=("login01", "login02")):
    from types import SimpleNamespace
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=list(hostnames))
    calls = []

    def fake_ssh(cmd, check=False, timeout=None):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=_SCONTROL_CONFIG, stderr="")

    monkeypatch.setattr(handler, "_ssh", fake_ssh)
    return handler, calls


def test_backend_identity_is_cached_after_first_fetch(monkeypatch):
    handler, calls = _identity_handler(monkeypatch)

    first = handler.get_backend_identity()
    second = handler.get_backend_identity()

    assert first.startswith("slurm-sha256:")
    assert second == first
    assert len(calls) == 1, "second call must not SSH to the login node"


def test_hostname_failover_invalidates_identity_cache(monkeypatch):
    handler, calls = _identity_handler(monkeypatch)

    first = handler.get_backend_identity()
    assert handler._switch_to_next_hostname() is True
    second = handler.get_backend_identity()

    assert second == first  # same cluster answered on the failover host
    assert len(calls) == 2, "failover must force a re-fetch, never reuse " \
                            "an identity fetched from a different hostname"


def test_identity_fetch_failure_with_empty_cache_fails_safe(monkeypatch, caplog):
    from types import SimpleNamespace
    from tao_sdk.platforms.slurm.handler import SlurmHandler
    handler = SlurmHandler(slurm_user="alice", slurm_hostnames=["login01"])
    monkeypatch.setattr(
        handler,
        "_ssh",
        lambda cmd, check=False, timeout=None: SimpleNamespace(
            returncode=1, stdout="", stderr="ssh: connect timed out"
        ),
    )

    with caplog.at_level("WARNING"):
        assert handler.get_backend_identity() == ""
    assert any("ClusterName" in r.message for r in caplog.records)


def test_identity_fetch_failure_with_warm_cache_keeps_monitoring_alive(monkeypatch):
    handler, calls = _identity_handler(monkeypatch)
    first = handler.get_backend_identity()

    # Network goes flaky: any further SSH would raise. The warm cache must
    # keep answering without a round-trip, so monitoring never stalls.
    def broken_ssh(cmd, check=False, timeout=None):
        raise TimeoutError("ssh timed out")

    monkeypatch.setattr(handler, "_ssh", broken_ssh)
    assert handler.get_backend_identity() == first

    # After failover the old identity may not describe the new host: the
    # cache must not be served, and the failed fetch fails safe to "".
    assert handler._switch_to_next_hostname() is True
    assert handler.get_backend_identity() == ""
