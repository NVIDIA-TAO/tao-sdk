# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable SDK state-path and SLURM resume identity regressions."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tao_sdk.job_store import JobStore, resolve_sdk_state_paths


def test_default_state_path_is_stable_across_working_directories(
    monkeypatch, tmp_path
):
    state_dir = tmp_path / "durable-state"
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(state_dir))

    monkeypatch.chdir(first_cwd)
    first = resolve_sdk_state_paths()
    monkeypatch.chdir(second_cwd)
    second = resolve_sdk_state_paths()

    assert first == second
    assert first == (
        str(state_dir / "tao_session_state.json"),
        str(state_dir / "tao_session_state.db"),
    )
    store = JobStore()
    try:
        assert store._db_path == first[1]  # noqa: SLF001
    finally:
        store.close()


def test_explicit_state_file_keeps_historical_relative_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(tmp_path / "ignored"))
    assert resolve_sdk_state_paths("relative/session.json") == (
        "relative/session.json",
        "relative/session.db",
    )
    assert resolve_sdk_state_paths("relative/session") == (
        "relative/session",
        "relative/session.db",
    )


def test_relative_tao_sdk_state_dir_is_rejected(monkeypatch):
    monkeypatch.setenv("TAO_SDK_STATE_DIR", "relative-state")
    with pytest.raises(ValueError, match="absolute path"):
        resolve_sdk_state_paths()


def test_default_state_migrates_legacy_cwd_database(monkeypatch, tmp_path):
    legacy_cwd = tmp_path / "legacy-cwd"
    legacy_cwd.mkdir()
    legacy_store = JobStore(legacy_cwd / "tao_session_state.db")
    legacy_store.create_job(
        "legacy-job", "clip", "train", "workspace", backend_type="slurm"
    )
    legacy_store.close()

    durable_dir = tmp_path / "durable"
    monkeypatch.chdir(legacy_cwd)
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(durable_dir))
    migrated = JobStore()
    try:
        assert migrated.get_job("legacy-job")["backend_type"] == "slurm"
    finally:
        migrated.close()


def test_default_state_merges_both_legacy_database_names(monkeypatch, tmp_path):
    legacy_cwd = tmp_path / "legacy-both"
    legacy_cwd.mkdir()

    first_path = legacy_cwd / "tao_session_state.db"
    first = JobStore(first_path)
    first.create_job(
        "shared-job", "clip", "train", "workspace",
        backend_type="slurm", status="Running",
    )
    first.add_checkpoint("shared-job", "/first/shared.pt", epoch=1)
    first.create_job(
        "first-job", "clip", "train", "workspace",
        backend_type="slurm", status="Complete",
    )
    first.close()

    second_path = legacy_cwd / "tao_session.db"
    second = JobStore(second_path)
    second.create_job(
        "shared-job", "clip", "train", "workspace",
        backend_type="slurm", status="Error",
    )
    second.add_checkpoint(
        "shared-job", "/second/must-not-merge.pt", epoch=99
    )
    second.create_job(
        "second-job", "clip", "train", "workspace",
        backend_type="virtualenv", status="Complete",
    )
    second.add_checkpoint("second-job", "/second/epoch-2.pt", epoch=2)
    second.close()

    durable_dir = tmp_path / "durable-both"
    monkeypatch.chdir(legacy_cwd)
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(durable_dir))
    migrated = JobStore()
    try:
        assert migrated.get_job("first-job")["status"] == "Complete"
        assert migrated.get_job("second-job")["backend_type"] == "virtualenv"
        assert migrated.get_job("shared-job")["status"] == "Running"
        assert [
            checkpoint["path"]
            for checkpoint in migrated.get_checkpoints("shared-job")
        ] == ["/first/shared.pt"]
        assert [
            checkpoint["path"]
            for checkpoint in migrated.get_checkpoints("second-job")
        ] == ["/second/epoch-2.pt"]
        assert first_path.is_file()
        assert second_path.is_file()
    finally:
        migrated.close()


def test_legacy_database_merge_rolls_back_and_retries_after_checkpoint_failure(
    monkeypatch, tmp_path,
):
    durable_dir = tmp_path / "durable-atomic"
    durable_dir.mkdir()
    target_path = durable_dir / "tao_session_state.db"
    target = JobStore(target_path)
    target.create_job(
        "existing-job", "clip", "train", "workspace", status="Running"
    )
    target.close()

    legacy_cwd = tmp_path / "legacy-atomic"
    legacy_cwd.mkdir()
    source_path = legacy_cwd / "tao_session_state.db"
    source = JobStore(source_path)
    source.create_job(
        "retry-job", "clip", "train", "workspace",
        backend_type="slurm", status="Complete",
    )
    source.close()

    connection = sqlite3.connect(source_path)
    connection.execute("DROP TABLE checkpoints")
    connection.execute("""
        CREATE TABLE checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            path TEXT,
            epoch INTEGER DEFAULT 0,
            format TEXT DEFAULT 'safetensors',
            created_at TEXT NOT NULL
        )
    """)
    connection.execute(
        """INSERT INTO checkpoints
           (job_id, path, epoch, format, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        ("retry-job", None, 3, "pth", "2026-01-01T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    monkeypatch.chdir(legacy_cwd)
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(durable_dir))
    with pytest.raises(sqlite3.IntegrityError):
        JobStore()

    after_failure = JobStore(target_path)
    try:
        assert after_failure.get_job("existing-job")["status"] == "Running"
        assert after_failure.get_job("retry-job") is None
    finally:
        after_failure.close()

    connection = sqlite3.connect(source_path)
    connection.execute(
        "UPDATE checkpoints SET path = ? WHERE job_id = ?",
        ("/fixed/retry-job/epoch-3.pt", "retry-job"),
    )
    connection.commit()
    connection.close()

    retried = JobStore()
    try:
        assert retried.get_job("retry-job")["status"] == "Complete"
        assert [
            checkpoint["path"]
            for checkpoint in retried.get_checkpoints("retry-job")
        ] == ["/fixed/retry-job/epoch-3.pt"]
    finally:
        retried.close()


def test_default_state_imports_legacy_cwd_json(monkeypatch, tmp_path):
    legacy_cwd = tmp_path / "legacy-json"
    legacy_cwd.mkdir()
    (legacy_cwd / "tao_session_state.json").write_text(json.dumps({
        "active_jobs": [{
            "job_id": "legacy-json-job",
            "network_arch": "clip",
            "action": "train",
            "workspace_id": "workspace",
            "backend_type": "slurm",
            "status": "Running",
        }],
    }))
    durable_dir = tmp_path / "durable-json"
    monkeypatch.chdir(legacy_cwd)
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(durable_dir))

    migrated = JobStore()
    try:
        assert migrated.get_job("legacy-json-job")["status"] == "Running"
        assert (durable_dir / "tao_session_state.json").exists()
    finally:
        migrated.close()


def test_default_state_merges_two_legacy_working_directories_sequentially(
    monkeypatch, tmp_path
):
    durable_dir = tmp_path / "durable-merged"
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(durable_dir))

    first_cwd = tmp_path / "old-cwd-one"
    first_cwd.mkdir()
    first_source = JobStore(first_cwd / "tao_session_state.db")
    first_source.create_job(
        "shared-job", "clip", "train", "workspace",
        backend_type="slurm", status="Running",
    )
    first_source.add_checkpoint(
        "shared-job", "/results/shared/epoch-1.pt", epoch=1
    )
    first_source.close()

    monkeypatch.chdir(first_cwd)
    first_global = JobStore()
    assert first_global.get_job("shared-job")["status"] == "Running"
    first_global.close()

    second_cwd = tmp_path / "old-cwd-two"
    second_cwd.mkdir()
    second_source = JobStore(second_cwd / "tao_session_state.db")
    second_source.create_job(
        "shared-job", "clip", "train", "workspace",
        backend_type="slurm", status="Error",
    )
    second_source.add_checkpoint(
        "shared-job", "/results/shared/must-not-merge.pt", epoch=99
    )
    second_source.create_job(
        "second-job", "clip", "train", "workspace",
        backend_type="slurm", status="Complete",
    )
    second_source.add_checkpoint(
        "second-job", "/results/second/epoch-2.pt", epoch=2
    )
    second_source.close()
    (second_cwd / "tao_session_state.json").write_text(json.dumps({
        "active_jobs": [{
            "job_id": "json-job",
            "network_arch": "clip",
            "action": "train",
            "workspace_id": "workspace",
            "backend_type": "slurm",
            "status": "Pending",
        }],
    }))

    monkeypatch.chdir(second_cwd)
    merged = JobStore()
    try:
        assert merged.get_job("shared-job")["status"] == "Running"
        assert merged.get_job("second-job")["status"] == "Complete"
        assert merged.get_job("json-job")["status"] == "Pending"
        assert [
            checkpoint["path"]
            for checkpoint in merged.get_checkpoints("shared-job")
        ] == ["/results/shared/epoch-1.pt"]
        assert [
            checkpoint["path"]
            for checkpoint in merged.get_checkpoints("second-job")
        ] == ["/results/second/epoch-2.pt"]
    finally:
        merged.close()


def test_default_state_hardens_owned_directory_and_rejects_symlink(
    monkeypatch, tmp_path
):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o777)
    os.chmod(state_dir, 0o777)
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(state_dir))
    resolve_sdk_state_paths()
    assert state_dir.stat().st_mode & 0o077 == 0

    linked = tmp_path / "linked-state"
    linked.symlink_to(state_dir, target_is_directory=True)
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(linked))
    with pytest.raises(ValueError, match="symbolic link"):
        resolve_sdk_state_paths()


def test_default_state_uses_windows_lock_without_posix_apis(tmp_path):
    """The package imports and opens its default DB on a Windows-like host."""
    state_dir = tmp_path / "windows-state"
    code = textwrap.dedent(f"""
        import builtins
        import os
        import subprocess
        import sys
        import types

        real_import = builtins.__import__

        def without_fcntl(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fcntl":
                raise ImportError("simulated Windows host")
            return real_import(name, globals, locals, fromlist, level)

        fake_msvcrt = types.ModuleType("msvcrt")
        fake_msvcrt.LK_NBLCK = 1
        fake_msvcrt.LK_UNLCK = 2
        fake_msvcrt.calls = []

        def locking(file_descriptor, mode, byte_count):
            assert file_descriptor >= 0
            assert byte_count == 1
            fake_msvcrt.calls.append(mode)

        fake_msvcrt.locking = locking
        sys.modules["msvcrt"] = fake_msvcrt
        builtins.__import__ = without_fcntl
        if hasattr(os, "geteuid"):
            delattr(os, "geteuid")

        def unexpected_chmod(*_args, **_kwargs):
            raise AssertionError("Windows default-state setup must not call chmod")

        os.chmod = unexpected_chmod
        os.environ["TAO_SDK_STATE_DIR"] = {str(state_dir)!r}

        import tao_sdk.job_store as job_store

        assert job_store._fcntl is None
        assert job_store._msvcrt is fake_msvcrt
        state_path, db_path = job_store.resolve_sdk_state_paths()
        store = job_store.JobStore()
        try:
            store.create_job("windows-job", "clip", "train", "workspace")
            assert store.get_job("windows-job")["job_id"] == "windows-job"
        finally:
            store.close()
        assert os.path.exists(db_path)
        assert state_path.endswith("tao_session_state.json")
        assert fake_msvcrt.calls.count(fake_msvcrt.LK_NBLCK) >= 2
        assert fake_msvcrt.calls.count(fake_msvcrt.LK_NBLCK) == (
            fake_msvcrt.calls.count(fake_msvcrt.LK_UNLCK)
        )
    """)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.fspath(tmp_path),
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.fspath(Path(__file__).parents[2])},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_concurrent_open_migrates_shared_legacy_schema_once(tmp_path):
    import threading

    db_path = tmp_path / "legacy-schema.db"
    connection = sqlite3.connect(db_path)
    connection.execute("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            network_arch TEXT NOT NULL,
            action TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            backend_type TEXT DEFAULT '',
            backend_job_id TEXT DEFAULT '',
            status TEXT DEFAULT 'Pending',
            parent_job_id TEXT,
            image TEXT DEFAULT '',
            specs TEXT DEFAULT '{}',
            results_dir TEXT DEFAULT '',
            local_results_dir TEXT DEFAULT '',
            base_checkpoint TEXT DEFAULT '',
            last_epoch INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    connection.commit()
    connection.close()

    ready = threading.Barrier(8)

    def open_store(_index):
        ready.wait(timeout=10)
        store = JobStore(db_path)
        try:
            return {
                row["name"]
                for row in store._conn.execute(  # noqa: SLF001
                    "PRAGMA table_info(jobs)"
                ).fetchall()
            }
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(open_store, range(8)))
    for columns in results:
        assert {"plan_id", "step_id", "instance_id"}.issubset(columns)


def test_open_waits_for_reader_during_wal_transition(tmp_path):
    """WAL conversion must tolerate a brief reader lock on a legacy database."""
    import time

    path = tmp_path / "reader.db"
    reader = sqlite3.connect(path)
    reader.execute("CREATE TABLE fixture (value INTEGER)")
    reader.commit()
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM fixture").fetchall()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(JobStore, path)
        time.sleep(0.2)
        reader.rollback()
        reader.close()
        store = future.result(timeout=10)
        try:
            assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            store.close()


def test_monitor_registration_never_replaces_concurrent_terminal_row(tmp_path):
    from tao_sdk.models import Job
    from tao_sdk.monitor import JobMonitor

    store = JobStore(tmp_path / "monitor.db")
    store.create_job(
        "job-1", "clip", "train", "workspace", backend_type="slurm",
        status="Canceled",
    )
    monitor = JobMonitor(
        state_file=str(tmp_path / "monitor.json"),
        store=store,
    )
    pending = Job(
        id="job-1", network_arch="clip", action="train",
        workspace_id="workspace", backend_type="slurm", status="Pending",
    )

    monitor.watch(pending, {}, MagicMock())

    assert store.get_job("job-1")["status"] == "Canceled"
    store.close()


def test_canceling_intent_blocks_stale_active_status_but_allows_terminal(tmp_path):
    store = JobStore(tmp_path / "canceling.db")
    store.create_job("job-1", "", "", "slurm", status="Running")
    assert store.compare_and_set_job_status(
        "job-1", ("Running",), "Canceling"
    )
    assert store.update_job_status("job-1", "Running") is False
    assert store.update_job_status("job-1", "Canceled") is True
    assert store.get_job("job-1")["status"] == "Canceled"
    store.close()


@pytest.mark.parametrize(
    "module_name,class_name,handler_patch",
    (
        (
            "tao_sdk.platforms.docker.sdk",
            "DockerSDK",
            "tao_sdk.platforms.docker.sdk.DockerHandler.from_credentials",
        ),
        (
            "tao_sdk.platforms.kubernetes.sdk",
            "KubernetesSDK",
            "tao_sdk.platforms.kubernetes.sdk.KubernetesHandler.from_credentials",
        ),
        (
            "tao_sdk.platforms.slurm.sdk",
            "SlurmSDK",
            "tao_sdk.platforms.slurm.sdk.SlurmHandler.from_credentials",
        ),
    ),
)
def test_sdk_constructors_use_shared_default_path_resolver(
    module_name, class_name, handler_patch
):
    import importlib

    module = importlib.import_module(module_name)
    handler = MagicMock()
    with patch(handler_patch, return_value=handler), patch.object(
        module,
        "resolve_sdk_state_paths",
        return_value=("/stable/state.json", "/stable/state.db"),
    ) as resolver, patch.object(module, "JobStore") as store_cls, patch.object(
        module, "JobMonitor"
    ) as monitor_cls:
        getattr(module, class_name)()

    resolver.assert_called_once_with(None)
    store_cls.assert_called_once_with(db_path="/stable/state.db")
    assert monitor_cls.call_args.kwargs["state_file"] == "/stable/state.json"


def test_brev_constructor_uses_shared_default_path_resolver():
    from tao_sdk.platforms.brev import sdk as module

    with patch(
        "tao_sdk.platforms.brev.handler.BrevHandler", return_value=MagicMock()
    ), patch("tao_sdk.platforms.brev.cli.BrevCLI"), patch.object(
        module,
        "resolve_sdk_state_paths",
        return_value=("/stable/state.json", "/stable/state.db"),
    ) as resolver, patch.object(module, "JobStore") as store_cls, patch.object(
        module, "JobMonitor"
    ) as monitor_cls:
        module.BrevSDK()

    resolver.assert_called_once_with(None)
    store_cls.assert_called_once_with(db_path="/stable/state.db")
    assert monitor_cls.call_args.kwargs["state_file"] == "/stable/state.json"


def _slurm_handler(monkeypatch, tmp_path, *, state_name="legacy.json"):
    from tao_sdk.platforms.slurm import handler as module

    monkeypatch.setattr(module, "STATE_PATH", tmp_path / state_name)
    handler = module.SlurmHandler(
        slurm_user="scientist",
        slurm_hostnames=["login-1"],
        base_results_dir="/lustre/scientist",
        partition="gpu",
        account="account",
    )
    handler.use_sqsh = False
    handler._scp_text = MagicMock()  # noqa: SLF001
    return handler


def test_slurm_runtime_round_trips_through_job_store_without_sidecar(
    monkeypatch, tmp_path
):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    store = JobStore(tmp_path / "jobs.db")
    handler = _slurm_handler(monkeypatch, tmp_path)
    sdk = object.__new__(SlurmSDK)
    sdk._store = store
    sdk._handler = handler
    sdk._jobs = {}

    runtime = handler.get_job_runtime_identity("job-1")
    store.create_job(
        job_id="job-1",
        network_arch="",
        action="",
        workspace_id="slurm",
        backend_type="slurm",
        backend_job_id="tao-job-job-1",
        specs={"preserved": True, "_slurm_runtime": runtime},
    )
    assert handler.restore_job_runtime("job-1", runtime)
    handler.set_runtime_state_callback(sdk._persist_slurm_runtime)
    handler._jobs["job-1"].update({  # noqa: SLF001
        "slurm_job_id": "42",
        "submission_attempted": True,
        "launch_uncertain": False,
    })
    handler._persist_runtime_state("job-1", required=True)  # noqa: SLF001

    row = store.get_job("job-1")
    specs = json.loads(row["specs"])
    assert specs["preserved"] is True
    assert specs["_slurm_runtime"]["slurm_job_id"] == "42"

    (tmp_path / "legacy.json").unlink(missing_ok=True)
    recovered_handler = _slurm_handler(
        monkeypatch, tmp_path, state_name="missing-sidecar.json"
    )
    recovered_sdk = object.__new__(SlurmSDK)
    recovered_sdk._store = store
    recovered_sdk._handler = recovered_handler
    recovered_sdk._jobs = {}

    job = recovered_sdk._load_job_from_store("job-1")
    assert job is not None
    assert recovered_handler._jobs["job-1"]["slurm_job_id"] == "42"  # noqa: SLF001
    store.close()


def test_slurm_sdk_success_persists_scheduler_id_before_return(
    monkeypatch, tmp_path
):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = _slurm_handler(monkeypatch, tmp_path)
    handler.get_canonical_results_root = MagicMock(
        return_value="/lustre/scientist/results"
    )
    handler.get_backend_identity = MagicMock(return_value="cluster-1")
    handler._ssh = MagicMock(  # noqa: SLF001
        return_value=SimpleNamespace(
            returncode=0,
            stdout="Submitted batch job 42\n",
            stderr="",
        )
    )
    store = JobStore(tmp_path / "sdk-jobs.db")
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    sdk._store = store
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    handler.set_runtime_state_callback(sdk._persist_slurm_runtime)

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        job = sdk.create_job(
            image="example.invalid/tao:test",
            command="true",
            gpu_count=1,
        )

    runtime = SlurmSDK._runtime_from_entry(store.get_job("job-1"))
    assert job.id == "job-1"
    assert runtime["slurm_job_id"] == "42"
    assert runtime["submission_attempted"] is True
    assert runtime["launch_uncertain"] is False
    store.close()


def test_slurm_launch_fence_persistence_failure_prevents_sbatch(
    monkeypatch, tmp_path
):
    handler = _slurm_handler(monkeypatch, tmp_path)
    handler._ssh = MagicMock()  # noqa: SLF001

    def persist(_job_id, runtime):
        if runtime.get("launch_uncertain"):
            raise OSError("state disk full")

    handler.set_runtime_state_callback(persist)
    with pytest.raises(RuntimeError, match="launch fence") as caught:
        handler.create_job(
            image="example.invalid/tao:test",
            network="",
            action="",
            job_id="job-1",
            entrypoint={"command": "true"},
            resource_shape="gpu.1",
        )

    assert caught.value.writer_quiescent is True
    handler._ssh.assert_not_called()  # noqa: SLF001


def test_slurm_accepted_job_is_canceled_if_scheduler_id_cannot_persist(
    monkeypatch, tmp_path
):
    handler = _slurm_handler(monkeypatch, tmp_path)
    handler._ssh = MagicMock(  # noqa: SLF001
        return_value=SimpleNamespace(
            returncode=0,
            stdout="Submitted batch job 42\n",
            stderr="",
        )
    )
    handler.cancel_job = MagicMock(return_value=True)

    def persist(_job_id, runtime):
        if runtime.get("slurm_job_id"):
            raise OSError("state disk full")

    handler.set_runtime_state_callback(persist)
    with pytest.raises(RuntimeError, match="scheduler identity") as caught:
        handler.create_job(
            image="example.invalid/tao:test",
            network="",
            action="",
            job_id="job-1",
            entrypoint={"command": "true"},
            resource_shape="gpu.1",
        )

    assert caught.value.writer_quiescent is True
    handler.cancel_job.assert_called_once_with("job-1")


def _sdk_with_launch_exception(exc):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = MagicMock()
    handler.get_canonical_results_root.return_value = "/lustre/results"
    handler.get_backend_identity.return_value = "cluster-1"
    handler.get_job_name.return_value = "tao-job-job-1"
    handler.get_job_runtime_identity.return_value = {
        "job_name": "tao-job-job-1",
        "slurm_job_id": "",
        "submission_attempted": False,
        "launch_uncertain": False,
    }
    handler.restore_job_runtime.return_value = True
    handler.resolve_resource_shape.return_value = ("gpu.1", None)
    handler.create_job.side_effect = exc
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    sdk._store = MagicMock()
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    return sdk


def test_slurm_exception_cleanup_preserves_observed_completion():
    error = RuntimeError("late client failure")
    error.writer_quiescent = True
    sdk = _sdk_with_launch_exception(error)
    sdk._handler.get_tao_job_status.return_value = None
    sdk._handler.normalize_status.return_value = "Complete"

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(RuntimeError, match="late client failure"):
            sdk.create_job(image="example.invalid/tao:test", command="true")

    sdk._store.update_job_status.assert_called_once_with("job-1", "Complete")


def test_slurm_interrupt_cleanup_preserves_observed_error():
    sdk = _sdk_with_launch_exception(KeyboardInterrupt("interrupted"))
    sdk._handler.cancel_job.return_value = True
    sdk._handler.get_tao_job_status.return_value = "Error"
    sdk._handler.normalize_status.return_value = "Error"

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(KeyboardInterrupt):
            sdk.create_job(image="example.invalid/tao:test", command="true")

    sdk._store.update_job_status.assert_called_once_with("job-1", "Error")


def test_slurm_sbatch_transport_failure_does_not_fail_over(monkeypatch):
    from tao_sdk.platforms.slurm.handler import SlurmHandler

    handler = SlurmHandler.__new__(SlurmHandler)
    handler.login_user = "scientist"
    handler.login_hostnames = ["login-1", "login-2"]
    handler.current_hostname_index = 0
    handler.login_hostname = "login-1"
    handler.ssh_options = []
    failed = subprocess.CompletedProcess(
        args=[],
        returncode=255,
        stdout="",
        stderr="ssh: connection timed out",
    )
    run = MagicMock(return_value=failed)
    monkeypatch.setattr("tao_sdk.platforms.slurm.handler.subprocess.run", run)

    result = handler._ssh(  # noqa: SLF001
        "sbatch /lustre/job.sbatch",
        check=False,
        allow_failover=False,
    )

    assert result is failed
    assert run.call_count == 1
    assert handler.login_hostname == "login-1"


def _slurm_runtime_for_cas(handler, job_id="job-1"):
    runtime = handler.get_job_runtime_identity(job_id)
    runtime.update({
        "slurm_job_id": "42",
        "submission_attempted": True,
        "launch_uncertain": False,
    })
    return runtime


def test_slurm_launch_fence_is_single_owner_and_terminal_status_blocks_retry(
    monkeypatch, tmp_path
):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    db_path = tmp_path / "race.db"
    first_store = JobStore(db_path)
    second_store = JobStore(db_path)
    first_handler = _slurm_handler(monkeypatch, tmp_path, state_name="first.json")
    second_handler = _slurm_handler(monkeypatch, tmp_path, state_name="second.json")
    runtime = _slurm_runtime_for_cas(first_handler)
    first_store.create_job(
        "job-1", "", "", "slurm", backend_type="slurm",
        status="Running", specs={"_slurm_runtime": runtime},
    )
    assert first_handler.restore_job_runtime("job-1", runtime)
    assert second_handler.restore_job_runtime("job-1", runtime)
    first_sdk = object.__new__(SlurmSDK)
    first_sdk._store = first_store
    first_sdk._handler = first_handler
    second_sdk = object.__new__(SlurmSDK)
    second_sdk._store = second_store
    second_sdk._handler = second_handler
    first_handler.set_runtime_state_callback(first_sdk._persist_slurm_runtime)
    second_handler.set_runtime_state_callback(second_sdk._persist_slurm_runtime)

    first_handler._begin_launch_fence("job-1")  # noqa: SLF001
    second_handler.check_path_exists = MagicMock(return_value=True)
    second_handler._ssh = MagicMock()  # noqa: SLF001
    success, _scheduler_id, message = second_handler._resubmit_failed_job(  # noqa: SLF001
        "job-1", failed_slurm_job_id="42", retry_count=0,
    )
    assert success is False
    assert "launch fence" in message
    second_handler._ssh.assert_not_called()  # noqa: SLF001

    first_store.update_job_status("job-1", "Canceled")
    fenced = dict(runtime)
    fenced.update({
        "launch_uncertain": True,
        "launch_token": "late-owner",
        "pre_launch_slurm_job_id": "42",
    })
    with pytest.raises(RuntimeError, match="durable status Canceled"):
        second_sdk._persist_slurm_runtime("job-1", fenced)
    first_store.close()
    second_store.close()


def test_slurm_fenced_retry_ignores_old_failed_scheduler_id(
    monkeypatch, tmp_path
):
    handler = _slurm_handler(monkeypatch, tmp_path)
    runtime = _slurm_runtime_for_cas(handler)
    runtime.update({
        "launch_uncertain": True,
        "launch_token": "retry-token",
        "pre_launch_slurm_job_id": "42",
        "failed_slurm_job_ids": ["42"],
    })
    assert handler.restore_job_runtime("job-1", runtime)
    handler._find_slurm_jobs_by_name = MagicMock(return_value=[  # noqa: SLF001
        ("42", "FAILED"),
        ("43", "PENDING"),
    ])
    handler.get_slurm_job_status = MagicMock(return_value="PENDING")

    assert handler.get_tao_job_status("job-1") == "Pending"
    assert handler._jobs["job-1"]["slurm_job_id"] == "43"  # noqa: SLF001
    assert handler._jobs["job-1"]["launch_uncertain"] is False  # noqa: SLF001
    assert handler._jobs["job-1"]["retry_count"] == 1  # noqa: SLF001


def test_slurm_fenced_cancel_quiesces_every_ambiguous_candidate(
    monkeypatch, tmp_path
):
    handler = _slurm_handler(monkeypatch, tmp_path)
    runtime = _slurm_runtime_for_cas(handler)
    runtime.update({
        "launch_uncertain": True,
        "launch_token": "retry-token",
        "pre_launch_slurm_job_id": "42",
        "failed_slurm_job_ids": ["42"],
    })
    assert handler.restore_job_runtime("job-1", runtime)
    handler._find_slurm_jobs_by_name = MagicMock(return_value=[  # noqa: SLF001
        ("42", "FAILED"),
        ("43", "RUNNING"),
        ("44", "PENDING"),
    ])
    handler._cancel_slurm_job_id = MagicMock(return_value=True)  # noqa: SLF001

    assert handler.cancel_job("job-1") is True
    assert handler._cancel_slurm_job_id.call_args_list == [  # noqa: SLF001
        call("job-1", "43"),
        call("job-1", "44"),
    ]


@pytest.mark.parametrize(
    "module_name,class_name,expected_backend",
    (
        ("tao_sdk.platforms.docker.sdk", "DockerSDK", "docker"),
        ("tao_sdk.platforms.kubernetes.sdk", "KubernetesSDK", "kubernetes"),
        ("tao_sdk.platforms.slurm.sdk", "SlurmSDK", "slurm"),
        ("tao_sdk.platforms.brev.sdk", "BrevSDK", "brev"),
    ),
)
def test_shared_store_loaders_reject_rows_from_other_backends(
    module_name, class_name, expected_backend
):
    import importlib

    cls = getattr(importlib.import_module(module_name), class_name)
    sdk = object.__new__(cls)
    sdk._store = MagicMock()
    sdk._store.get_job.return_value = {
        "job_id": "job-1",
        "backend_type": "different-backend",
        "specs": "{}",
    }
    sdk._handler = MagicMock()
    sdk._jobs = {}

    assert sdk._load_job_from_store("job-1") is None
    sdk._handler.restore_job_runtime.assert_not_called()


def test_slurm_terminal_status_is_available_without_live_cluster_identity(tmp_path):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    store = JobStore(tmp_path / "terminal.db")
    store.create_job(
        "job-1", "", "", "slurm", backend_type="slurm", status="Complete"
    )
    sdk = object.__new__(SlurmSDK)
    sdk._store = store
    sdk._handler = MagicMock()
    sdk._handler.get_backend_identity.side_effect = ConnectionError("offline")
    sdk._monitor = MagicMock()
    sdk._jobs = {}

    assert sdk.get_job_status("job-1").status == "Complete"
    assert sdk.cancel_job("job-1") is True
    sdk._handler.get_backend_identity.assert_not_called()
    store.close()


def test_slurm_cleanup_terminal_observation_disables_retry():
    error = RuntimeError("late client failure")
    error.writer_quiescent = True
    sdk = _sdk_with_launch_exception(error)
    sdk._handler.normalize_status.return_value = "Complete"

    with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
        with pytest.raises(RuntimeError):
            sdk.create_job(image="example.invalid/tao:test", command="true")

    sdk._handler.get_tao_job_status.assert_called_once_with(
        "job-1", allow_retry=False
    )


class _ObservedRLock:
    """Test lock that exposes when a second thread is queued behind it."""

    def __init__(self):
        self._lock = threading.RLock()
        self._guard = threading.Lock()
        self._attempts = 0
        self.second_attempted = threading.Event()

    def __enter__(self):
        with self._guard:
            self._attempts += 1
            if self._attempts == 2:
                self.second_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def test_same_slurm_handler_serializes_concurrent_retry_polls(
    monkeypatch, tmp_path
):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = _slurm_handler(monkeypatch, tmp_path)
    store = JobStore(tmp_path / "same-handler.db")
    sdk = object.__new__(SlurmSDK)
    sdk._store = store
    sdk._handler = handler
    runtime = _slurm_runtime_for_cas(handler)
    store.create_job(
        "job-1", "", "", "slurm", backend_type="slurm",
        status="Running", specs={"_slurm_runtime": runtime},
    )
    assert handler.restore_job_runtime("job-1", runtime)
    handler.set_runtime_state_callback(sdk._persist_slurm_runtime)
    handler.check_path_exists = MagicMock(return_value=True)
    handler.get_slurm_job_status = MagicMock(
        side_effect=lambda scheduler_id: (
            "NODE_FAIL" if str(scheduler_id) == "42" else "PENDING"
        )
    )
    sbatch_calls = []

    def ssh(command, **_kwargs):
        if command.startswith("sbatch "):
            sbatch_calls.append(command)
            return SimpleNamespace(
                returncode=0,
                stdout="Submitted batch job 43\n",
                stderr="",
            )
        raise AssertionError(f"unexpected SSH command: {command}")

    handler._ssh = ssh  # noqa: SLF001
    entered_resubmit = threading.Event()
    release_resubmit = threading.Event()
    original_resubmit = handler._resubmit_failed_job  # noqa: SLF001

    def blocked_resubmit(*args, **kwargs):
        entered_resubmit.set()
        assert release_resubmit.wait(timeout=5)
        return original_resubmit(*args, **kwargs)

    handler._resubmit_failed_job = blocked_resubmit  # noqa: SLF001
    observed_lock = _ObservedRLock()
    handler._lifecycle_locks["job-1"] = observed_lock  # noqa: SLF001

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handler.get_tao_job_status, "job-1")
        assert entered_resubmit.wait(timeout=5)
        second = pool.submit(handler.get_tao_job_status, "job-1")
        assert observed_lock.second_attempted.wait(timeout=5)
        release_resubmit.set()
        assert first.result(timeout=5) == "Pending"
        assert second.result(timeout=5) == "Pending"

    assert len(sbatch_calls) == 1
    assert handler._jobs["job-1"]["slurm_job_id"] == "43"  # noqa: SLF001
    store.close()


def test_canceling_after_slurm_fence_prevents_sbatch(
    monkeypatch, tmp_path
):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = _slurm_handler(monkeypatch, tmp_path)
    handler.get_canonical_results_root = MagicMock(
        return_value="/lustre/scientist/results"
    )
    handler.get_backend_identity = MagicMock(return_value="cluster-1")
    store = JobStore(tmp_path / "cancel-race.db")
    sdk = object.__new__(SlurmSDK)
    sdk._handler = handler
    sdk._store = store
    sdk._monitor = MagicMock()
    sdk._jobs = {}
    handler.set_runtime_state_callback(sdk._persist_slurm_runtime)

    fence_persisted = threading.Event()
    release_confirmation = threading.Event()
    original_confirm = handler._confirm_launch_fence  # noqa: SLF001

    def blocked_confirm(job_id, launch_token):
        fence_persisted.set()
        assert release_confirmation.wait(timeout=5)
        return original_confirm(job_id, launch_token)

    handler._confirm_launch_fence = blocked_confirm  # noqa: SLF001
    commands = []

    def ssh(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    handler._ssh = ssh  # noqa: SLF001
    cancel_claimed = threading.Event()
    original_claim = sdk._claim_canceling

    def observed_claim(job_id):
        result = original_claim(job_id)
        cancel_claimed.set()
        return result

    sdk._claim_canceling = observed_claim

    def create():
        with patch("tao_sdk.platforms.slurm.sdk.uuid.uuid4", return_value="job-1"):
            return sdk.create_job(
                image="example.invalid/tao:test",
                command="true",
                gpu_count=1,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        create_future = pool.submit(create)
        assert fence_persisted.wait(timeout=5)
        cancel_future = pool.submit(sdk.cancel_job, "job-1")
        assert cancel_claimed.wait(timeout=5)
        release_confirmation.set()
        with pytest.raises(RuntimeError, match="launch fence"):
            create_future.result(timeout=5)
        cancel_future.result(timeout=5)

    assert not any(command.startswith("sbatch ") for command in commands)
    assert store.get_job("job-1")["status"] in {"Canceling", "Error"}
    store.close()


def test_canceling_after_final_fence_check_cancels_accepted_slurm_job(
    monkeypatch, tmp_path
):
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = _slurm_handler(monkeypatch, tmp_path)
    store = JobStore(tmp_path / "accepted-cancel-race.db")
    sdk = object.__new__(SlurmSDK)
    sdk._store = store
    sdk._handler = handler
    runtime = handler.get_job_runtime_identity("job-1")
    store.create_job(
        "job-1", "", "", "slurm", backend_type="slurm",
        status="Pending", specs={"_slurm_runtime": runtime},
    )
    assert handler.restore_job_runtime("job-1", runtime)
    handler.set_runtime_state_callback(sdk._persist_slurm_runtime)

    sbatch_entered = threading.Event()
    release_sbatch = threading.Event()
    scancel_issued = threading.Event()
    scheduler_status_reads = 0
    sbatch_count = 0

    def ssh(command, **_kwargs):
        nonlocal scheduler_status_reads, sbatch_count
        if command.startswith("sbatch "):
            sbatch_count += 1
            sbatch_entered.set()
            assert release_sbatch.wait(timeout=5)
            return SimpleNamespace(
                returncode=0,
                stdout="Submitted batch job 42\n",
                stderr="",
            )
        if command.startswith("squeue --name=") or command.startswith(
            "sacct -nP -X --name="
        ):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command.startswith("squeue --job=42"):
            scheduler_status_reads += 1
            state = "RUNNING" if scheduler_status_reads == 1 else "CANCELLED"
            return SimpleNamespace(returncode=0, stdout=f"{state}\n", stderr="")
        if command == "scancel 42":
            scancel_issued.set()
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected SSH command: {command}")

    handler._ssh = ssh  # noqa: SLF001

    def create():
        return handler.create_job(
            image="example.invalid/tao:test",
            network="",
            action="",
            job_id="job-1",
            entrypoint={"command": "true"},
            resource_shape="gpu.1",
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(create)
        assert sbatch_entered.wait(timeout=5)
        assert store.compare_and_set_job_status(
            "job-1", ("Pending",), "Canceling"
        )
        release_sbatch.set()
        with pytest.raises(RuntimeError, match="scheduler identity") as caught:
            future.result(timeout=5)

    assert caught.value.writer_quiescent is True
    assert sbatch_count == 1
    assert scancel_issued.is_set()
    assert store.get_job("job-1")["status"] == "Canceling"
    store.close()


def test_slurm_cold_load_cannot_restore_over_newer_retry_generation(
    monkeypatch, tmp_path
):
    from tao_sdk.artifacts import artifact_specs
    from tao_sdk.platforms.slurm.sdk import SlurmSDK

    handler = _slurm_handler(monkeypatch, tmp_path)
    handler.get_backend_identity = MagicMock(return_value="cluster-1")
    old_runtime = handler.get_job_runtime_identity("job-1")
    old_runtime.update({
        "slurm_job_id": "42",
        "submission_attempted": True,
        "launch_token": "generation-one",
        "revision": 1,
    })
    newer_runtime = dict(old_runtime)
    newer_runtime.update({
        "slurm_job_id": "43",
        "failed_slurm_job_ids": ["42"],
        "retry_count": 1,
        "launch_token": "generation-two",
        "pre_launch_slurm_job_id": "42",
        "revision": 2,
    })
    store = JobStore(tmp_path / "cold-load-race.db")
    specs = artifact_specs(
        "lustre",
        "lustre:///lustre/scientist/results/job-1",
        backend_identity="cluster-1",
    )
    specs["_slurm_runtime"] = old_runtime
    store.create_job(
        "job-1", "", "", "slurm", backend_type="slurm",
        status="Running", specs=specs,
    )
    assert handler.restore_job_runtime("job-1", old_runtime)
    sdk = object.__new__(SlurmSDK)
    sdk._store = store
    sdk._handler = handler
    sdk._jobs = {}

    real_get_job = store.get_job
    stale_entry = dict(real_get_job("job-1"))
    stale_read_started = threading.Event()
    release_stale_read = threading.Event()
    first_read = True

    def blocking_get_job(job_id):
        nonlocal first_read
        if first_read:
            first_read = False
            stale_read_started.set()
            assert release_stale_read.wait(timeout=5)
            return dict(stale_entry)
        return real_get_job(job_id)

    store.get_job = blocking_get_job
    observed_lock = _ObservedRLock()
    handler._lifecycle_locks["job-1"] = observed_lock  # noqa: SLF001

    def advance_retry_generation():
        with handler._job_lifecycle_lock("job-1"):  # noqa: SLF001
            current_entry = real_get_job("job-1")
            current_specs = json.loads(current_entry["specs"])
            replacement = dict(current_specs)
            replacement["_slurm_runtime"] = newer_runtime
            assert store.update_job_specs("job-1", replacement)
            assert handler._restore_job_runtime_locked(  # noqa: SLF001
                "job-1", newer_runtime
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        cold_load = pool.submit(sdk._load_job_from_store, "job-1")
        assert stale_read_started.wait(timeout=5)
        retry = pool.submit(advance_retry_generation)
        assert observed_lock.second_attempted.wait(timeout=5)
        release_stale_read.set()
        assert cold_load.result(timeout=5) is not None
        retry.result(timeout=5)

    assert handler._jobs["job-1"]["revision"] == 2  # noqa: SLF001
    assert handler._jobs["job-1"]["slurm_job_id"] == "43"  # noqa: SLF001
    legacy_snapshot = dict(old_runtime)
    legacy_snapshot.pop("revision")
    assert handler.restore_job_runtime("job-1", legacy_snapshot)
    assert handler._jobs["job-1"]["revision"] == 2  # noqa: SLF001
    assert handler._jobs["job-1"]["slurm_job_id"] == "43"  # noqa: SLF001
    handler.get_slurm_job_status = MagicMock(return_value="PENDING")
    status = sdk.get_job_status("job-1")
    assert status.status == "Pending"
    handler.get_slurm_job_status.assert_called_once_with("43")
    store.close()
