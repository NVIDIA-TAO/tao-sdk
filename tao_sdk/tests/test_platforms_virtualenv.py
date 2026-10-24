# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for direct virtual-environment Python jobs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import venv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tao_sdk.job_store import JobStore
from tao_sdk.platforms.virtualenv import VirtualEnvSDK
from tao_sdk.platforms.virtualenv.sdk import _JOB_WRAPPER_SOURCE


@pytest.fixture(scope="module")
def real_venv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("virtualenv-sdk") / "venv"
    venv.EnvBuilder(with_pip=False).create(path)
    return path


@pytest.fixture
def sdk(real_venv, tmp_path) -> VirtualEnvSDK:
    return VirtualEnvSDK(
        venv_path=real_venv,
        work_dir=tmp_path / "work",
        state_file=tmp_path / "state.db",
    )


def _write_script(tmp_path: Path, source: str, name: str = "train.py") -> Path:
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    return script


def _wait_for_terminal(sdk: VirtualEnvSDK, job_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = sdk.get_job_status(job_id)
        if status.status in {"Complete", "Error", "Canceled"}:
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal status")


def _launch_from_short_lived_process(
    real_venv: Path, work_dir: Path, state_file: Path, script: Path,
) -> str:
    source = (
        "from tao_sdk.platforms.virtualenv import VirtualEnvSDK\n"
        f"sdk = VirtualEnvSDK({str(real_venv)!r}, "
        f"work_dir={str(work_dir)!r}, state_file={str(state_file)!r})\n"
        f"job = sdk.create_python_job(script={str(script)!r}, gpu_count=0)\n"
        "print(job.id, flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def test_package_exports_python_script_capability():
    from tao_sdk import PythonScriptSDK, VirtualEnvSDK as ExportedSDK
    from tao_sdk.platforms import PythonScriptSDK as PlatformCapability

    assert ExportedSDK is VirtualEnvSDK
    assert PlatformCapability is PythonScriptSDK


def test_default_state_and_jobs_are_stable_across_working_directories(
    real_venv, tmp_path, monkeypatch,
):
    state_root = tmp_path / "sdk-state"
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(state_root))

    monkeypatch.chdir(first_cwd)
    first = VirtualEnvSDK(real_venv)
    first._store.create_job(
        "job-stable",
        "",
        "train",
        "local",
        backend_type="virtualenv",
        status="Complete",
    )

    monkeypatch.chdir(second_cwd)
    second = VirtualEnvSDK(real_venv)
    try:
        assert first.work_dir == state_root / "virtualenv"
        assert second.work_dir == first.work_dir
        assert [job["job_id"] for job in second.list_jobs()] == ["job-stable"]
    finally:
        first._store.close()
        second._store.close()


def test_default_state_merges_legacy_virtualenv_database(
    real_venv, tmp_path, monkeypatch,
):
    state_root = tmp_path / "sdk-state"
    stable_db = state_root / "virtualenv" / "tao_session.db"
    stable_db.parent.mkdir(parents=True)
    stable = JobStore(stable_db)
    stable.create_job(
        "shared-job", "clip", "train", "local",
        backend_type="virtualenv", status="Running",
    )
    stable.add_checkpoint(
        "shared-job", "/stable/shared/epoch-1.pt", epoch=1
    )
    stable.close()

    legacy_cwd = tmp_path / "legacy-cwd"
    legacy_db = legacy_cwd / ".tao" / "virtualenv" / "tao_session.db"
    legacy_db.parent.mkdir(parents=True)
    legacy = JobStore(legacy_db)
    legacy.create_job(
        "shared-job", "clip", "train", "local",
        backend_type="virtualenv", status="Error",
    )
    legacy.add_checkpoint(
        "shared-job", "/legacy/shared/must-not-merge.pt", epoch=99
    )
    legacy.create_job(
        "legacy-job", "clip", "train", "local",
        backend_type="virtualenv", backend_job_id="4242",
        status="Complete", local_results_dir="/legacy/results/legacy-job",
    )
    legacy.add_checkpoint(
        "legacy-job", "/legacy/results/legacy-job/epoch-2.pt", epoch=2
    )
    legacy.close()

    monkeypatch.setenv("TAO_SDK_STATE_DIR", str(state_root))
    monkeypatch.chdir(legacy_cwd)
    migrated = VirtualEnvSDK(real_venv)
    try:
        assert migrated._store.get_job("shared-job")["status"] == "Running"
        assert [
            checkpoint["path"]
            for checkpoint in migrated._store.get_checkpoints("shared-job")
        ] == ["/stable/shared/epoch-1.pt"]

        restored = migrated._store.get_job("legacy-job")
        assert restored["backend_job_id"] == "4242"
        assert restored["status"] == "Complete"
        assert restored["local_results_dir"] == "/legacy/results/legacy-job"
        assert [
            checkpoint["path"]
            for checkpoint in migrated._store.get_checkpoints("legacy-job")
        ] == ["/legacy/results/legacy-job/epoch-2.pt"]
        assert legacy_db.is_file()
    finally:
        migrated._store.close()


def test_real_venv_job_routes_nested_outputs_and_preserves_argv(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        """
import json
import os
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output_dir = Path(spec["train"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "argv": sys.argv[2:],
    "executable": sys.executable,
    "virtual_env": os.environ["VIRTUAL_ENV"],
    "path_head": os.environ["PATH"].split(os.pathsep)[0],
    "job_id": os.environ["TAO_JOB_ID"],
    "results_root": os.environ["TAO_RESULTS_ROOT"],
    "output_dir": str(output_dir),
}
(output_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
print("metric: 0.125", flush=True)
""",
    )
    original_specs = {"train": {"epochs": 1, "output_dir": "ignored"}}
    malicious_token = "$(touch should-not-exist)"

    job = sdk.create_python_job(
        script=script,
        specs=original_specs,
        config_format="json",
        script_args=[
            "{config_path}", "token with spaces", malicious_token,
            "{job_id}", "{results_dir}",
        ],
        outputs={"train.output_dir": {"type": "folder"}},
        network_arch="tiny-public-model",
        action="train",
        gpu_count=0,
    )

    assert job.backend_type == "virtualenv"
    assert job.status == "Running"
    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    assert original_specs["train"]["output_dir"] == "ignored"

    results_dir = Path(sdk.get_job_results_dir(job.id))
    result_file = results_dir / "train_output_dir" / "result.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["argv"] == [
        "token with spaces", malicious_token, job.id, str(results_dir),
    ]
    assert Path(payload["executable"]).resolve() == sdk.python_path.resolve()
    assert payload["virtual_env"] == str(sdk.venv_path)
    assert payload["path_head"] == str(sdk.venv_path / "bin")
    assert payload["job_id"] == job.id
    assert payload["results_root"] == str(results_dir)
    assert payload["output_dir"] == str(results_dir / "train_output_dir")
    assert not (tmp_path / "should-not-exist").exists()

    stored = sdk.get_job(job.id)
    assert stored["status"] == "Complete"
    assert stored["network_arch"] == "tiny-public-model"
    assert sdk.list_jobs(status="Complete")[0]["job_id"] == job.id
    assert sdk.list_results(job.id) == ["train_output_dir/result.json"]
    assert sdk.get_results(job.id) == sdk.list_results(job.id)
    assert json.loads(sdk.read_job_result_file(
        job.id, "train_output_dir/result.json"
    ))["job_id"] == job.id
    assert "metric: 0.125" in sdk.get_job_logs(job.id)


def test_yaml_nested_spec_serialization(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        """
import sys
from pathlib import Path
print(Path(sys.argv[1]).read_text(encoding="utf-8"), flush=True)
""",
    )
    job = sdk.create_python_job(
        script=script,
        specs={"train": {"epochs": 3, "enabled": True}},
        config_format="yaml",
        script_args=["{config_path}"],
        gpu_count=0,
    )
    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    logs = sdk.get_job_logs(job.id)
    assert "train:" in logs
    assert "epochs: 3" in logs
    assert "enabled: true" in logs


def test_submission_normalizes_and_persists_specs_without_mutating_caller(
    sdk, tmp_path,
):
    script = _write_script(tmp_path, "print('normalized', flush=True)\n")
    specs = {
        "train": {
            "epochs": np.int64(3),
            "lr": np.float64(0.125),
            "shape": np.array([16, 32]),
            "data_path": tmp_path / "data",
        },
        "results_dir": "caller-value",
    }

    job = sdk.create_python_job(
        script=script,
        specs=specs,
        config_format="json",
        outputs={"results_dir": {"type": "folder"}},
        gpu_count=0,
    )
    assert _wait_for_terminal(sdk, job.id).status == "Complete"

    entry = sdk.get_job(job.id)
    stored = json.loads(entry["specs"])
    config_path = Path(entry["local_results_dir"]).parent / "spec" / "spec.json"
    assert json.loads(config_path.read_text(encoding="utf-8")) == stored
    assert stored["train"] == {
        "epochs": 3,
        "lr": 0.125,
        "shape": [16, 32],
        "data_path": str(tmp_path / "data"),
    }
    assert stored["results_dir"] == str(
        Path(entry["local_results_dir"]) / "results_dir"
    )
    assert isinstance(specs["train"]["epochs"], np.integer)
    assert isinstance(specs["train"]["shape"], np.ndarray)
    assert specs["results_dir"] == "caller-value"


def test_toml_output_null_is_routed_before_serialization(sdk, tmp_path):
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib

    script = _write_script(tmp_path, "print('routed toml', flush=True)\n")
    specs = {"results_dir": None}

    job = sdk.create_python_job(
        script=script,
        specs=specs,
        config_format="toml",
        outputs={"results_dir": {"type": "folder"}},
        gpu_count=0,
    )

    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    entry = sdk.get_job(job.id)
    expected_output = str(Path(entry["local_results_dir"]) / "results_dir")
    config_path = Path(entry["local_results_dir"]).parent / "spec" / "spec.toml"
    assert tomllib.loads(config_path.read_text(encoding="utf-8")) == {
        "results_dir": expected_output,
    }
    assert json.loads(entry["specs"])["results_dir"] == expected_output
    assert specs["results_dir"] is None


def test_remote_declared_input_is_rejected_before_launch(sdk, tmp_path):
    script = _write_script(tmp_path, "print('must not run')\n")
    specs = {
        "dataset": {
            "sources": [{"images": "s3://public-bucket/images"}],
        },
    }
    with pytest.raises(ValueError, match="local declared inputs only"):
        sdk.create_python_job(
            script=script,
            specs=specs,
            inputs={"dataset.sources[0].images": {"type": "folder"}},
        )
    assert sdk.list_jobs() == []


def test_local_file_uri_is_not_treated_as_remote(sdk, tmp_path):
    script = _write_script(tmp_path, "print('local input accepted', flush=True)\n")
    local_input = tmp_path / "dataset.json"
    local_input.write_text("{}", encoding="utf-8")

    job = sdk.create_python_job(
        script=script,
        specs={"dataset": {"path": local_input.as_uri()}},
        inputs={"dataset.path": {"type": "file"}},
        gpu_count=0,
    )

    assert _wait_for_terminal(sdk, job.id).status == "Complete"


def test_failure_status_analysis_and_log_tail(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        "import sys\nprint('CUDA out of memory', flush=True)\nsys.exit(7)\n",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)
    status = _wait_for_terminal(sdk, job.id)

    assert status.status == "Error"
    assert "code 7" in status.message
    analysis = sdk.get_failure_analysis(job.id)
    assert analysis["category"] == "out_of_memory"
    assert analysis["return_code"] == 7
    assert "CUDA out of memory" in analysis["log_tail"]
    assert sdk.get_job_logs(job.id, tail=1) == "CUDA out of memory\n"


def test_cancel_terminates_process_group(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        """
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(f"child_pid={child.pid}", flush=True)
time.sleep(60)
""",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)

    deadline = time.monotonic() + 5
    child_pid = None
    while time.monotonic() < deadline:
        logs = sdk.get_job_logs(job.id)
        if "child_pid=" in logs:
            child_pid = int(logs.split("child_pid=", 1)[1].splitlines()[0])
            break
        time.sleep(0.02)
    assert child_pid is not None
    assert os.getpgid(child_pid) == int(job.backend_job_id)

    assert sdk.cancel_job(job.id, timeout=1.0) is True
    assert _wait_for_terminal(sdk, job.id).status == "Canceled"
    assert sdk.cancel_job(job.id) is False
    assert sdk.get_failure_analysis(job.id) is None


def test_cancel_failure_stays_nonterminal_and_blocks_artifact_deletion(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        "import time\ntime.sleep(60)\n",
        name="cancel_failure.py",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)

    with patch.object(
        sdk,
        "_terminate_process_group",
        side_effect=RuntimeError("process group survived"),
    ):
        assert not sdk.cancel_job(job.id, timeout=0.1)

    assert sdk._store.get_job(job.id)["status"] == "Canceling"
    assert not sdk.delete_job_artifacts(job.id)

    assert sdk.cancel_job(job.id, timeout=0.2)
    assert sdk.get_job_status(job.id).status == "Canceled"


def test_natural_completion_waits_for_background_process_group(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        """
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(f"background_pid={child.pid}", flush=True)
""",
        name="background_writer.py",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)

    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    child_pid = int(
        sdk.get_job_logs(job.id).split("background_pid=", 1)[1].splitlines()[0]
    )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()


def test_gpu_ids_are_explicit_and_count_must_match(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        "import os\nprint(os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'), flush=True)\n",
    )
    with pytest.raises(ValueError, match="must match"):
        sdk.create_python_job(script=script, gpu_count=2, gpu_ids=[3])

    job = sdk.create_python_job(script=script, gpu_count=2, gpu_ids="3,5")
    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    assert sdk.get_job_logs(job.id).strip() == "3,5"


def test_gpu_ids_accept_integral_scalar_types():
    assert VirtualEnvSDK._validate_gpu_request(
        2, [np.int64(3), np.uint32(5)]
    ) == [3, 5]


@pytest.mark.parametrize(
    "gpu_id",
    [True, False, 1.0, 1.5, np.float64(2.0), "1.5"],
)
def test_gpu_ids_reject_bool_and_non_integral_values(gpu_id):
    with pytest.raises(ValueError, match="non-negative integers"):
        VirtualEnvSDK._validate_gpu_request(1, [gpu_id])


@pytest.mark.parametrize("specs", [[], (), "", 0, False])
def test_non_dictionary_specs_are_rejected_without_job_artifacts(
    sdk, tmp_path, specs,
):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())

    with pytest.raises(TypeError, match="dictionary or None"):
        sdk.create_python_job(script=script, specs=specs, gpu_count=0)

    assert set(sdk.jobs_dir.iterdir()) == before


def test_output_routing_failure_removes_job_artifacts(sdk, tmp_path):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())

    with pytest.raises(ValueError, match="missing list element"):
        sdk.create_python_job(
            script=script,
            specs={"items": []},
            outputs={"items[0].path": {"type": "folder"}},
            gpu_count=0,
        )

    assert set(sdk.jobs_dir.iterdir()) == before
    assert sdk.list_jobs() == []


def test_spec_write_failure_removes_job_artifacts(sdk, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())

    def fail_write(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(sdk, "_write_spec", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        sdk.create_python_job(script=script, gpu_count=0)

    assert set(sdk.jobs_dir.iterdir()) == before
    assert sdk.list_jobs() == []


def test_wrapper_write_failure_removes_job_artifacts(sdk, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())
    real_write_text = Path.write_text

    def fail_wrapper_write(path, *args, **kwargs):
        if path.name == "launch_job.py":
            raise OSError("wrapper disk unavailable")
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_wrapper_write)

    with pytest.raises(OSError, match="wrapper disk unavailable"):
        sdk.create_python_job(script=script, gpu_count=0)

    assert set(sdk.jobs_dir.iterdir()) == before
    assert sdk.list_jobs() == []


def test_store_creation_failure_removes_job_artifacts(sdk, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())

    def fail_create_job(*args, **kwargs):
        raise RuntimeError("database unavailable before insert")

    monkeypatch.setattr(sdk._store, "create_job", fail_create_job)

    with pytest.raises(RuntimeError, match="database unavailable before insert"):
        sdk.create_python_job(script=script, gpu_count=0)

    assert set(sdk.jobs_dir.iterdir()) == before
    assert sdk.list_jobs() == []


def test_invalid_format_is_rejected_without_job_artifacts(sdk, tmp_path):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())

    with pytest.raises(ValueError, match="config_format"):
        sdk.create_python_job(script=script, config_format="ini", gpu_count=0)

    assert set(sdk.jobs_dir.iterdir()) == before
    assert sdk.list_jobs() == []


def test_completed_job_is_available_from_persistent_store(real_venv, tmp_path):
    work_dir = tmp_path / "persistent"
    state_file = tmp_path / "persistent.db"
    first = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    script = _write_script(tmp_path, "print('persisted', flush=True)\n")
    job = first.create_python_job(script=script, gpu_count=0)
    assert _wait_for_terminal(first, job.id).status == "Complete"

    second = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    assert second.get_job_status(job.id).status == "Complete"
    assert second.get_job(job.id)["backend_type"] == "virtualenv"
    assert "persisted" in second.get_job_logs(job.id)


def test_running_job_completes_after_orchestrator_restart(real_venv, tmp_path):
    work_dir = tmp_path / "recovered"
    state_file = tmp_path / "recovered.db"
    script = _write_script(
        tmp_path,
        "import time\ntime.sleep(0.5)\nprint('recovered-ok', flush=True)\n",
    )
    job_id = _launch_from_short_lived_process(
        real_venv, work_dir, state_file, script
    )

    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    assert recovered.get_job_status(job_id).status in {
        "Pending", "Running", "Complete",
    }
    assert _wait_for_terminal(recovered, job_id).status == "Complete"
    assert "recovered-ok" in recovered.get_job_logs(job_id)


def test_recovered_running_job_can_be_canceled(real_venv, tmp_path):
    work_dir = tmp_path / "recovered-cancel"
    state_file = tmp_path / "recovered-cancel.db"
    script = _write_script(
        tmp_path,
        "import time\nprint('running', flush=True)\ntime.sleep(60)\n",
    )
    job_id = _launch_from_short_lived_process(
        real_venv, work_dir, state_file, script
    )

    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    assert recovered.get_job_status(job_id).status == "Running"
    assert recovered.cancel_job(job_id, timeout=1.0)
    assert recovered.get_job_status(job_id).status == "Canceled"


def test_post_spawn_registration_failure_reaps_process(
    sdk, tmp_path, monkeypatch,
):
    from tao_sdk.platforms.virtualenv import sdk as sdk_module

    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    spawned = []
    real_popen = subprocess.Popen

    def tracking_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)  # pylint: disable=consider-using-with
        spawned.append(process)
        return process

    def fail_update_backend(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(sdk_module.subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(sdk._store, "update_job_backend_id", fail_update_backend)

    with pytest.raises(RuntimeError, match="database unavailable"):
        sdk.create_python_job(script=script, gpu_count=0)

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert sdk._processes == {}


def test_exception_immediately_after_spawn_is_rolled_back(
    sdk, tmp_path, monkeypatch,
):
    from tao_sdk.platforms.virtualenv import sdk as sdk_module

    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    spawned = []
    real_popen = subprocess.Popen

    def tracking_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)  # pylint: disable=consider-using-with
        spawned.append(process)
        return process

    def fail_marker(pid):
        raise RuntimeError("marker read failed")

    monkeypatch.setattr(sdk_module.subprocess, "Popen", tracking_popen)
    monkeypatch.setattr(sdk_module, "_process_start_marker", fail_marker)

    with pytest.raises(RuntimeError, match="marker read failed") as raised:
        sdk.create_python_job(script=script, gpu_count=0)

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert raised.value.tao_job_id == sdk.list_jobs()[0]["job_id"]
    assert raised.value.tao_launch_uncertain is False
    assert sdk.list_jobs()[0]["status"] == "Error"


def test_failed_post_spawn_rollback_retains_recoverable_process_identity(
    sdk, tmp_path, monkeypatch,
):
    from tao_sdk.platforms.virtualenv import sdk as sdk_module

    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    real_terminate = sdk._terminate_process_group

    def fail_marker(_pid):
        raise RuntimeError("marker read failed")

    def fail_termination(*_args, **_kwargs):
        raise RuntimeError("termination not confirmed")

    monkeypatch.setattr(sdk_module, "_process_start_marker", fail_marker)
    monkeypatch.setattr(sdk, "_terminate_process_group", fail_termination)

    with pytest.raises(RuntimeError, match="marker read failed") as raised:
        sdk.create_python_job(script=script, gpu_count=0)

    job_id = raised.value.tao_job_id
    process = sdk._processes[job_id]
    entry = sdk._store.get_job(job_id)
    metadata = sdk._read_metadata_from_entry(entry)
    assert entry["backend_job_id"] == str(process.pid)
    assert entry["status"] == "Canceling"
    assert metadata["status"] == "Canceling"
    assert metadata["pid"] == process.pid
    assert Path(metadata["cancel_path"]).exists()

    real_terminate(process.pid, process=process, timeout=0.2)


def test_invalid_hooks_are_rejected_before_process_launch(sdk, tmp_path):
    script = _write_script(tmp_path, "print('must not launch')\n")
    before = set(sdk.jobs_dir.iterdir())

    with pytest.raises(TypeError, match="hooks must be a dictionary"):
        sdk.create_python_job(script=script, hooks="on_complete")

    assert set(sdk.jobs_dir.iterdir()) == before


def test_concurrent_submissions_are_registered_without_sqlite_races(sdk, tmp_path):
    script = _write_script(tmp_path, "print('complete', flush=True)\n")

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(executor.map(
            lambda _: sdk.create_python_job(script=script, gpu_count=0),
            range(20),
        ))

    for job in jobs:
        assert _wait_for_terminal(sdk, job.id).status == "Complete"
    assert len(sdk.list_jobs()) == 20


def test_log_tail_reads_requested_suffix(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        "for index in range(20000): print(f'line-{index}', flush=True)\n",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)
    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    assert sdk.get_job_logs(job.id, tail=3).splitlines() == [
        "line-19997", "line-19998", "line-19999",
    ]


def test_log_tail_is_byte_bounded_for_single_line(sdk, tmp_path):
    from tao_sdk.platforms.virtualenv.sdk import _LOG_TAIL_MAX_BYTES

    script = _write_script(
        tmp_path,
        f"import sys\nsys.stdout.write('x' * {_LOG_TAIL_MAX_BYTES + 4096})\n",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)
    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    tail = sdk.get_job_logs(job.id, tail=10_000)
    assert len(tail.encode()) == _LOG_TAIL_MAX_BYTES


def test_job_logs_can_be_streamed_in_bounded_chunks(sdk, tmp_path):
    script = _write_script(tmp_path, "print('0123456789', flush=True)\n")
    job = sdk.create_python_job(script=script, gpu_count=0)
    assert _wait_for_terminal(sdk, job.id).status == "Complete"

    chunks = list(sdk.iter_job_log_chunks(job.id, chunk_size=4))

    assert "".join(chunks) == sdk.get_job_logs(job.id)
    assert all(len(chunk) <= 4 for chunk in chunks)


def test_cancel_kills_worker_that_ignores_sigterm(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print(f'worker_pid={os.getpid()}', flush=True)\n"
        "time.sleep(60)\n",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)
    deadline = time.monotonic() + 5
    worker_pid = None
    while time.monotonic() < deadline:
        logs = sdk.get_job_logs(job.id)
        if "worker_pid=" in logs:
            worker_pid = int(logs.split("worker_pid=", 1)[1].splitlines()[0])
            break
        time.sleep(0.02)
    assert worker_pid is not None

    assert sdk.cancel_job(job.id, timeout=0.2)
    assert sdk.get_job_status(job.id).status == "Canceled"
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_cancel_reaps_children_after_wrapper_exits_without_quiescence(
    sdk, tmp_path, monkeypatch,
):
    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    job = sdk.create_python_job(script=script, gpu_count=0)
    process = sdk._processes[job.id]
    real_terminate = sdk._terminate_process_group
    terminate = MagicMock(side_effect=real_terminate)

    monkeypatch.setattr(process, "poll", lambda: 0)
    monkeypatch.setattr(
        sdk, "_finalize_after_process_group_quiescence", lambda *a, **k: "Running"
    )
    monkeypatch.setattr(sdk, "_terminate_process_group", terminate)

    assert sdk.cancel_job(job.id, timeout=0.2)
    terminate.assert_called_once_with(process.pid, process=process, timeout=0.2)
    assert sdk.get_job_status(job.id).status == "Canceled"


def test_successful_script_does_not_leave_background_worker(sdk, tmp_path):
    script = _write_script(
        tmp_path,
        "import subprocess, sys\n"
        "worker = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        "print(f'worker_pid={worker.pid}', flush=True)\n",
    )
    job = sdk.create_python_job(script=script, gpu_count=0)

    assert _wait_for_terminal(sdk, job.id).status == "Complete"
    logs = sdk.get_job_logs(job.id)
    worker_pid = int(logs.split("worker_pid=", 1)[1].splitlines()[0])
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and Path(f"/proc/{worker_pid}").exists():
        time.sleep(0.02)

    assert not Path(f"/proc/{worker_pid}").exists()


def test_cancel_wins_cross_instance_finalize_race(
    real_venv, tmp_path, monkeypatch,
):
    work_dir = tmp_path / "terminal-race"
    state_file = tmp_path / "terminal-race.db"
    first = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    second = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    job = first.create_python_job(script=script, gpu_count=0)

    finalize_waiting = threading.Event()
    allow_finalize = threading.Event()
    real_compare = first._store.compare_and_set_job_status

    def delayed_compare(job_id, expected_statuses, status):
        if status == "Complete":
            finalize_waiting.set()
            assert allow_finalize.wait(timeout=5)
        return real_compare(job_id, expected_statuses, status)

    monkeypatch.setattr(
        first._store, "compare_and_set_job_status", delayed_compare
    )
    finalize_thread = threading.Thread(
        target=first._finalize_job, args=(job.id, 0)
    )
    finalize_thread.start()
    assert finalize_waiting.wait(timeout=5)
    assert second.cancel_job(job.id, timeout=0.2)
    allow_finalize.set()
    finalize_thread.join(timeout=5)

    assert first.get_job_status(job.id).status == "Canceled"
    assert second.get_job_status(job.id).status == "Canceled"


def test_cancel_wins_cross_instance_lost_launcher_race(
    real_venv, tmp_path, monkeypatch,
):
    work_dir = tmp_path / "lost-launcher-race"
    state_file = tmp_path / "lost-launcher-race.db"
    launcher = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    checker = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    canceler = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    job = launcher.create_python_job(script=script, gpu_count=0)

    monkeypatch.setattr(checker, "_process_matches", lambda *args: False)
    status = checker.get_job_status(job.id)
    assert status.status == "Running"
    assert "non-terminal" in status.message

    assert canceler.cancel_job(job.id, timeout=0.2)
    assert checker.get_job_status(job.id).status == "Canceled"
    assert canceler.get_job_status(job.id).status == "Canceled"


def test_pending_record_recovers_if_orchestrator_crashes_after_spawn(
    real_venv, tmp_path,
):
    work_dir = tmp_path / "pending-recovery"
    state_file = tmp_path / "pending-recovery.db"
    script = _write_script(
        tmp_path,
        "import time\ntime.sleep(0.4)\nprint('pending-recovered', flush=True)\n",
    )
    source = (
        "import os\n"
        "from tao_sdk.platforms.virtualenv import VirtualEnvSDK\n"
        f"sdk = VirtualEnvSDK({str(real_venv)!r}, "
        f"work_dir={str(work_dir)!r}, state_file={str(state_file)!r})\n"
        "sdk._store.update_job_backend_id = lambda *a, **k: os._exit(23)\n"
        f"sdk.create_python_job(script={str(script)!r}, gpu_count=0)\n"
    )
    crashed = subprocess.run([sys.executable, "-c", source], check=False)
    assert crashed.returncode == 23

    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    jobs = recovered.list_jobs()
    assert len(jobs) == 1
    job_id = jobs[0]["job_id"]
    assert recovered.get_job_status(job_id).status in {
        "Pending", "Running", "Complete",
    }
    assert _wait_for_terminal(recovered, job_id).status == "Complete"
    assert "pending-recovered" in recovered.get_job_logs(job_id)


def test_recovered_pending_job_can_cancel_without_opening_start_gate(
    real_venv, tmp_path,
):
    work_dir = tmp_path / "pending-cancel-gate"
    state_file = tmp_path / "pending-cancel-gate.db"
    marker = tmp_path / "pending-training-started"
    script = _write_script(
        tmp_path,
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )
    source = (
        "import os\n"
        "from tao_sdk.platforms.virtualenv import VirtualEnvSDK\n"
        f"sdk = VirtualEnvSDK({str(real_venv)!r}, "
        f"work_dir={str(work_dir)!r}, state_file={str(state_file)!r})\n"
        "sdk._store.update_job_backend_id = lambda *a, **k: os._exit(25)\n"
        f"sdk.create_python_job(script={str(script)!r}, gpu_count=0)\n"
    )
    crashed = subprocess.run([sys.executable, "-c", source], check=False)
    assert crashed.returncode == 25

    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    job = recovered.list_jobs()[0]
    assert recovered.cancel_job(job["job_id"])

    job_dir = Path(job["local_results_dir"]).parent
    launcher_record = job_dir / "launcher_status.json"
    deadline = time.monotonic() + 2
    launcher_pid = None
    while time.monotonic() < deadline:
        if launcher_record.exists():
            launcher_pid = json.loads(launcher_record.read_text())["pid"]
            if not Path(f"/proc/{launcher_pid}").exists():
                break
        time.sleep(0.02)

    assert recovered.get_job_status(job["job_id"]).status == "Canceled"
    assert not marker.exists()
    if launcher_pid is not None:
        assert not Path(f"/proc/{launcher_pid}").exists()


def test_pending_record_stays_nonterminal_if_host_crashes_before_spawn(
    real_venv, tmp_path, monkeypatch,
):
    from tao_sdk.platforms.virtualenv import sdk as sdk_module

    work_dir = tmp_path / "pre-spawn-crash"
    state_file = tmp_path / "pre-spawn-crash.db"
    script = _write_script(tmp_path, "print('must-not-start')\n")
    source = (
        "import os\n"
        "from tao_sdk.platforms.virtualenv import VirtualEnvSDK\n"
        "from tao_sdk.platforms.virtualenv import sdk as sdk_module\n"
        f"sdk = VirtualEnvSDK({str(real_venv)!r}, "
        f"work_dir={str(work_dir)!r}, state_file={str(state_file)!r})\n"
        "sdk_module.subprocess.Popen = lambda *a, **k: os._exit(24)\n"
        f"sdk.create_python_job(script={str(script)!r}, gpu_count=0)\n"
    )
    crashed = subprocess.run([sys.executable, "-c", source], check=False)
    assert crashed.returncode == 24

    monkeypatch.setattr(sdk_module, "_PENDING_LAUNCH_GRACE_SECONDS", 0.0)
    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    jobs = recovered.list_jobs()
    assert len(jobs) == 1
    status = recovered.get_job_status(jobs[0]["job_id"])

    assert status.status == "Pending"
    assert "non-terminal" in status.message


def test_cancel_before_spawn_keeps_training_script_behind_start_gate(
    sdk, tmp_path, monkeypatch,
):
    from tao_sdk.platforms.virtualenv import sdk as sdk_module

    marker = tmp_path / "training-started"
    script = _write_script(
        tmp_path,
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )
    popen_entered = threading.Event()
    allow_popen = threading.Event()
    real_popen = subprocess.Popen

    def blocked_popen(*args, **kwargs):
        popen_entered.set()
        assert allow_popen.wait(timeout=5)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(sdk_module.subprocess, "Popen", blocked_popen)
    launch_errors = []

    def launch():
        try:
            sdk.create_python_job(script=script, gpu_count=0)
        except BaseException as exc:
            launch_errors.append(exc)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    assert popen_entered.wait(timeout=5)
    jobs = sdk.list_jobs()
    assert len(jobs) == 1

    assert sdk.cancel_job(jobs[0]["job_id"])
    assert not marker.exists()
    allow_popen.set()
    launch_thread.join(timeout=5)

    assert not launch_thread.is_alive()
    assert launch_errors
    assert not marker.exists()
    assert sdk.get_job_status(jobs[0]["job_id"]).status == "Canceled"


def test_validation_errors_are_actionable(real_venv, tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        VirtualEnvSDK(tmp_path / "missing-venv", work_dir=tmp_path / "work")

    sdk = VirtualEnvSDK(real_venv, work_dir=tmp_path / "work")
    script = _write_script(tmp_path, "print('ok')\n")
    with pytest.raises(ValueError, match="script does not exist"):
        sdk.create_python_job(script=tmp_path / "missing.py")
    with pytest.raises(ValueError, match="Working directory does not exist"):
        sdk.create_python_job(script=script, cwd=tmp_path / "missing-cwd")
    with pytest.raises(TypeError, match="argv sequence"):
        sdk.create_python_job(script=script, script_args="--epochs 2")
    with pytest.raises(ValueError, match="Unsupported script_args placeholder"):
        sdk.create_python_job(script=script, script_args=["{unknown}"])
    with pytest.raises(ValueError, match="config_format"):
        sdk.create_python_job(script=script, config_format="ini")


def test_toml_missing_writer_has_install_guidance(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tomli_w", None)
    monkeypatch.setitem(sys.modules, "toml", None)
    with pytest.raises(RuntimeError, match=r"nvidia-tao-sdk\[toml\]"):
        VirtualEnvSDK._write_spec(
            tmp_path / "spec.toml", {"train": {"epochs": 1}}, "toml"
        )


def test_result_reader_rejects_directory_traversal(sdk):
    assert sdk.read_job_result_file("missing", "anything") == ""
    sdk._store.create_job(
        job_id="path-test",
        network_arch="",
        action="train",
        workspace_id="local",
        backend_type="virtualenv",
        results_dir=str(sdk.jobs_dir / "path-test" / "results"),
        local_results_dir=str(sdk.jobs_dir / "path-test" / "results"),
        status="Complete",
    )
    (sdk.jobs_dir / "path-test" / "results").mkdir(parents=True)
    with pytest.raises(ValueError, match="stay within"):
        sdk.read_job_result_file("path-test", "../metadata.json")


def test_launcher_ps_fallback_does_not_count_its_own_helper():
    # Without /proc (macOS) the launcher enumerates its group with `ps`; that
    # helper shares the group and must not be reported as a leaked member.
    harness = (
        "import os, sys\n"
        "real_scandir = os.scandir\n"
        "def scandir(path='.'):\n"
        "    if str(path) == '/proc':\n"
        "        raise OSError('no /proc')\n"
        "    return real_scandir(path)\n"
        "os.scandir = scandir\n"
        "namespace = {'__name__': 'launcher'}\n"
        "exec(compile(sys.stdin.read(), 'launch_job.py', 'exec'), namespace)\n"
        "print(namespace['_active_group_members']())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", harness],
        input=_JOB_WRAPPER_SOURCE,
        capture_output=True,
        text=True,
        timeout=30,
        start_new_session=True,  # own process group: no other live members
        check=True,
    )

    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("return_code,background", [(0, False), (7, False), (0, True)])
def test_real_job_ps_fallback_preserves_exit_and_cleans_workers(
    sdk, tmp_path, monkeypatch, return_code, background
):
    """Exercise the generated launcher, real ps and workers, not mocked ps output."""
    import tao_sdk.platforms.virtualenv.sdk as module

    no_proc = (
        "import os\n"
        "_real_scandir = os.scandir\n"
        "def _without_proc(path='.'):\n"
        "    if str(path) == '/proc':\n"
        "        raise FileNotFoundError('/proc unavailable')\n"
        "    return _real_scandir(path)\n"
        "os.scandir = _without_proc\n"
    )
    monkeypatch.setattr(module, "_JOB_WRAPPER_SOURCE", no_proc + module._JOB_WRAPPER_SOURCE)
    script = _write_script(tmp_path,
                           "import subprocess, sys\n" +
                           ("subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                            if background else "") +
                           f"sys.exit({return_code})\n")
    job = sdk.create_python_job(script=script, gpu_count=0)
    status_path = sdk.jobs_dir / job.id / "exit_status.json"
    deadline = time.monotonic() + 15
    while not status_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    record = json.loads(status_path.read_text())
    assert record["group_quiescent"] is True, record
    assert record["return_code"] == return_code, record
    assert _wait_for_terminal(sdk, job.id).status == ("Error" if return_code else "Complete")


@pytest.mark.parametrize("ignore_term", [False, True])
def test_recovered_cancel_without_pid_identity_uses_launcher(
    real_venv, tmp_path, monkeypatch, ignore_term
):
    script = _write_script(tmp_path,
                           "import signal,time\n" +
                           ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") +
                           "print('ready', flush=True)\ntime.sleep(60)\n")
    work_dir, state_file = tmp_path / "jobs", tmp_path / "jobs.db"
    job_id = _launch_from_short_lived_process(real_venv, work_dir, state_file, script)
    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    deadline = time.monotonic() + 10
    while "ready" not in recovered.get_job_logs(job_id):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    monkeypatch.setattr(recovered, "_process_matches", lambda *args: False)

    def forbidden_signal(*args, **kwargs):
        pytest.fail("Must not signal a process group without verified identity")
    monkeypatch.setattr(recovered, "_terminate_process_group", forbidden_signal)
    assert recovered.cancel_job(job_id, timeout=0.2)
    assert recovered.get_job_status(job_id).status == "Canceled"
    record = json.loads((recovered.jobs_dir / job_id / "exit_status.json").read_text())
    assert record["group_quiescent"] is True
    assert record["return_code"] in {-signal.SIGTERM, -signal.SIGKILL}


def test_cooperative_cancel_requires_cleanup_acknowledgement(real_venv, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    work_dir, state_file = tmp_path / "jobs", tmp_path / "jobs.db"
    job_id = _launch_from_short_lived_process(real_venv, work_dir, state_file, script)
    recovered = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    with monkeypatch.context() as patch:
        patch.setattr(recovered, "_process_matches", lambda *args: False)
        patch.setattr(recovered, "_read_exit_record", lambda *args: None)
        assert not recovered.cancel_job(job_id, timeout=0.01)
        assert recovered.get_job_status(job_id).status == "Canceling"
    # The real launcher acknowledged; a subsequent recovery can finalize it.
    # Normal status polling must reconcile a late acknowledgement, without
    # requiring another cancellation request.
    assert recovered.get_job_status(job_id).status == "Canceled"


def test_dead_recovered_launcher_does_not_claim_running(real_venv, tmp_path):
    """A host-like group death cannot prove cleanup but must report it honestly."""
    script = _write_script(tmp_path, "import time\ntime.sleep(60)\n")
    work_dir, state_file = tmp_path / "dead", tmp_path / "dead.db"
    job_id = _launch_from_short_lived_process(real_venv, work_dir, state_file, script)
    sdk = VirtualEnvSDK(real_venv, work_dir=work_dir, state_file=state_file)
    pid = int(sdk.get_job(job_id)["backend_job_id"])
    os.killpg(pid, signal.SIGKILL)  # Exact group created by this test.
    assert not sdk.cancel_job(job_id, timeout=0.01)
    status = sdk.get_job_status(job_id)
    assert status.status not in {"Canceled", "Complete", "Error"}
    assert "quiescence" in status.message
    assert "is running" not in status.message


def test_wrapper_reaps_child_after_polling_exception(tmp_path, monkeypatch):
    """Exceptional polling still reaps the child after group cleanup."""
    namespace = {"__name__": "wrapper_test"}
    exec(_JOB_WRAPPER_SOURCE, namespace)  # pylint: disable=exec-used
    gate = tmp_path / "start"
    gate.touch()
    status = tmp_path / "exit.json"
    monkeypatch.setattr(sys, "argv", [
        "wrapper", str(status), str(tmp_path / "launch.json"), str(gate),
        str(tmp_path / "cancel"), "unused-command",
    ])
    child = MagicMock()
    child.poll.side_effect = KeyboardInterrupt("interrupted polling")
    cleanup = MagicMock()
    namespace["_stop_remaining_group_members"] = cleanup
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    assert namespace["main"]() == 127
    cleanup.assert_called_once_with()
    child.wait.assert_called_once_with()
    record = json.loads(status.read_text())
    assert record["group_quiescent"] is True
    assert "KeyboardInterrupt" in record["error"]
