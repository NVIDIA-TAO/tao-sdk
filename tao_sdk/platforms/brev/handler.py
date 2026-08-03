# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev execution handler for TAO SDK.

Runs jobs on Brev GPU instances via the ``brev`` CLI. The handler is
stateless — it executes whatever command is given on the specified (or
newly created) instance. The agent decides the instance lifecycle
(reuse vs ephemeral).

Instance lifecycle:
  - If ``instance_id`` is in backend_details, reuse that instance.
  - If not, create a new instance for the job.
  - The handler never deletes instances on its own — the agent manages that
    via ``delete_instance()``.
"""

from __future__ import annotations

import logging
import re
import shlex
import uuid

from tao_sdk.platforms.base import BaseHandler
from tao_sdk.platforms.brev.cli import BrevCLI, BrevCLIError

logger = logging.getLogger(__name__)

# Job log file path on the remote instance (per job_id)
_LOG_PATH_TEMPLATE = "/tmp/tao-job-{job_id}.log"
_EXIT_PATH_TEMPLATE = "/tmp/tao-job-{job_id}.exit"
_CID_PATH_TEMPLATE = "/tmp/tao-job-{job_id}.cid"
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

# Cold-start first-exec timeout: covers SSH bring-up + container pull.
# The previous default (30s) truncated valid startup as `exec failed`.
_FIRST_EXEC_TIMEOUT = 600


class BrevHandler(BaseHandler):
    """Execution handler for Brev GPU instances.

    Implements the ExecutionHandler protocol by mapping job operations
    to ``brev exec`` commands on remote instances.
    """

    def __init__(self, cli: BrevCLI | None = None, creds: dict | None = None):
        self._cli = cli or BrevCLI()
        self._creds = creds or {}
        # Track job → (instance, container/pid) mapping in memory.
        # The job store is the durable source of truth; this is a runtime cache.
        self._job_instances: dict[str, str] = {}  # job_id → instance_name
        self._job_pids: dict[str, str] = {}
        self._job_container_ids: dict[str, str] = {}
        self._job_launch_uncertain: set[str] = set()
        self._job_terminal_status: dict[str, str] = {}

    @classmethod
    def from_credentials(cls, creds: dict) -> "BrevHandler":
        """Construct a BrevHandler from a credentials dict.

        If BREV_API_TOKEN is in creds, the CLI auto-authenticates
        via ``brev login --token`` (headless mode). Otherwise, the
        user must have run ``brev login`` manually.
        """
        api_token = creds.get("BREV_API_TOKEN", "")
        cli = BrevCLI(api_token=api_token)
        return cls(cli=cli, creds=creds)

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(
        self,
        image: str,
        network: str,
        action: str,
        cloud_metadata: dict,
        specs: dict,
        job_id: str,
        docker_env_vars: dict,
        resource_shape: str | None,
        dedicated_node_group: str | None,
        num_nodes: int,
        entrypoint: dict | None = None,
        **kwargs,
    ) -> object:
        """Run a job on a Brev instance.

        If ``instance_id`` is in kwargs, reuses that instance. Otherwise
        creates a new one based on resource_shape/gpu type.

        The command to run comes from the entrypoint (built by the SDK in
        ``BrevSDK.create_job``). The handler doesn't care whether
        it's a docker run, a python script, or any other command.
        """
        instance_id = kwargs.get("instance_id", "")
        gpu_type = kwargs.get("gpu_type", "")
        cloud_cred_id = kwargs.get("cloud_cred_id", "")
        workspace_group_id = kwargs.get("workspace_group_id", "")
        first_exec = False

        if entrypoint and entrypoint.get("command"):
            command = entrypoint["command"]
        else:
            error = ValueError(
                "Brev handler requires an entrypoint with 'command'. "
                "Ensure the skill has a script_runner config."
            )
            error.writer_quiescent = True
            raise error

        # Resolve instance: reuse existing or create new
        try:
            if instance_id:
                instance = self._cli.get_instance(instance_id)
                if not instance:
                    logger.info("Instance '%s' not found, creating...", instance_id)
                    self._cli.create_instance(
                        instance_id,
                        gpu_type=gpu_type,
                        cloud_cred_id=cloud_cred_id or None,
                        workspace_group_id=workspace_group_id or None,
                    )
                    self._wait_for_instance(instance_id)
                    first_exec = True
            else:
                # Generate a name from the job_id
                instance_id = f"tao-{job_id[:8]}"
                self._cli.create_instance(
                    instance_id,
                    gpu_type=gpu_type,
                    cloud_cred_id=cloud_cred_id or None,
                    workspace_group_id=workspace_group_id or None,
                )
                self._wait_for_instance(instance_id)
                first_exec = True
        except Exception as exc:
            # Instance lookup/provisioning happens before the remote Docker
            # command is attempted. Surface these definitive failures instead
            # of treating them as an ambiguously running training writer.
            exc.writer_quiescent = True
            raise

        self._job_instances[job_id] = instance_id

        # Set environment variables on the remote instance
        try:
            env_exports = ""
            all_env = {**docker_env_vars}
            # Add cloud storage creds for S3 access
            for key in ("ACCESS_KEY", "SECRET_KEY", "S3_ENDPOINT_URL",
                         "S3_BUCKET_NAME", "NGC_KEY", "HF_TOKEN"):
                val = self._creds.get(key, "")
                if val:
                    all_env[key] = val
            if all_env:
                env_exports = " ".join(
                    f"{k}={_shell_quote(str(v))}" for k, v in all_env.items()
                ) + " "
        except Exception as exc:
            exc.writer_quiescent = True
            raise

        # Run the command in background on the instance
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        exit_path = self.get_exit_status_path(job_id)
        cid_path = self.get_container_id_path(job_id)
        launch_and_report = (
            f"({command}) > {log_path} 2>&1 & launcher_pid=$!; "
            "container_id=''; attempt=0; "
            "while [ \"$attempt\" -lt 50 ]; do "
            f"  if [ -s {shlex.quote(cid_path)} ]; then "
            f"    container_id=$(cat -- {shlex.quote(cid_path)} 2>/dev/null); "
            "    break; "
            "  fi; "
            "  if ! kill -0 \"$launcher_pid\" 2>/dev/null; then break; fi; "
            "  attempt=$((attempt + 1)); sleep 0.1; "
            "done; printf '%s %s\\n' \"$launcher_pid\" \"$container_id\""
        )
        cleanup = (
            f"rm -f -- {shlex.quote(exit_path)} "
            f"{shlex.quote(exit_path)}.tmp.* {shlex.quote(cid_path)}; "
        )
        full_cmd = (
            cleanup + f"export {env_exports.strip()} && "
            f"{launch_and_report}"
        ) if env_exports.strip() else (
            cleanup + launch_and_report
        )

        # Cold-start path: allow time for sshd + container pull on the
        # newly created instance. Reused instances use the normal timeout.
        exec_timeout = _FIRST_EXEC_TIMEOUT if first_exec else 30
        self._job_launch_uncertain.add(job_id)
        try:
            result = self._cli.exec(instance_id, full_cmd, timeout=exec_timeout)
        except Exception as exc:
            return self._handle_ambiguous_launch(
                job_id,
                instance_id,
                f"remote launch transport failed: {exc}",
            )
        except BaseException:
            self.delete_job(job_id)
            raise
        launch_fields = result.stdout.strip().split()
        pid = launch_fields[0] if launch_fields else ""
        if result.returncode != 0 or not pid.isdigit():
            detail = (result.stderr or result.stdout or "remote launch failed").strip()
            return self._handle_ambiguous_launch(job_id, instance_id, detail)
        self._job_pids[job_id] = pid
        container_id = launch_fields[1].lower() if len(launch_fields) == 2 else ""
        if _CONTAINER_ID_RE.fullmatch(container_id):
            self._job_container_ids[job_id] = container_id
        else:
            container_id = ""
        self._job_launch_uncertain.discard(job_id)

        logger.info(
            "Job %s started on instance '%s' (PID=%s, log=%s)",
            job_id, instance_id, pid, log_path,
        )

        runtime = {
            "job_id": job_id,
            "instance_id": instance_id,
            "pid": pid,
            "launch_uncertain": False,
        }
        if container_id:
            runtime["container_id"] = container_id
        return runtime

    def _handle_ambiguous_launch(
        self,
        job_id: str,
        instance_id: str,
        detail: str,
    ) -> dict:
        """Stop an ambiguously launched writer or return a recoverable handle."""
        if not self.delete_job(job_id, require_observed=True):
            logger.error(
                "Brev launch for %s was ambiguous and writer termination could "
                "not be confirmed after it was positively observed; returning "
                "recoverable runtime identity",
                job_id,
            )
            return {
                "job_id": job_id,
                "instance_id": instance_id,
                "pid": "",
                "launch_uncertain": True,
            }
        error = RuntimeError(
            f"Brev job {job_id} did not return a valid launcher PID; its writer "
            f"was confirmed stopped: {detail}"
        )
        error.writer_quiescent = True
        raise error

    def get_job_name(self, job_id: str) -> str:
        """Return the job ID as-is (no backend-specific naming)."""
        return job_id

    @staticmethod
    def get_container_name(job_id: str) -> str:
        """Return the deterministic Docker container name for a Brev job."""
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(job_id)).strip("-")
        return f"tao-job-{safe}"[:128]

    @staticmethod
    def get_exit_status_path(job_id: str) -> str:
        """Return the remote durable exit-code marker for a Brev job."""
        return _EXIT_PATH_TEMPLATE.format(job_id=job_id)

    @staticmethod
    def get_container_id_path(job_id: str) -> str:
        """Return the remote Docker cidfile path for a Brev job."""
        return _CID_PATH_TEMPLATE.format(job_id=job_id)

    def restore_job_runtime(self, job_id: str, runtime: dict | None) -> bool:
        """Restore the durable instance/PID/launch state after an SDK restart."""
        if not isinstance(runtime, dict):
            return False
        instance_id = runtime.get("instance_id")
        pid = runtime.get("pid", "")
        container_id = runtime.get("container_id", "")
        launch_uncertain = runtime.get("launch_uncertain", False)
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(pid, str)
            or (pid and not pid.isdigit())
            or not isinstance(container_id, str)
            or (container_id and not _CONTAINER_ID_RE.fullmatch(container_id))
            or not isinstance(launch_uncertain, bool)
        ):
            return False
        self._job_instances[job_id] = instance_id
        if pid:
            self._job_pids[job_id] = pid
        else:
            self._job_pids.pop(job_id, None)
        if container_id:
            self._job_container_ids[job_id] = container_id.lower()
        else:
            self._job_container_ids.pop(job_id, None)
        if launch_uncertain:
            self._job_launch_uncertain.add(job_id)
        else:
            self._job_launch_uncertain.discard(job_id)
        return True

    def get_job_runtime_identity(self, job_id: str) -> dict | None:
        """Return the current durable Brev runtime coordinates."""
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return None
        runtime = {
            "instance_id": instance_id,
            "pid": self._job_pids.get(job_id, ""),
            "launch_uncertain": job_id in self._job_launch_uncertain,
        }
        container_id = self._job_container_ids.get(job_id, "")
        if container_id:
            runtime["container_id"] = container_id
        return runtime

    def get_tao_job_status(self, job_id: str) -> str | None:
        """Check if the job process is still running on the instance.

        Returns 'Running', 'Error', or None (complete).
        """
        terminal_status = self._job_terminal_status.get(job_id)
        if terminal_status == "Complete":
            return None
        if terminal_status in {"Error", "Canceled"}:
            return terminal_status
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            logger.warning(
                "Brev runtime identity for %s is unavailable; keeping the job "
                "non-terminal until it can be recovered",
                job_id,
            )
            return "Pending"

        # Bind the remote writer to Docker's durable cidfile.  A deterministic
        # name alone is not an identity: after ``docker run --rm`` exits,
        # another container can legitimately reuse it.
        container_name_value = self.get_container_name(job_id)
        container_name = shlex.quote(container_name_value)
        expected_name = shlex.quote(f"/{container_name_value}")
        expected_id = self._job_container_ids.get(job_id, "")
        cid_path = shlex.quote(self.get_container_id_path(job_id))
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        exit_path = shlex.quote(self.get_exit_status_path(job_id))
        inspect_format = shlex.quote(
            '{{.Id}}|{{.Name}}|{{index .Config.Labels "tao-sdk-job-id"}}|'
            '{{index .Config.Labels "tao-sdk-backend"}}|'
            "{{.State.Running}}|{{.State.ExitCode}}"
        )
        id_format = shlex.quote("{{.Id}}")
        pid = self._job_pids.get(job_id, "")
        launcher_running = (
            f"ps -p {shlex.quote(pid)} -o args= 2>/dev/null | "
            f"grep -F -- {container_name} >/dev/null"
            if pid.isdigit() else "false"
        )
        terminal_probe = (
            f"if {launcher_running}; then echo LAUNCHER_RUNNING; "
            f"elif [ -f {exit_path} ]; then "
            f"  exit_code=$(cat -- {exit_path} 2>/dev/null); "
            "  if [ \"$exit_code\" = 0 ]; then echo EXIT_DONE; "
            "  elif echo \"$exit_code\" | grep -qE '^[0-9]+$'; then "
            "    echo EXIT_FAILED:$exit_code; else echo EXIT_UNKNOWN; fi; "
            f"elif [ -f {log_path} ]; then echo LOG_ONLY; "
            "else echo MISSING; fi"
        )
        result = self._cli.exec(
            instance_id,
            f"expected_id={shlex.quote(expected_id)}; "
            f"cidfile_id=$(cat -- {cid_path} 2>/dev/null || true); "
            "if [ -n \"$expected_id\" ] && [ -n \"$cidfile_id\" ] && "
            "   [ \"$expected_id\" != \"$cidfile_id\" ]; then "
            "  echo IDENTITY_MISMATCH; "
            "else "
            "  [ -n \"$expected_id\" ] || expected_id=$cidfile_id; "
            "  if ! printf '%s\\n' \"$expected_id\" | "
            "       grep -qE '^[0-9a-fA-F]{64}$'; then "
            f"    named=$(docker inspect {container_name} 2>&1); named_rc=$?; "
            "    if [ $named_rc -eq 0 ]; then echo IDENTITY_PENDING; "
            "    elif echo \"$named\" | "
            "         grep -qiE 'No such (object|container)'; then "
            f"      {terminal_probe}; "
            "    else echo UNKNOWN; fi; "
            "  else "
            "    echo CONTAINER_ID:$expected_id; "
            f"    out=$(docker inspect -f {inspect_format} \"$expected_id\" 2>&1); "
            "    rc=$?; if [ $rc -eq 0 ]; then "
            "      set -f; old_ifs=$IFS; IFS='|'; set -- $out; IFS=$old_ifs; "
            "      actual_id=$1; actual_name=$2; job_label=$3; "
            "      backend_label=$4; running=$5; exit_code=$6; "
            f"      if [ \"$actual_id\" != \"$expected_id\" ] || "
            f"         [ \"$actual_name\" != {expected_name} ] || "
            f"         [ \"$job_label\" != {shlex.quote(str(job_id))} ] || "
            "         [ \"$backend_label\" != brev ]; then "
            "        echo IDENTITY_MISMATCH; "
            "      elif [ \"$running\" = true ]; then "
            "        echo CONTAINER_RUNNING:$actual_id; "
            "      elif [ \"${exit_code:-1}\" = 0 ]; then "
            "        echo CONTAINER_DONE:$actual_id; "
            "      else echo CONTAINER_FAILED:$actual_id; fi; "
            "    elif echo \"$out\" | "
            "         grep -qiE 'No such (object|container)'; then "
            f"      named=$(docker inspect -f {id_format} "
            f"{container_name} 2>&1); named_rc=$?; "
            "      if [ $named_rc -eq 0 ]; then "
            "        if [ \"$named\" = \"$expected_id\" ]; then echo UNKNOWN; "
            "        else echo IDENTITY_MISMATCH; fi; "
            "      elif echo \"$named\" | "
            "           grep -qiE 'No such (object|container)'; then "
            f"        {terminal_probe}; "
            "      else echo UNKNOWN; fi; "
            "    else echo UNKNOWN; fi; "
            "  fi; "
            "fi",
            timeout=15,
        )
        if result.returncode != 0:
            logger.warning(
                "Could not inspect Brev writer container for %s: %s",
                job_id,
                (result.stderr or result.stdout or "transport failure").strip(),
            )
            return "Running"
        output = result.stdout.strip()
        output_lines = output.splitlines()
        if output_lines:
            identity_match = re.fullmatch(
                r"CONTAINER_ID:([0-9a-fA-F]{64})",
                output_lines[0].strip(),
            )
            if identity_match:
                observed_id = identity_match.group(1).lower()
                recorded_id = self._job_container_ids.get(job_id, "")
                if recorded_id and recorded_id != observed_id:
                    logger.error(
                        "Refusing Brev status for %s: recorded container %s "
                        "conflicts with cidfile container %s",
                        job_id,
                        recorded_id,
                        observed_id,
                    )
                    return "Pending"
                self._job_container_ids[job_id] = observed_id
                output = "\n".join(output_lines[1:]).strip()
        marker_match = re.fullmatch(
            r"(CONTAINER_(?:RUNNING|DONE|FAILED)):([0-9a-fA-F]{64})",
            output,
        )
        if marker_match:
            marker, observed_id = marker_match.groups()
            observed_id = observed_id.lower()
            recorded_id = self._job_container_ids.get(job_id, "")
            if recorded_id and recorded_id != observed_id:
                logger.error(
                    "Refusing Brev status for %s: recorded container %s was "
                    "replaced by %s",
                    job_id,
                    recorded_id,
                    observed_id,
                )
                return "Pending"
            self._job_container_ids[job_id] = observed_id
            output = marker

        if output in {"CONTAINER_RUNNING", "CONTAINER_DONE", "CONTAINER_FAILED"}:
            self._job_launch_uncertain.discard(job_id)
            if output == "CONTAINER_RUNNING":
                return "Running"
            terminal = "Complete" if output == "CONTAINER_DONE" else "Error"
            self._job_terminal_status[job_id] = terminal
            return None if terminal == "Complete" else terminal
        if output == "EXIT_DONE" or output.startswith("EXIT_FAILED:"):
            self._job_launch_uncertain.discard(job_id)
            terminal = "Complete" if output == "EXIT_DONE" else "Error"
            self._job_terminal_status[job_id] = terminal
            return None if terminal == "Complete" else terminal
        if output == "LAUNCHER_RUNNING":
            return "Running"
        if output in {"IDENTITY_MISMATCH", "IDENTITY_PENDING"}:
            logger.error(
                "Refusing Brev status lookup for %s because the named "
                "container does not match its durable identity",
                job_id,
            )
            return "Pending"
        if "UNKNOWN" in output:
            # A transport/daemon failure is not evidence that the writer is
            # gone. Keep the job non-terminal so cleanup cannot race it.
            return "Running"
        if "MISSING" in output:
            return "Pending"
        if "LOG_ONLY" in output:
            # docker run --rm removes the only authoritative exit code. A log
            # file alone proves neither success nor failure.
            return "Pending"
        return "Pending"

    def normalize_status(self, raw_status: str | None) -> str:
        """Map raw status to SDK status string."""
        if raw_status is None:
            return "Complete"
        return raw_status

    def get_job_message(self, job_id: str) -> str:
        """Get last status line from job log."""
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return ""
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        result = self._cli.exec(
            instance_id, f"tail -1 {log_path} 2>/dev/null || echo ''",
            timeout=10,
        )
        return result.stdout.strip()

    def get_job_logs(self, job_id: str, tail_lines: int | None = None) -> str | None:
        """Fetch job logs from the remote instance."""
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return None
        log_path = _LOG_PATH_TEMPLATE.format(job_id=job_id)
        if tail_lines:
            cmd = f"tail -{tail_lines} {log_path} 2>/dev/null"
        else:
            cmd = f"cat {log_path} 2>/dev/null"
        result = self._cli.exec(instance_id, cmd, timeout=30)
        return result.stdout if result.returncode == 0 else None

    def delete_job(self, job_id: str, *, require_observed: bool = False) -> bool:
        """Stop the named writer container and its launcher process.

        When launch transport failed, an immediate absent container does not
        prove that the remote command cannot complete late. In that path,
        ``require_observed`` only reports success after the named writer was
        positively seen and then removed.
        """
        instance_id = self._job_instances.get(job_id)
        if not instance_id:
            return False
        observed_status = self._job_terminal_status.get(job_id)
        if observed_status in {"Complete", "Error", "Canceled"}:
            return True
        container_name_value = self.get_container_name(job_id)
        container_name = shlex.quote(container_name_value)
        expected_name = shlex.quote(f"/{container_name_value}")
        expected_id = self._job_container_ids.get(job_id, "")
        cid_path = shlex.quote(self.get_container_id_path(job_id))
        exit_path = shlex.quote(self.get_exit_status_path(job_id))
        inspect_format = shlex.quote(
            '{{.Id}}|{{.Name}}|{{index .Config.Labels "tao-sdk-job-id"}}|'
            '{{index .Config.Labels "tao-sdk-backend"}}|'
            "{{.State.Running}}|{{.State.ExitCode}}"
        )
        id_format = shlex.quote("{{.Id}}")
        pid = self._job_pids.get(job_id, "")
        kill_launcher = (
            f"if ps -p {shlex.quote(pid)} -o args= 2>/dev/null | "
            f"grep -F -- {container_name} >/dev/null; then "
            f"kill {shlex.quote(pid)} 2>/dev/null || true; fi; "
            if pid.isdigit() else ""
        )
        launcher_still_running = (
            f"ps -p {shlex.quote(pid)} -o args= 2>/dev/null | "
            f"grep -F -- {container_name} >/dev/null"
            if pid.isdigit() else "false"
        )
        result = self._cli.exec(
            instance_id,
            f"expected_id={shlex.quote(expected_id)}; "
            f"cidfile_id=$(cat -- {cid_path} 2>/dev/null || true); "
            f"recorded_exit=$(cat -- {exit_path} 2>/dev/null || true); "
            "if [ -n \"$expected_id\" ] && [ -n \"$cidfile_id\" ] && "
            "   [ \"$expected_id\" != \"$cidfile_id\" ]; then "
            "  echo IDENTITY_MISMATCH; exit 1; "
            "fi; "
            "[ -n \"$expected_id\" ] || expected_id=$cidfile_id; "
            "if ! printf '%s\\n' \"$expected_id\" | "
            "     grep -qE '^[0-9a-fA-F]{64}$'; then "
            f"  named=$(docker inspect {container_name} 2>&1); named_rc=$?; "
            "  if [ $named_rc -eq 0 ]; then echo IDENTITY_PENDING; exit 1; "
            "  elif ! echo \"$named\" | "
            "       grep -qiE 'No such (object|container)'; then "
            "    echo UNKNOWN >&2; exit 1; "
            "  elif [ \"$recorded_exit\" = 0 ]; then echo COMPLETE_RECORDED; "
            "    exit 0; "
            "  elif echo \"$recorded_exit\" | grep -qE '^[0-9]+$'; then "
            "    echo ERROR_RECORDED; exit 0; "
            "  else echo IDENTITY_PENDING; exit 1; fi; "
            "fi; "
            f"before=$(docker inspect -f {inspect_format} "
            "\"$expected_id\" 2>&1); before_rc=$?; "
            "if [ $before_rc -eq 0 ]; then "
            "  set -f; old_ifs=$IFS; IFS='|'; set -- $before; IFS=$old_ifs; "
            "  actual_id=$1; actual_name=$2; job_label=$3; backend_label=$4; "
            "  before_running=$5; before_exit=$6; "
            f"  if [ \"$actual_id\" != \"$expected_id\" ] || "
            f"     [ \"$actual_name\" != {expected_name} ] || "
            f"     [ \"$job_label\" != {shlex.quote(str(job_id))} ] || "
            "     [ \"$backend_label\" != brev ]; then "
            "    echo IDENTITY_MISMATCH; exit 1; "
            "  fi; "
            f"  named_id=$(docker inspect -f {id_format} "
            f"{container_name} 2>&1); named_rc=$?; "
            "  if [ $named_rc -ne 0 ]; then echo UNKNOWN >&2; exit 1; "
            "  elif [ \"$named_id\" != \"$expected_id\" ]; then "
            "    echo IDENTITY_MISMATCH; exit 1; "
            "  fi; "
            "  docker stop --time 10 \"$expected_id\" >/dev/null 2>&1 || true; "
            "  docker rm -f \"$expected_id\" >/dev/null 2>&1 || true; "
            f"  {kill_launcher}"
            "  after=$(docker inspect \"$expected_id\" 2>&1); after_rc=$?; "
            f"  named_after=$(docker inspect -f {id_format} "
            f"{container_name} 2>&1); named_after_rc=$?; "
            f"  if {launcher_still_running}; then echo RUNNING; exit 1; "
            "  elif [ $after_rc -eq 0 ]; then echo RUNNING; exit 1; "
            "  elif ! echo \"$after\" | "
            "       grep -qiE 'No such (object|container)'; then "
            "    echo UNKNOWN >&2; exit 1; "
            "  elif [ $named_after_rc -eq 0 ]; then "
            "    echo IDENTITY_MISMATCH; exit 1; "
            "  elif ! echo \"$named_after\" | "
            "       grep -qiE 'No such (object|container)'; then "
            "    echo UNKNOWN >&2; exit 1; "
            "  elif [ \"$before_running\" = false ]; then "
            "    if [ \"$before_exit\" = 0 ]; then echo COMPLETE_OBSERVED; "
            "    else echo ERROR_OBSERVED; fi; "
            "  else echo STOPPED_OBSERVED; fi; "
            "elif echo \"$before\" | "
            "     grep -qiE 'No such (object|container)'; then "
            f"  named_id=$(docker inspect -f {id_format} "
            f"{container_name} 2>&1); named_rc=$?; "
            "  if [ $named_rc -eq 0 ]; then echo IDENTITY_MISMATCH; exit 1; "
            "  elif ! echo \"$named_id\" | "
            "       grep -qiE 'No such (object|container)'; then "
            "    echo UNKNOWN >&2; exit 1; "
            "  elif [ \"$recorded_exit\" = 0 ]; then echo COMPLETE_RECORDED; "
            "  elif echo \"$recorded_exit\" | grep -qE '^[0-9]+$'; then "
            "    echo ERROR_RECORDED; "
            "  else echo ABSENT; fi; "
            "else echo UNKNOWN >&2; exit 1; fi",
            timeout=30,
        )
        stopped = "STOPPED_OBSERVED" in result.stdout
        completed = "COMPLETE_OBSERVED" in result.stdout
        failed = "ERROR_OBSERVED" in result.stdout
        completed = completed or "COMPLETE_RECORDED" in result.stdout
        failed = failed or "ERROR_RECORDED" in result.stdout
        absent = "ABSENT" in result.stdout
        if result.returncode != 0 or not (stopped or completed or failed or absent):
            logger.warning(
                "Could not confirm exact Brev container termination for %s: %s",
                job_id,
                (result.stderr or result.stdout or "identity unavailable").strip(),
            )
            return False
        require_positive_observation = (
            require_observed or job_id in self._job_launch_uncertain
        )
        if require_positive_observation and not (stopped or completed or failed):
            logger.warning(
                "Brev container %s was not observed before cancellation; its "
                "launch remains recoverable because the remote exec may finish late",
                job_id,
            )
            return False
        self._job_instances.pop(job_id, None)
        self._job_pids.pop(job_id, None)
        self._job_container_ids.pop(job_id, None)
        self._job_launch_uncertain.discard(job_id)
        self._job_terminal_status[job_id] = (
            "Complete" if completed else "Error" if failed else "Canceled"
        )
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        return self.delete_job(job_id)

    # ------------------------------------------------------------------
    # Resource discovery
    # ------------------------------------------------------------------

    def get_available_instances(self, **kwargs) -> dict:
        """List available GPU instance types from Brev search."""
        try:
            results = self._cli.search()
            available = {}
            for i, item in enumerate(results):
                platform_id = f"brev-{i}"
                available[platform_id] = {
                    "cluster": "brev",
                    "gpu_type": item.get("gpu", ""),
                    "instance_type": item.get("type", ""),
                    "gpu_count": item.get("gpuCount", 1),
                    "backend_type": "brev",
                }
            return available
        except Exception as e:
            logger.warning("Failed to query Brev instances: %s", e)
            return {}

    def resolve_resource_shape(
        self, num_gpus: int, preferred_node_group: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Map GPU count to a Brev instance type."""
        instances = self.get_available_instances()
        for pid, info in instances.items():
            if info.get("gpu_count", 0) >= num_gpus:
                return info["gpu_type"], "brev"
        return None, None

    # ------------------------------------------------------------------
    # Instance management (called by agent, not by SDK)
    # ------------------------------------------------------------------

    def delete_instance(self, instance_id: str) -> bool:
        """Delete a Brev instance. Called by agent when done with it."""
        return self._cli.delete_instance(instance_id)

    def list_instances(self) -> list[dict]:
        """List running Brev instances."""
        try:
            return self._cli.list_instances()
        except BrevCLIError:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_instance(self, instance_id: str, timeout: int = 300):
        """Wait for a Brev instance to reach RUNNING state AND be exec-ready.

        Two-stage wait:
          1. Poll status until RUNNING/READY.
          2. Probe `brev exec ... -- true` until it succeeds — status flips
             well before sshd/hostname resolution is ready, so a status-only
             wait surfaces as `hostname not resolvable` / `Connection refused`
             on the first real exec.
        """
        import time
        deadline = time.time() + timeout
        status_ok = False
        while time.time() < deadline:
            inst = self._cli.get_instance(instance_id)
            if inst and inst.get("status", "").lower() in ("running", "ready"):
                logger.info("Instance '%s' reached RUNNING", instance_id)
                status_ok = True
                break
            time.sleep(10)
        if not status_ok:
            raise BrevCLIError(
                f"Instance '{instance_id}' did not reach RUNNING state "
                f"within {timeout}s"
            )
        # Bound the exec-readiness probe by whatever wall-clock budget remains
        # (with a 60s floor for very tight timeouts).
        remaining = max(int(deadline - time.time()), 60)
        self._cli.wait_for_exec_ready(instance_id, timeout=remaining)


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\"'\"'") + "'"
