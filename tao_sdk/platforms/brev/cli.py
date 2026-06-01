# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Brev CLI wrapper with token refresh and JSON output parsing."""

from __future__ import annotations

import json
import logging
import subprocess
import time

logger = logging.getLogger(__name__)

# Cold-start exec on a freshly created instance must allow time for
# sshd bring-up + container pull. The default 120s `_timeout` truncates
# legitimate startup as `exec failed`; 600s covers observed cold paths.
_COLD_START_EXEC_TIMEOUT = 600

# Substrings (lowercased) that indicate the CLI hit an interactive auth
# prompt with no stdin (`brev ls` after a desynced cached session). Worth
# a single token re-login + retry before failing.
_AUTH_EOF_MARKERS = ("eof", "unable to read", "stdin", "no tty")


class BrevCLI:
    """Wrapper around the ``brev`` CLI binary.

    Handles JSON output parsing, token refresh on auth failures,
    and timeout management for long-running commands.

    If ``api_token`` is provided, the CLI auto-authenticates via
    ``brev login --token`` on first use (headless mode). Get the
    token once from the Brev console and set it as ``BREV_API_TOKEN``
    in your environment (e.g., ``~/.config/tao/.env``).
    """

    def __init__(self, api_token: str = "", timeout: int = 120):
        self._api_token = api_token
        self._timeout = timeout
        self._last_refresh: float = 0
        self._authenticated = False

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _ensure_authenticated(self):
        """Auto-login via token if not yet authenticated."""
        if self._authenticated or not self._api_token:
            return
        try:
            result = subprocess.run(
                ["brev", "login", "--token", self._api_token],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("Brev auto-login successful (headless token)")
                self._authenticated = True
            else:
                logger.warning("Brev auto-login failed: %s", result.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Brev auto-login failed: %s", e)

    def _run(self, args: list[str], timeout: int | None = None,
             check: bool = True) -> subprocess.CompletedProcess:
        """Run a brev CLI command and return the result."""
        self._ensure_authenticated()
        cmd = ["brev"] + args
        timeout = timeout or self._timeout
        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            if check and result.returncode != 0:
                err_lower = result.stderr.lower()
                is_unauthorized = "unauthorized" in err_lower or "401" in result.stderr
                is_auth_eof = any(m in err_lower for m in _AUTH_EOF_MARKERS)
                if is_unauthorized or is_auth_eof:
                    # 401: token expired. EOF: cached session desynced and the
                    # CLI tried to prompt on a stdin-less shell. Both recover
                    # with a forced token re-login when a token is present.
                    logger.info(
                        "Brev auth failure (%s), forcing token re-login...",
                        "401" if is_unauthorized else "auth-EOF",
                    )
                    self._force_login()
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=timeout,
                    )
                    if result.returncode != 0:
                        raise BrevCLIError(
                            f"brev {args[0]} failed after token refresh: {result.stderr}"
                        )
                else:
                    raise BrevCLIError(
                        f"brev {args[0]} failed (exit {result.returncode}): {result.stderr}"
                    )
            return result
        except subprocess.TimeoutExpired:
            raise BrevCLIError(f"brev {args[0]} timed out after {timeout}s")
        except FileNotFoundError:
            raise BrevCLIError(
                "brev CLI not found. Install from https://docs.nvidia.com/brev/ "
                "or run: curl -fsSL https://brev.dev/install | bash"
            )

    def _run_json(self, args: list[str], **kwargs) -> dict | list:
        """Run a brev command with --json flag and parse output."""
        result = self._run(args + ["--json"], **kwargs)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise BrevCLIError(
                f"Failed to parse JSON from brev {args[0]}: {result.stdout[:500]}"
            )

    def refresh_token(self):
        """Refresh the Brev auth token.

        Brev tokens expire hourly. ``brev ls`` triggers a refresh.
        """
        now = time.time()
        if now - self._last_refresh < 60:
            return  # Don't refresh more than once per minute
        try:
            self._run(["ls"], check=False, timeout=30)
            self._last_refresh = now
            logger.info("Brev token refreshed")
        except BrevCLIError as e:
            logger.warning("Token refresh failed: %s", e)

    def _force_login(self):
        """Re-run `brev login --token` to rebuild the cached session.

        Used by the auth-failure retry path in ``_run`` — distinct from
        ``refresh_token`` (which uses `brev ls` and is rate-limited).
        """
        if not self._api_token:
            return
        try:
            result = subprocess.run(
                ["brev", "login", "--token", self._api_token],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                self._authenticated = True
                self._last_refresh = time.time()
                logger.info("Brev forced re-login succeeded")
            else:
                logger.warning("Brev forced re-login failed: %s", result.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning("Brev forced re-login failed: %s", e)

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------

    def create_instance(self, name: str, gpu_type: str | None = None,
                        cloud_cred_id: str | None = None,
                        workspace_group_id: str | None = None,
                        **kwargs) -> dict:
        """Create a new Brev GPU instance.

        Args:
            name: Instance name (must be unique within org).
            gpu_type: GPU type filter (e.g., 'A100', 'L40S').
            cloud_cred_id: Cloud credential ID for placement. Required on
                accounts with more than one cloud credential — otherwise
                `brev create` rejects with a placement error.
            workspace_group_id: Workspace group ID for placement. Required on
                accounts with more than one workspace group.

        Returns:
            Instance info dict with id, name, status.
        """
        args = ["create", name]
        if gpu_type:
            args.extend(["--gpu-name", gpu_type])
        if cloud_cred_id:
            args.extend(["--cloud-cred-id", cloud_cred_id])
        if workspace_group_id:
            args.extend(["--workspace-group-id", workspace_group_id])
        result = self._run(args, timeout=300)
        logger.info("Created Brev instance '%s'", name)
        # Parse instance ID from output
        return {"name": name, "stdout": result.stdout}

    def list_instances(self) -> list[dict]:
        """List all Brev instances."""
        return self._run_json(["ls"])

    def get_instance(self, name: str) -> dict | None:
        """Get a specific instance by name. Returns None if not found."""
        instances = self.list_instances()
        if isinstance(instances, list):
            for inst in instances:
                if inst.get("name") == name:
                    return inst
        return None

    def delete_instance(self, name: str) -> bool:
        """Delete a Brev instance."""
        try:
            # `brev delete` has no --yes/-y flag; passing one errors with
            # "unknown flag: --yes". Recent CLIs are already non-interactive.
            self._run(["delete", name])
            logger.info("Deleted Brev instance '%s'", name)
            return True
        except BrevCLIError as e:
            logger.error("Failed to delete instance '%s': %s", name, e)
            return False

    def start_instance(self, name: str) -> bool:
        """Start a stopped instance."""
        try:
            self._run(["start", name])
            return True
        except BrevCLIError:
            return False

    def stop_instance(self, name: str) -> bool:
        """Stop a running instance."""
        try:
            self._run(["stop", name])
            return True
        except BrevCLIError:
            return False

    # ------------------------------------------------------------------
    # Remote execution
    # ------------------------------------------------------------------

    def exec(self, instance: str, command: str,
             timeout: int | None = None) -> subprocess.CompletedProcess:
        """Execute a command on a Brev instance via SSH.

        Args:
            instance: Instance name.
            command: Shell command to run remotely.
            timeout: Command timeout in seconds. Cold-start callers (first
                exec after instance create) should pass at least
                ``_COLD_START_EXEC_TIMEOUT`` to cover sshd bring-up +
                container pull; the default 120s truncates legitimate
                startup as `exec failed`.

        Returns:
            CompletedProcess with stdout/stderr.
        """
        return self._run(
            ["exec", instance, "--", "bash", "-c", command],
            timeout=timeout or self._timeout,
            check=False,  # Don't raise on non-zero — let caller handle
        )

    def wait_for_exec_ready(self, instance: str,
                            timeout: int = 300,
                            probe_interval: int = 5) -> None:
        """Block until `brev exec <instance> -- true` succeeds.

        A freshly created instance reports RUNNING long before sshd,
        hostname resolution, and the user shell are ready. The first real
        `brev exec` against an unsettled instance fails with
        `hostname not resolvable`, `Connection refused`, or a silent
        timeout. Call this after the status poll, before the first real
        command, to absorb the bring-up window.
        """
        deadline = time.time() + timeout
        last_err = ""
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    ["brev", "exec", instance, "--", "true"],
                    capture_output=True, text=True, timeout=probe_interval * 3,
                )
                if result.returncode == 0:
                    logger.info("Instance '%s' is exec-ready", instance)
                    return
                last_err = (result.stderr or result.stdout).strip()
            except subprocess.TimeoutExpired:
                last_err = f"probe timed out after {probe_interval * 3}s"
            except FileNotFoundError:
                raise BrevCLIError("brev CLI not found")
            time.sleep(probe_interval)
        raise BrevCLIError(
            f"Instance '{instance}' never became exec-ready within {timeout}s "
            f"(last: {last_err})"
        )

    def exec_background(self, instance: str, command: str) -> str:
        """Execute a command in the background on a Brev instance.

        Wraps the command with nohup and redirects output.
        Returns immediately.
        """
        bg_cmd = f"nohup bash -c {_shell_quote(command)} > /tmp/tao-job.log 2>&1 & echo $!"
        result = self.exec(instance, bg_cmd, timeout=30)
        pid = result.stdout.strip()
        logger.info("Started background process PID=%s on '%s'", pid, instance)
        return pid

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, gpu_name: str | None = None,
               min_vram: int | None = None) -> list[dict]:
        """Search for available GPU instance types."""
        args = ["search"]
        if gpu_name:
            args.extend(["--gpu-name", gpu_name])
        if min_vram:
            args.extend(["--min-vram", str(min_vram)])
        try:
            return self._run_json(args, timeout=30)
        except BrevCLIError:
            return []


class BrevCLIError(Exception):
    """Raised when a brev CLI command fails."""
    pass


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell embedding."""
    return "'" + s.replace("'", "'\"'\"'") + "'"
