"""Where Slurm commands run: this login node, or another one over ssh.

Starting an ssh session on a cluster login node can take seconds (shell
start-up files, MFA-guarded sshd) while the Slurm commands themselves are
quick, so a `Remote` runs a whole refresh in one session, and streams a whole
probe grid through another. Sessions reuse an ssh control master.

Every command runs without the terminal and never prompts: slurmtop does not
log in for the user. A cluster that needs a password or MFA is reached through
a control master the user opened with `ssh` in a terminal of their own.
"""

import os
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Separates command outputs within one remote session; Slurm never prints it.
RECORD = "\x1e"
# Concurrent local probes; each is a will-run test in slurmctld, so stay polite.
PROBE_WORKERS = 4
PROBE_TIMEOUT = 30
# Control sockets when the user's ssh config does not multiplex already.
CONTROL_DIR = Path(tempfile.gettempdir()) / f"slurmtop-{os.getuid()}"
CONTROL_PERSIST = "10m"


class ConnectionLost(Exception):
    """ssh could not reach the cluster; the message is ssh's own."""


def connect_hint(ssh):
    """How the user gets `ssh` connected, shown with a `ConnectionLost`."""
    return (
        "slurmtop never asks for a password or MFA: log in with "
        f"`ssh {ssh}` in another terminal, with ControlMaster set for it in "
        "~/.ssh/config so slurmtop can share that connection."
    )


def _run(args, timeout):
    """Return (returncode, stdout, stderr); 127 if missing, 124 on timeout."""
    try:
        done = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"'{args[0]}' not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"Timed out after {timeout}s"
    return done.returncode, done.stdout, done.stderr


def _combined(stdout, stderr):
    return "\n".join(part.strip() for part in (stdout, stderr) if part.strip())


class Local:
    """Run commands on this machine."""

    ssh = None

    def run(self, args, timeout=60):
        """Return stdout, or None when the command is missing, fails or hangs."""
        code, stdout, _ = _run(args, timeout)
        return stdout if code == 0 else None

    def act(self, args, timeout=60):
        """Run a command that acts, such as `sbatch` or `scancel`:
        (returncode, output).

        The output has stdout and stderr together, so a rejection says why.
        """
        code, stdout, stderr = _run(args, timeout)
        return code, _combined(stdout, stderr)

    def run_batch(self, commands, timeout=120):
        """Run commands in parallel; their stdout in order, None for failures."""
        if not commands:
            return []
        with ThreadPoolExecutor(len(commands)) as pool:
            return list(pool.map(lambda args: self.run(args, timeout), commands))

    def run_stream(self, commands, on_result, should_stop=lambda: False):
        """Run commands a few at a time, reporting each as it finishes.

        `on_result(index, returncode, output)` gets stdout and stderr together.
        Once `should_stop()` is true, commands not yet started are dropped.
        """
        with ThreadPoolExecutor(PROBE_WORKERS) as pool:
            futures = {
                pool.submit(_run, args, PROBE_TIMEOUT): index
                for index, args in enumerate(commands)
            }
            for future in as_completed(futures):
                if should_stop():
                    pool.shutdown(cancel_futures=True)
                    break
                code, stdout, stderr = future.result()
                on_result(futures[future], code, _combined(stdout, stderr))


class Remote:
    """Run commands on a cluster's login node through `ssh <destination>`.

    `destination` is anything ssh accepts, such as a `~/.ssh/config` alias.
    Commands run with the login shell's PATH (read by `detect_commands`),
    since non-login sessions often lack Slurm's directory.
    """

    def __init__(self, ssh):
        self.ssh = ssh
        self.path = None
        self._control = None

    def control_options(self):
        """Reuse a control master: the user's own if configured, else ours."""
        if self._control is None:
            _, stdout, _ = _run(["ssh", "-G", self.ssh], 10)
            settings = dict(
                line.split(" ", 1) for line in stdout.splitlines() if " " in line
            )
            options = ["-o", "ControlMaster=auto"]
            if settings.get("controlpath", "none") == "none":
                CONTROL_DIR.mkdir(mode=0o700, exist_ok=True)
                options += ["-o", f"ControlPath={CONTROL_DIR}/%C"]
            if settings.get("controlpersist", "no") in ("no", "false", "0"):
                options += ["-o", f"ControlPersist={CONTROL_PERSIST}"]
            self._control = options
        return self._control

    def ssh_command(self, script):
        """ssh running `script` with sh, never allocating a terminal or prompting."""
        options = [*self.control_options(), "-T", "-o", "ConnectTimeout=15"]
        options += ["-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
        return ["ssh", *options, self.ssh, shlex.join(["sh", "-c", script])]

    def _prelude(self, login_path=False):
        if login_path:
            # A login shell sets up PATH the way an interactive user sees it.
            return (
                "p=$(bash -lc 'printf %s \"$PATH\"' </dev/null 2>/dev/null)"
                ' && [ -n "$p" ] && PATH=$p; export PATH\n'
            )
        if self.path:
            return f"PATH={shlex.quote(self.path)}; export PATH\n"
        return ""

    def run(self, args, timeout=60):
        return self.run_batch([args], timeout)[0]

    def act(self, args, timeout=60):
        """`Local.act` on the cluster; raises `ConnectionLost` if ssh fails."""
        script = (
            f"{self._prelude()}{shlex.join(args)} </dev/null 2>&1; "
            "printf '\\n\\036%s\\n' $?"
        )
        code, stdout, stderr = _run(self.ssh_command(script), timeout)
        output, record, status = stdout.rpartition(RECORD)
        if not record:
            raise ConnectionLost(stderr.strip() or f"ssh exited with status {code}")
        return int(status), output.strip()

    def run_batch(self, commands, timeout=120, login_path=False):
        """Run commands in parallel in one session; stdout in order, None on failure.

        Raises `ConnectionLost` when ssh itself fails.
        """
        if not commands:
            return []
        lines = [
            self._prelude(login_path),
            "d=$(mktemp -d) || exit 97",
            "trap 'rm -rf \"$d\"' EXIT",
        ]
        for index, args in enumerate(commands):
            lines.append(
                f'{{ {shlex.join(args)} >"$d/{index}" 2>/dev/null; '
                f'echo $? >"$d/{index}.rc"; }} &'
            )
        indexes = " ".join(str(index) for index in range(len(commands)))
        lines += [
            "wait",
            f"for i in {indexes}; do "
            'printf \'\\036%s %s\\n\' "$i" "$(cat "$d/$i.rc")"; cat "$d/$i"; done',
        ]
        code, stdout, stderr = _run(self.ssh_command("\n".join(lines)), timeout)
        if RECORD not in stdout:
            raise ConnectionLost(stderr.strip() or f"ssh exited with status {code}")
        outputs = [None] * len(commands)
        for record in stdout.split(RECORD)[1:]:
            header, _, output = record.partition("\n")
            index, _, returncode = header.partition(" ")
            if returncode.strip() == "0":
                outputs[int(index)] = output
        return outputs

    def run_stream(self, commands, on_result, should_stop=lambda: False):
        """Run commands over a few sessions, reporting each result as it lands.

        Commands are dealt round-robin to up to `PROBE_WORKERS` sessions, each
        running its share one after another. `on_result(index, returncode,
        output)` gets stdout and stderr together, as soon as each finishes.
        Raises `ConnectionLost` when ssh fails before any result.
        """
        if not commands:
            return
        shares = [
            list(range(len(commands)))[start::PROBE_WORKERS]
            for start in range(min(PROBE_WORKERS, len(commands)))
        ]
        with ThreadPoolExecutor(len(shares)) as pool:
            outcomes = list(
                pool.map(
                    lambda share: self._stream_session(
                        share, commands, on_result, should_stop
                    ),
                    shares,
                )
            )
        if not any(received for received, _ in outcomes) and not should_stop():
            raise ConnectionLost(next(error for _, error in outcomes if error))

    def _stream_session(self, share, commands, on_result, should_stop):
        """Run `commands[i]` for each i in `share` in one session.

        Returns (results received, ssh's error text or None).
        """
        lines = [self._prelude()]
        for index in share:
            lines.append(
                f"timeout {PROBE_TIMEOUT} {shlex.join(commands[index])} "
                f"</dev/null 2>&1; printf '\\n\\036%s %s\\n' {index} $?"
            )
        try:
            process = subprocess.Popen(
                self.ssh_command("\n".join(lines)),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return 0, "'ssh' not found"
        received = 0
        output = []
        try:
            for line in process.stdout:
                if not line.startswith(RECORD):
                    output.append(line)
                    continue
                index, _, returncode = line[1:].strip().partition(" ")
                received += 1
                on_result(int(index), int(returncode), "".join(output).strip())
                output = []
                if should_stop():
                    break
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
        if received:
            return received, None
        error = process.stderr.read().strip()
        return 0, error or f"ssh exited with status {process.returncode}"
