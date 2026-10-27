import io
import shlex
import subprocess
import unittest
from unittest.mock import Mock, patch

from sqtop import hosts


def completed(stdout="", stderr="", returncode=0):
    return Mock(stdout=stdout, stderr=stderr, returncode=returncode)


class LocalTests(unittest.TestCase):
    @patch.object(hosts.subprocess, "run")
    def test_returns_stdout_and_never_reads_the_terminal(self, run):
        run.return_value = completed("out\n")

        self.assertEqual(hosts.Local().run(["scontrol", "show", "node"]), "out\n")
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_reports_failures_as_none(self):
        for effect in (
            completed(returncode=1),
            FileNotFoundError(),
            subprocess.TimeoutExpired("x", 1),
        ):
            with (
                self.subTest(effect=effect),
                patch.object(
                    hosts.subprocess,
                    "run",
                    **(
                        {"side_effect": effect}
                        if isinstance(effect, Exception)
                        else {"return_value": effect}
                    ),
                ),
            ):
                self.assertIsNone(hosts.Local().run(["squeue"]))

    @patch.object(hosts.subprocess, "run")
    def test_runs_a_batch_in_order(self, run):
        run.side_effect = lambda args, **kwargs: completed(
            "" if args[0] == "bad" else args[0], returncode=args[0] == "bad"
        )

        self.assertEqual(
            hosts.Local().run_batch([["a"], ["bad"], ["c"]]), ["a", None, "c"]
        )

    @patch.object(hosts.subprocess, "run")
    def test_streams_combined_output_with_each_index(self, run):
        run.side_effect = lambda args, **kwargs: completed(
            f"{args[0]} out", f"{args[0]} err", returncode=7
        )
        seen = []

        hosts.Local().run_stream([["a"], ["b"]], lambda *result: seen.append(result))

        self.assertEqual(sorted(seen), [(0, 7, "a out\na err"), (1, 7, "b out\nb err")])

    def test_streams_missing_commands_and_timeouts_as_results(self):
        seen = []
        with patch.object(hosts.subprocess, "run", side_effect=FileNotFoundError):
            hosts.Local().run_stream([["sbatch"]], lambda *r: seen.append(r))
        with patch.object(
            hosts.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 30)
        ):
            hosts.Local().run_stream([["srun"]], lambda *r: seen.append(r))

        self.assertEqual(
            seen, [(0, 127, "'sbatch' not found"), (0, 124, "Timed out after 30s")]
        )


class SubmitTests(unittest.TestCase):
    @patch.object(hosts.subprocess, "run")
    def test_returns_the_status_with_both_outputs(self, run):
        run.return_value = completed("", "sbatch: error: no", returncode=1)

        self.assertEqual(hosts.Local().act(["sbatch"]), (1, "sbatch: error: no"))


class RemoteTests(unittest.TestCase):
    def remote(
        self, ssh_config="controlmaster auto\ncontrolpath /s/%C\ncontrolpersist 600\n"
    ):
        remote = hosts.Remote("fir")
        with patch.object(hosts.subprocess, "run", return_value=completed(ssh_config)):
            remote.control_options()
        return remote

    def test_reuses_the_users_own_control_master(self):
        self.assertEqual(self.remote().control_options(), ["-o", "ControlMaster=auto"])

    def test_adds_a_control_master_when_the_config_has_none(self):
        remote = self.remote("controlmaster false\ncontrolpath none\n")

        options = remote.control_options()

        self.assertIn("ControlMaster=auto", options)
        self.assertIn(f"ControlPath={hosts.CONTROL_DIR}/%C", options)
        self.assertIn(f"ControlPersist={hosts.CONTROL_PERSIST}", options)

    def test_runs_scripts_with_sh_without_a_terminal_or_prompts(self):
        command = self.remote().ssh_command("echo hi")

        self.assertEqual(command[0], "ssh")
        self.assertIn("-T", command)
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-2], "fir")
        self.assertEqual(shlex.split(command[-1]), ["sh", "-c", "echo hi"])

    @patch.object(hosts.subprocess, "run")
    def test_runs_a_batch_in_one_session(self, run):
        run.return_value = completed("\x1e0 0\nnodes\n\x1e1 1\n\x1e2 0\nparts")
        remote = self.remote()
        remote.path = "/opt/slurm/bin:/usr/bin"

        outputs = remote.run_batch([["scontrol", "show", "node"], ["bad"], ["x"]])

        self.assertEqual(outputs, ["nodes\n", None, "parts"])
        run.assert_called_once()
        script = shlex.split(run.call_args.args[0][-1])[2]
        self.assertIn("PATH=/opt/slurm/bin:/usr/bin; export PATH", script)
        self.assertIn("scontrol show node", script)

    @patch.object(hosts.subprocess, "run")
    def test_reads_the_login_path_when_asked(self, run):
        run.return_value = completed("\x1e0 0\n/usr/bin\n")

        self.remote().run_batch([["printenv", "PATH"]], login_path=True)

        script = shlex.split(run.call_args.args[0][-1])[2]
        self.assertIn("bash -lc", script)

    @patch.object(hosts.subprocess, "run")
    def test_raises_when_ssh_fails(self, run):
        run.return_value = completed(
            stderr="Permission denied (publickey).", returncode=255
        )

        with self.assertRaisesRegex(hosts.ConnectionLost, "Permission denied"):
            self.remote().run_batch([["scontrol", "show", "node"]])

    @patch.object(hosts.subprocess, "run")
    def test_submits_on_the_cluster(self, run):
        run.return_value = completed("Submitted batch job 5\n\n\x1e0\n")
        remote = self.remote()
        remote.path = "/opt/slurm/bin"

        self.assertEqual(remote.act(["sbatch", "-t1"]), (0, "Submitted batch job 5"))
        script = shlex.split(run.call_args.args[0][-1])[2]
        self.assertIn("PATH=/opt/slurm/bin; export PATH", script)
        self.assertIn("sbatch -t1 </dev/null 2>&1", script)

        run.return_value = completed(stderr="Connection closed.", returncode=255)
        with self.assertRaisesRegex(hosts.ConnectionLost, "Connection closed"):
            remote.act(["sbatch"])

    def popen(self, stdout, returncode=0, stderr=""):
        process = Mock(returncode=returncode)
        # Like a pipe, and unlike str.splitlines, StringIO splits only on newlines.
        process.stdout = io.StringIO(stdout)
        process.stderr.read.return_value = stderr
        process.poll.return_value = returncode
        return process

    def test_streams_results_from_parallel_sessions(self):
        def popen(args, **kwargs):
            script = shlex.split(args[-1])[2]
            indexes = [int(part.split()[0]) for part in script.split("' ")[1:]]
            lines = "".join(
                f"output {index}\n\n\x1e{index} {index % 2}\n" for index in indexes
            )
            return self.popen(lines)

        seen = []
        with patch.object(hosts.subprocess, "Popen", side_effect=popen) as spawn:
            self.remote().run_stream(
                [["sbatch", str(i)] for i in range(6)], lambda *r: seen.append(r)
            )

        self.assertEqual(spawn.call_count, hosts.PROBE_WORKERS)
        self.assertEqual(sorted(seen), [(i, i % 2, f"output {i}") for i in range(6)])

    def test_raises_when_no_session_returns_anything(self):
        with patch.object(
            hosts.subprocess,
            "Popen",
            return_value=self.popen("", 255, "Connection refused"),
        ):
            with self.assertRaisesRegex(hosts.ConnectionLost, "Connection refused"):
                self.remote().run_stream([["sbatch"]], lambda *r: None)


if __name__ == "__main__":
    unittest.main()
