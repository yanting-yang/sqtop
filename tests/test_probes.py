import datetime
import shlex
import unittest
from unittest.mock import Mock

from fakes import FakeHost, scheduler_reply

from sqtop import probes, profiles, snapshot
from sqtop.probes import Grid, Request, Settings
from sqtop.snapshot import Limit, Scope

GENERIC = profiles.Profile()


class DurationTextTests(unittest.TestCase):
    def test_reads_compact_and_slurm_walltimes(self):
        for text, minutes in [
            ("90m", 90),
            ("3h", 180),
            ("1d", 1440),
            ("1d12h", 2160),
            ("3:00:00", 180),
            ("1-00:00:00", 1440),
            ("30", 30),
        ]:
            with self.subTest(text=text):
                self.assertEqual(probes.parse_walltime(text), minutes)

    def test_lists_distinct_sorted_walltimes(self):
        self.assertEqual(probes.parse_walltimes("1d, 3h 3:00:00,90m"), (90, 180, 1440))
        self.assertEqual(probes.parse_walltimes(""), ())

    def test_rejects_invalid_walltimes(self):
        for text in ("soon", "0m", "3h,x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                probes.parse_walltimes(text)

    def test_formats_waits_from_now(self):
        now = datetime.datetime(2026, 10, 7, 12, 0)

        self.assertEqual(probes.format_wait(now, now), "now")
        self.assertEqual(
            probes.format_wait(now - datetime.timedelta(hours=1), now), "now"
        )
        self.assertEqual(
            probes.format_wait(now + datetime.timedelta(seconds=61), now),
            "0-00:02:00",
        )
        self.assertEqual(
            probes.format_wait(now + datetime.timedelta(hours=27), now),
            "1-03:00:00",
        )


class RequestPlanTests(unittest.TestCase):
    def test_probes_powers_of_two_up_to_the_ceiling(self):
        self.assertEqual(probes.counts_for(8, 8, "pow2"), [1, 2, 4, 8])
        self.assertEqual(probes.counts_for(6, 8, "pow2"), [1, 2, 4, 6])
        self.assertEqual(probes.counts_for(1, 8, "pow2"), [1])

    def test_can_probe_every_count_or_only_whole_nodes(self):
        self.assertEqual(probes.counts_for(3, 4, "all"), [1, 2, 3])
        self.assertEqual(probes.counts_for(1, 4, "whole_node"), [4])

    def test_can_probe_one_gpu_and_the_whole_node(self):
        self.assertEqual(probes.counts_for(8, 8, "one_or_node"), [1, 8])
        self.assertEqual(probes.counts_for(1, 1, "one_or_node"), [1])

    def test_lists_gpu_requests_then_cpu_only(self):
        requests = probes.plan_requests({"h100": 4, "l40s": 2}, {"h100": 2})

        self.assertEqual(
            [request.label for request in requests],
            ["1x h100", "2x h100", "1x l40s", "2x l40s", "CPU only"],
        )
        self.assertEqual(requests[1].directives(), ["--gpus-per-node=h100:2"])
        self.assertEqual(requests[-1].directives(), [])

    def test_asks_for_untyped_gpus_without_a_model(self):
        requests = probes.plan_requests({"gpu": 2}, {})

        self.assertEqual(requests[1].directives(), ["--gpus-per-node=2"])


class MaxGpuCountTests(unittest.TestCase):
    CAPACITIES = {"nvidia_b200": 4, "nvidia_b200_2g.45gb": 8}

    def test_caps_each_model_at_its_own_limit(self):
        limits = [Limit("QOS per user", "", "tres:gres/gpu:nvidia_b200", 1)]

        self.assertEqual(
            probes.max_gpu_counts(self.CAPACITIES, limits),
            {"nvidia_b200": 1, "nvidia_b200_2g.45gb": 8},
        )

    def test_caps_every_model_at_the_combined_gpu_limit(self):
        limits = [
            Limit("QOS per job", "", "tres:gres/gpu", 2),
            Limit("Account lab", "", "tres:gres/gpu", 6),
        ]

        self.assertEqual(
            probes.max_gpu_counts(self.CAPACITIES, limits),
            {"nvidia_b200": 2, "nvidia_b200_2g.45gb": 2},
        )

    def test_ignores_unlimited_and_unknown_caps(self):
        limits = [
            Limit("QOS per user", "", "tres:gres/gpu", None),
            Limit("QOS per user", "", "tres:gres/gpu", "?"),
        ]

        self.assertEqual(
            probes.max_gpu_counts(self.CAPACITIES, limits), self.CAPACITIES
        )

    def test_still_probes_one_gpu_when_the_cap_is_zero(self):
        limits = [Limit("QOS per user", "", "tres:gres/gpu", 0)]

        self.assertEqual(
            probes.max_gpu_counts({"nvidia_b200": 4}, limits), {"nvidia_b200": 1}
        )


class WalltimePlanTests(unittest.TestCase):
    PARTITIONS = [
        {"max_time": 180, "usable": True},
        {"max_time": 720, "usable": True},
        {"max_time": 180, "usable": True},
        {"max_time": 10080, "usable": False},
        {"max_time": None, "usable": True},
    ]

    def test_derives_walltimes_from_usable_partition_limits(self):
        self.assertEqual(probes.plan_walltimes(self.PARTITIONS), [180, 720])

    def test_ignores_preemptible_partitions(self):
        partitions = [
            *self.PARTITIONS,
            {"max_time": 175680, "usable": True, "preempt_mode": "REQUEUE"},
        ]

        self.assertEqual(probes.plan_walltimes(partitions), [180, 720])

    def test_prefers_the_override(self):
        self.assertEqual(
            probes.plan_walltimes(self.PARTITIONS, (1440, 60, 1440)), [60, 1440]
        )

    def test_falls_back_when_no_partition_sets_a_limit(self):
        self.assertEqual(
            probes.plan_walltimes([{"max_time": None}]),
            list(probes.DEFAULT_WALLTIMES),
        )

    def test_thins_many_tiers_keeping_both_ends(self):
        partitions = [{"max_time": minutes} for minutes in range(60, 660, 60)]

        walltimes = probes.plan_walltimes(partitions)

        self.assertEqual(len(walltimes), probes.MAX_DERIVED_WALLTIMES)
        self.assertEqual((walltimes[0], walltimes[-1]), (60, 600))


class GridTests(unittest.TestCase):
    def grid(self, scope=Scope(), settings=Settings(), max_wall=None):
        return Grid(
            scope=scope,
            settings=settings,
            requests=[Request("l40s", 2), Request()],
            walltimes=[180, 1440],
            max_wall=max_wall,
        )

    def test_builds_short_directives_with_scope_and_extra_options(self):
        grid = self.grid(
            Scope("lab", "high", True, False),
            Settings(cpus=8, mem="64G", extra=("--constraint=ib",)),
        )

        self.assertEqual(
            grid.probe_directives(0, 1),
            [
                "--test-only",
                "--qos=high",
                "--gpus-per-node=l40s:2",
                "-c8",
                "--mem=64G",
                "-t1-00:00:00",
                "--constraint=ib",
            ],
        )
        self.assertEqual(
            grid.probe_directives(1, 0),
            [
                "--test-only",
                "--qos=high",
                "-c8",
                "--mem=64G",
                "-t0-03:00:00",
                "--constraint=ib",
            ],
        )

    def test_asks_for_a_whole_node_instead_of_cpus_and_memory(self):
        settings = Settings(whole_node=True, cpus=8, mem="64G")
        directives = self.grid(settings=settings).probe_directives(0, 0)

        self.assertEqual(
            directives,
            [
                "--test-only",
                "--gpus-per-node=l40s:2",
                "--exclusive",
                "--mem=0",
                "-t0-03:00:00",
            ],
        )
        self.assertEqual(settings.describe(), "sbatch --exclusive --mem=0")
        self.assertEqual(Settings(mem="64G").describe(), "sbatch -c32 --mem=64G")

    def test_adds_email_to_the_run_command_but_not_the_probe(self):
        grid = self.grid(
            settings=Settings(mail_user="me@example.com", extra=("--constraint=ib",))
        )

        self.assertEqual(
            grid.run_directives(1, 0),
            [
                "-c32",
                "--mem=128G",
                "-t0-03:00:00",
                "--mail-type=BEGIN,END,FAIL",
                "--mail-user=me@example.com",
                "--constraint=ib",
            ],
        )
        self.assertEqual(
            grid.probe_directives(1, 0),
            ["--test-only", "-c32", "--mem=128G", "-t0-03:00:00", "--constraint=ib"],
        )
        self.assertNotIn("--mail-user=me@example.com", self.grid().run_directives(1, 0))

    def test_skips_walltimes_over_the_wall_cap(self):
        grid = self.grid(max_wall=180)

        self.assertFalse(grid.skipped(0))
        self.assertTrue(grid.skipped(1))
        self.assertEqual(grid.cells(), [(0, 0), (1, 0)])

    def test_runs_sbatch_and_returns_the_copyable_command(self):
        host = FakeHost(probe=lambda args: scheduler_reply("2030-01-01T00:00:00"))

        probes.run_grid(self.grid(max_wall=180), host)

        self.assertEqual(
            host.streamed[0][0],
            [
                "sbatch",
                "--test-only",
                "--gpus-per-node=l40s:2",
                "-c32",
                "--mem=128G",
                "-t0-03:00:00",
                "--output=/dev/null",
                "--wrap=sleep infinity",
            ],
        )
        result = probes.run_grid(self.grid(), host).results[0, 0]
        self.assertTrue(result.ok)
        self.assertEqual(result.start, datetime.datetime(2030, 1, 1))
        self.assertEqual(result.partition, "gpu")
        self.assertEqual(result.message, "")
        self.assertEqual(
            result.probe,
            "sbatch --test-only --gpus-per-node=l40s:2 -c32 --mem=128G -t0-03:00:00 "
            '--output=/dev/null --wrap="sleep infinity"',
        )
        self.assertEqual(result.output, scheduler_reply("2030-01-01T00:00:00")[1])
        self.assertEqual(shlex.split(result.command), list(result.run))
        self.assertEqual(
            shlex.split(result.command),
            [
                "sbatch",
                "--gpus-per-node=l40s:2",
                "-c32",
                "--mem=128G",
                "-t0-03:00:00",
                "--output=/dev/null",
                "--wrap=sleep infinity",
            ],
        )

    def test_runs_srun_with_the_login_shell_and_cleans_the_reason(self):
        grid = self.grid(settings=Settings(command="srun"), max_wall=180)
        grid.shell = "zsh"
        host = FakeHost(probe=lambda args: (1, "srun: error: nope."))

        result = probes.run_grid(grid, host).results[1, 0]

        self.assertEqual(
            host.streamed[0][1],
            ["srun", "--test-only", "-c32", "--mem=128G", "-t0-03:00:00", "zsh"],
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.message, "nope")
        self.assertEqual(
            result.probe, "srun --test-only -c32 --mem=128G -t0-03:00:00 zsh"
        )
        self.assertEqual(result.output, "srun: error: nope.")
        self.assertEqual(result.command, "srun -c32 --mem=128G -t0-03:00:00 --pty zsh")

    def test_runs_every_cell_and_reports_each_result(self):
        grid = self.grid(max_wall=180)
        seen = []

        probes.run_grid(
            grid,
            FakeHost(),
            lambda row, column, result: seen.append((row, column)),
        )

        self.assertEqual(sorted(seen), [(0, 0), (1, 0)])
        self.assertEqual(sorted(grid.results), [(0, 0), (1, 0)])

    def test_stops_once_asked(self):
        grid = self.grid()

        probes.run_grid(grid, FakeHost(), should_stop=lambda: True)

        self.assertEqual(grid.results, {})

    def test_recleans_rejections_once_an_accepted_reply_shows_the_banner(self):
        replies = iter(
            [
                (1, RCL_BANNER + "sbatch: error: too big\n"),
                (0, RCL_BANNER + "sbatch: Job 2 to start at 2030-01-01T00:00:00\n"),
            ]
        )
        grid = self.grid(max_wall=180)
        shared = grid.banner

        probes.run_grid(grid, FakeHost(probe=lambda args: next(replies)))

        self.assertEqual(grid.results[0, 0].message, "too big")
        self.assertTrue(grid.results[1, 0].ok)
        # A later grid sharing the set cleans its reasons from the start.
        later = self.grid(max_wall=180)
        later.banner = shared
        host = FakeHost(probe=lambda args: (1, RCL_BANNER + "sbatch: error: no\n"))
        self.assertEqual(probes.run_grid(later, host).results[0, 0].message, "no")

    def test_says_something_when_a_reply_is_all_noise(self):
        host = FakeHost(probe=lambda args: (1, "sbatch: error: ----------"))

        result = probes.run_grid(self.grid(max_wall=180), host).results[0, 0]

        self.assertEqual(result.message, "rejected without a reason")


class BuildGridTests(unittest.TestCase):
    def test_sizes_the_grid_from_usable_nodes_partitions_and_limits(self):
        nodes = [
            {"name": "a", "cfg_gpus": {"l40s": 4}, "partitions": ["gpu"]},
            {"name": "b", "cfg_gpus": {"h100": 8}, "partitions": ["private"]},
        ]
        partitions = [
            {"name": "gpu", "max_time": 180, "usable": True},
            {"name": "private", "max_time": 4320, "usable": False},
        ]
        snap = Mock(nodes=nodes, partitions=partitions, gpu_names={"l40s": "L40S"})
        snap.limits.return_value = [
            Limit("QOS per user", "GPUs", "tres:gres/gpu", 2),
            Limit("QOS per job", "wall time", "wall", 120),
        ]

        info = snapshot.ClusterInfo("c", "1", "me", "login", None, shell="zsh")

        grid = probes.build_grid(
            snap, Scope("lab", "normal"), Settings(), GENERIC, info
        )

        self.assertEqual(
            [request.label for request in grid.requests],
            ["1x L40S", "2x L40S", "CPU only"],
        )
        self.assertEqual(grid.requests[0].directives(), ["--gpus-per-node=l40s:1"])
        self.assertEqual(grid.walltimes, [180])
        self.assertEqual(grid.max_wall, 120)
        self.assertEqual(grid.shell, "zsh")

        # A whole node probes all of its GPUs, whatever the caps or profile say.
        whole = probes.build_grid(
            snap, Scope("lab", "normal"), Settings(whole_node=True), GENERIC, info
        )
        self.assertEqual(
            [request.label for request in whole.requests], ["4x L40S", "CPU only"]
        )

    def test_asks_for_whole_gpu_nodes_where_gpus_go_by_node(self):
        nodes = [{"name": "a", "cfg_gpus": {"h100": 4}, "partitions": ["gpu"]}]
        partitions = [{"name": "gpu", "max_time": 180, "usable": True}]
        snap = Mock(nodes=nodes, partitions=partitions, gpu_names={})
        snap.limits.return_value = []
        info = snapshot.ClusterInfo("tamia", "1", "me", "login", None)
        profile = profiles.Profile("tamia", gpu_counts="whole_node")

        grid = probes.build_grid(snap, Scope(), Settings(), profile, info)

        self.assertEqual(
            [request.label for request in grid.requests], ["4x h100", "CPU only"]
        )
        # Every CPU and all the memory with the GPUs; CPU-only keeps -c/--mem.
        self.assertEqual(
            grid.probe_directives(0, 0),
            [
                "--test-only",
                "--gpus-per-node=h100:4",
                "--exclusive",
                "--mem=0",
                "-t0-03:00:00",
            ],
        )
        self.assertEqual(
            grid.run_directives(1, 0), ["-c32", "--mem=128G", "-t0-03:00:00"]
        )
        self.assertTrue(grid.gpus_by_node)
        self.assertFalse(
            probes.build_grid(snap, Scope(), Settings(), GENERIC, info).gpus_by_node
        )
        # Asking for whole nodes covers every row, so it is not noted twice.
        whole = probes.build_grid(
            snap, Scope(), Settings(whole_node=True), profile, info
        )
        self.assertFalse(whole.gpus_by_node)
        self.assertEqual(
            whole.run_directives(1, 0), ["--exclusive", "--mem=0", "-t0-03:00:00"]
        )

    def test_leaves_cpus_and_memory_to_a_site_that_sizes_them_by_gpu(self):
        # Trillium GPU: one GPU or a whole node, and --mem is refused.
        nodes = [{"name": "a", "cfg_gpus": {"h100": 4}, "partitions": ["gpu"]}]
        partitions = [{"name": "gpu", "max_time": 180, "usable": True}]
        snap = Mock(nodes=nodes, partitions=partitions, gpu_names={})
        snap.limits.return_value = []
        info = snapshot.ClusterInfo("grillium", "1", "me", "login", None)
        profile = profiles.Profile(
            "trillium-gpu", gpu_counts="one_or_node", resources="by_gpu"
        )

        grid = probes.build_grid(snap, Scope(), Settings(), profile, info)

        self.assertEqual(
            [request.label for request in grid.requests],
            ["1x h100", "4x h100", "CPU only"],
        )
        self.assertEqual(
            grid.probe_directives(0, 0),
            ["--test-only", "--gpus-per-node=h100:1", "-t0-03:00:00"],
        )
        # CPU only keeps the request's CPUs and memory.
        self.assertEqual(
            grid.run_directives(2, 0), ["-c32", "--mem=128G", "-t0-03:00:00"]
        )
        self.assertTrue(grid.sized_by_gpu)
        self.assertFalse(
            probes.build_grid(snap, Scope(), Settings(), GENERIC, info).sized_by_gpu
        )
        # A whole node still asks for every CPU and all the memory.
        whole = probes.build_grid(
            snap, Scope(), Settings(whole_node=True), profile, info
        )
        self.assertFalse(whole.sized_by_gpu)
        self.assertEqual(
            whole.run_directives(0, 0),
            ["--gpus-per-node=h100:4", "--exclusive", "--mem=0", "-t0-03:00:00"],
        )


# Printed with every RCL reply, accepted or not, once the submit filter
# passes a request.
RCL_BANNER = (
    "sbatch: error: " + "=" * 71 + "\n"
    "sbatch: error: STORAGE POLICY: Please ensure large job outputs "
    "are written to /scratch\n"
    "sbatch: error: Please delete files from /scratch after you are "
    "done with your work.   \n"
    "sbatch: error: " + "=" * 71 + "\n"
)
# Printed with every Fir and Tamia reply that asks for memory in G.
TAMIA_NOTE = (
    "sbatch: NOTE: Your memory request of 32768.0M was likely submitted as 32.0G. "
    "Please note that Slurm interprets memory requests denominated in G as "
    "multiples of 1024M, not 1000M.\n"
)


class CleanMessageTests(unittest.TestCase):
    def test_strips_prefixes_and_unspecified_error_boilerplate(self):
        raw = (
            "srun: error: Limited account 'you': GPUs - at most 1 MIG slice per job "
            "(you requested 2). allocation failure: Unspecified error"
        )

        self.assertEqual(
            probes.clean_message(raw),
            "Limited account 'you': GPUs - at most 1 MIG slice per job (you requested 2)",
        )

    def test_leaves_out_lines_an_accepted_reply_printed_too(self):
        accepted = RCL_BANNER + "sbatch: Job 1 to start at 2030-01-01T00:00:00\n"
        banner = set()
        grid = Grid(Scope(), Settings(), [Request()], [60], banner=banner)
        probes.learn_banner(grid, accepted)

        self.assertEqual(
            probes.clean_message(
                RCL_BANNER + "allocation failure: Invalid qos specification\n",
                banner,
            ),
            "allocation failure: Invalid qos specification",
        )
        # The start line is unique to each job, so it is not kept.
        self.assertFalse(any("to start at" in line for line in banner))

    def test_strips_tamia_colour_codes_and_memory_note(self):
        rejected = (
            TAMIA_NOTE
            + "sbatch: error: \x1b[00;31m----------------------------------\n"
            "sbatch: error: The h100 GPUs are only allocated by node.\n"
            "sbatch: error: ----------------------------------\x1b[00m\n"
            "allocation failure: Requested node configuration is not available\n"
        )

        self.assertEqual(
            probes.clean_message(rejected, set(probes.reply_lines(TAMIA_NOTE))),
            "The h100 GPUs are only allocated by node. allocation failure: "
            "Requested node configuration is not available",
        )
        self.assertTrue(probes.clean_message(rejected).startswith("NOTE: Your memory"))

    def test_leaves_local_errors_alone(self):
        self.assertEqual(probes.clean_message("'srun' not found"), "'srun' not found")


if __name__ == "__main__":
    unittest.main()
