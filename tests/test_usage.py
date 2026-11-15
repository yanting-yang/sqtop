import unittest

from fakes import FakeHost

from sqtop import hosts, usage
from sqtop.usage import Row, Sample

# `sstat -a -n -P -j <job> -o JobID,TRESUsageInTot,TRESUsageInMax` on Nibi, for
# a job with 32 CPUs, 128G and 8 H100s whose shell is step 2: the extern step
# has no CPU time, and the site gathers no GPU use.
NIBI = (
    "25570255.extern|energy=0|energy=0\n"
    "25570255.batch|cpu=00:00:00,energy=0,fs/disk=26906,mem=792K,pages=1,vmem=0|"
    "cpu=00:00:00,energy=0,fs/disk=26906,mem=1324K,pages=1,vmem=0\n"
    "25570255.2|cpu=22-10:29:19,energy=0,fs/disk=3736339,mem=21937792K,"
    "pages=17982346,vmem=0|cpu=22-10:29:19,energy=0,fs/disk=3736339,"
    "mem=134259892K,pages=17982346,vmem=0\n"
)
NIBI_HELD = "cpu=32,mem=128G,node=1,billing=97600,gres/gpu=8,gres/gpu:h100=8|\n"


def sample(taken=0.0, steps=None, **fields):
    """A job with 8 CPUs and 64G, which has used 100 CPU seconds."""
    values = {
        "taken": taken,
        "steps": {"7.batch": 100.0} if steps is None else steps,
        "mem": 16 * 1024.0,
        "mem_peak": 20 * 1024.0,
        "gpu_util": None,
        "gpu_mem": None,
        "cpus": 8,
        "mem_total": 64 * 1024.0,
        "gpus": {},
    }
    return Sample(**{**values, **fields})


class TakeTests(unittest.TestCase):
    def host(self, **responses):
        return FakeHost(
            {
                "sstat": NIBI,
                "squeue -h -j": NIBI_HELD,
                "date": "1794688203.637\n",
                **responses,
            }
        )

    def test_reads_a_jobs_usage_next_to_what_it_holds(self):
        host = self.host()

        taken = usage.take(host, "25570255")

        self.assertEqual(taken.taken, 1794688203.637)
        self.assertEqual(taken.steps, {"25570255.batch": 0.0, "25570255.2": 1938559.0})
        self.assertEqual(round(taken.mem), 21424)
        self.assertEqual(round(taken.mem_peak), 131113)
        self.assertIsNone(taken.gpu_util)
        self.assertIsNone(taken.gpu_mem)
        self.assertEqual(
            (taken.cpus, taken.mem_total, taken.gpus), (32, 131072.0, {"h100": 8})
        )
        # One session: sstat, the cluster's clock and what the job holds.
        self.assertEqual(
            host.batches,
            [
                [
                    [
                        "sstat",
                        "-a",
                        "-n",
                        "-P",
                        "-j",
                        "25570255",
                        "-o",
                        "JobID,TRESUsageInTot,TRESUsageInMax",
                    ],
                    ["date", "+%s.%N"],
                    ["squeue", "-h", "-j", "25570255", "-O", "tres-alloc:|"],
                ]
            ],
        )

    def test_asks_what_the_job_holds_only_once(self):
        host = self.host()
        first = usage.take(host, "25570255")

        second = usage.take(host, "25570255", first)

        self.assertEqual([args[0] for args in host.batches[1]], ["sstat", "date"])
        self.assertEqual(
            (second.cpus, second.mem_total, second.gpus),
            (first.cpus, first.mem_total, first.gpus),
        )

    def test_has_nothing_for_a_job_without_running_steps(self):
        # sstat prints an error and no line for a pending or finished job.
        self.assertIsNone(usage.take(self.host(sstat=""), "1"))
        self.assertIsNone(usage.take(self.host(sstat=None), "1"))
        self.assertIsNone(usage.take(self.host(**{"squeue -h -j": ""}), "1"))

    def test_times_a_sample_by_this_clock_without_the_clusters(self):
        for clock in (None, "N\n"):  # no `date`, or one without %N
            taken = usage.take(self.host(date=clock), "25570255")
            self.assertGreater(taken.taken, 1_700_000_000)

    def test_raises_when_the_cluster_cannot_be_reached(self):
        with self.assertRaises(hosts.ConnectionLost):
            usage.take(FakeHost(lost="timed out"), "1")


class RateTests(unittest.TestCase):
    def test_keeps_the_samples_of_the_window(self):
        samples = []
        for taken in (0, 10, 20, 30, 40):
            samples = usage.add(samples, sample(taken))

        self.assertEqual([s.taken for s in samples], [10, 20, 30, 40])
        # After a pause, the earlier ones are too old to measure from.
        self.assertEqual([s.taken for s in usage.add(samples, sample(500))], [500])
        # A clock that went back starts over.
        self.assertEqual([s.taken for s in usage.add(samples, sample(5))], [5])

    def test_measures_cpus_kept_busy_between_samples(self):
        first = sample(0, {"7.batch": 100.0})
        last = sample(10, {"7.batch": 140.0})

        self.assertEqual(usage.busy_cpus([first, last]), 4.0)
        # Over the whole span of the samples, not only the last interval.
        middle = sample(5, {"7.batch": 139.0})
        self.assertEqual(usage.busy_cpus([first, middle, last]), 4.0)

    def test_needs_two_samples_some_time_apart(self):
        self.assertIsNone(usage.busy_cpus([sample()]))
        self.assertIsNone(usage.busy_cpus([sample(3), sample(3)]))

    def test_counts_steps_that_started_and_not_ones_that_ended(self):
        first = sample(0, {"7.batch": 100.0, "7.0": 5000.0})
        last = sample(10, {"7.batch": 110.0, "7.1": 20.0})

        # 10s by the batch step and 20s by the new step; step 0 ended, so its
        # CPU time left the total without the job having given any back.
        self.assertEqual(usage.busy_cpus([first, last]), 3.0)


class RowTests(unittest.TestCase):
    def test_measures_cpu_use_once_there_are_two_samples(self):
        first = sample(0, {"7.batch": 100.0})

        self.assertEqual(
            usage.rows([first])[0], Row("CPU", None, "measuring, of 8 cores")
        )
        last = sample(10, {"7.batch": 124.0})
        self.assertEqual(
            usage.rows([first, last])[0], Row("CPU", 0.3, "2.4 of 8 cores")
        )
        # More than it holds (threads beyond its cores) still fills the bar.
        busy = sample(10, {"7.batch": 200.0})
        self.assertEqual(
            usage.rows([first, busy])[0], Row("CPU", 1.0, "10.0 of 8 cores")
        )

    def test_shows_memory_against_what_the_job_holds(self):
        self.assertEqual(
            usage.rows([sample()])[1],
            Row("Memory", 0.25, "16.0 of 64 GB · peak 20.0 GB", limit=True),
        )
        unknown = sample(mem_total=None)
        self.assertEqual(
            usage.rows([unknown])[1], Row("Memory", None, "16.0 GB · peak 20.0 GB")
        )

    def test_has_no_gpu_rows_for_a_job_without_gpus(self):
        self.assertEqual(
            [row.label for row in usage.rows([sample()])], ["CPU", "Memory"]
        )

    def test_says_when_the_site_gathers_no_gpu_use(self):
        held = sample(gpus={"h100": 8})

        self.assertEqual(
            usage.rows([held])[2:],
            [Row("GPU", None, "use of the 8 h100 is not reported by Slurm here")],
        )

    def test_shows_whole_gpus_by_utilization(self):
        # RCL, one whole GPU: 87% busy, holding 65022M.
        one = sample(gpus={"nvidia_b200": 1}, gpu_util=87.0, gpu_mem=65022.0)
        self.assertEqual(
            usage.rows([one])[2:],
            [
                Row("GPU", 0.87, "of 1 nvidia_b200"),
                Row("GPU mem", None, "63.5 GB in use"),
            ],
        )
        # Utilization is summed over the GPUs, so it is shared out among them.
        four = sample(gpus={"h100": 4}, gpu_util=200.0, gpu_mem=4096.0)
        self.assertEqual(usage.rows([four])[2], Row("GPU", 0.5, "of 4 h100"))
        mixed = sample(gpus={"h100": 1, "l40s": 1}, gpu_util=100.0, gpu_mem=0.0)
        self.assertEqual(usage.rows([mixed])[2], Row("GPU", 0.5, "of 2 GPUs"))

    def test_shows_mig_slices_by_memory(self):
        # RCL, one 90 GB slice holding 22432M: NVML has no utilization for it,
        # and Slurm reports 0, which would read as an idle GPU.
        held = sample(gpus={"nvidia_b200_3g.90gb": 1}, gpu_util=0.0, gpu_mem=22432.0)

        util, memory = usage.rows([held])[2:]

        self.assertEqual(
            util,
            Row(
                "GPU",
                None,
                "utilization is not reported for MIG slices (1 nvidia_b200_3g.90gb)",
            ),
        )
        self.assertEqual(memory.label, "GPU mem")
        self.assertEqual(memory.text, "21.9 of 90 GB")
        self.assertAlmostEqual(memory.fraction, 22432 / (90 * 1024))
        self.assertTrue(memory.limit)
        # Two slices hold twice the memory.
        two = sample(gpus={"nvidia_b200_2g.45gb": 2}, gpu_util=0.0, gpu_mem=1024.0)
        self.assertEqual(usage.rows([two])[3].text, "1.0 of 90 GB")


if __name__ == "__main__":
    unittest.main()
