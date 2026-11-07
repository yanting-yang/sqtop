import unittest

from sqtop import snapshot, views

from test_snapshot import node, partition


def summary(*nodes):
    return snapshot.summarize_partitions([partition("gpu")], list(nodes), {}, [], None)[
        0
    ]


class PartitionGpuTests(unittest.TestCase):
    """What the Partitions panel shows of RCL's one node, by partition."""

    def summaries(self, partition_gpus=None):
        names = ("mig", "full", "cpu")
        nv2 = node(
            "nv2",
            partitions=names,
            cfg_gpus={
                "nvidia_b200": 4,
                "nvidia_b200_2g.45gb": 8,
                "nvidia_b200_3g.90gb": 4,
            },
            alloc_gpus={"nvidia_b200_3g.90gb": 3},
        )
        return snapshot.summarize_partitions(
            [partition(name) for name in names], [nv2], {}, [], None, partition_gpus
        )

    def test_names_the_gpus_each_partition_offers(self):
        mig, full, cpu = self.summaries({"mig": "mig", "full": "whole", "cpu": "none"})

        self.assertEqual(
            views.node_gpus(mig), "nvidia_b200_2g.45gb:8 + nvidia_b200_3g.90gb:4"
        )
        self.assertEqual(
            views.free_resources(mig),
            "8/8 nvidia_b200_2g.45gb, 1/4 nvidia_b200_3g.90gb",
        )
        self.assertEqual(views.load_percent(mig), 25)
        self.assertEqual(views.node_gpus(full), "nvidia_b200:4")
        self.assertEqual(views.free_resources(full), "4/4 nvidia_b200")
        self.assertEqual(views.load_percent(full), 0)
        # No GPUs here: its load and free are its CPUs'.
        self.assertEqual(views.node_gpus(cpu), "-")
        self.assertEqual(views.free_resources(cpu), "40/60 CPUs")
        self.assertEqual(views.load_percent(cpu), 33)
        self.assertEqual(views.node_kinds(cpu), ["1 node: 60 CPUs / 512 GB"])

    def test_without_a_rule_every_partition_shows_the_whole_node(self):
        mig, full, cpu = self.summaries()

        for summary_ in (mig, full, cpu):
            self.assertEqual(views.node_gpus(summary_), "16 GPUs (3 types)")
            self.assertEqual(views.free_resources(summary_), "13/16 GPUs (3 types)")


class PerNodeTests(unittest.TestCase):
    def test_shows_a_uniform_partition_once(self):
        p = summary(node("a"), node("b"))

        self.assertEqual(views.node_cpus(p), "60")
        self.assertEqual(views.node_mem(p), "512G")
        self.assertEqual(views.node_gpus(p), "l40s:4")

    def test_shows_two_kinds_side_by_side(self):
        # Like Killarney's H100 partitions (one larger node) and Tamia's
        # H100/H200 partitions.
        p = summary(
            node("a", cpu_efctv=48, cfg_gpus={"h100": 4}),
            node("b", cpu_efctv=64, mem_tot=1000 * 1024, cfg_gpus={"h200": 8}),
        )

        self.assertEqual(views.node_cpus(p), "48/64")
        self.assertEqual(views.node_mem(p), "512/1000G")
        self.assertEqual(views.node_gpus(p), "h100:4 / h200:8")

    def test_shows_a_range_for_more_kinds(self):
        p = summary(
            node("a", cpu_efctv=32, cfg_gpus={"a": 1}),
            node("b", cpu_efctv=48, cfg_gpus={"b": 2}),
            node("c", cpu_efctv=64, cfg_gpus={"c": 4}),
        )

        self.assertEqual(views.node_cpus(p), "32-64")
        self.assertEqual(views.node_gpus(p), "1-4 GPUs (3 kinds)")

    def test_summarizes_mig_nodes_under_their_full_names(self):
        mig = {
            "nvidia_h100_80gb_hbm3_1g.10gb": 8,
            "nvidia_h100_80gb_hbm3_2g.20gb": 4,
        }
        p = summary(node("a", cfg_gpus=mig))
        self.assertEqual(
            views.node_gpus(p),
            "nvidia_h100_80gb_hbm3_1g.10gb:8 + nvidia_h100_80gb_hbm3_2g.20gb:4",
        )

        mig["nvidia_h100_80gb_hbm3_3g.40gb"] = 4
        p = summary(node("a", cfg_gpus=mig))
        self.assertEqual(views.node_gpus(p), "16 GPUs (3 types)")

    def test_marks_cpu_only_nodes(self):
        self.assertEqual(views.node_gpus(summary(node("a", cfg_gpus={}))), "-")
        p = summary(node("a", cfg_gpus={}), node("b"))
        self.assertEqual(views.node_gpus(p), "0 / l40s:4")

    def test_shows_nothing_for_a_partition_without_nodes(self):
        p = summary()

        self.assertEqual(
            (views.node_cpus(p), views.node_mem(p), views.node_gpus(p)), ("-", "-", "-")
        )

    def test_lists_each_kind_with_its_node_count(self):
        p = summary(node("a"), node("b"), node("c", cpu_efctv=96))

        self.assertEqual(
            views.node_kinds(p),
            [
                "2 nodes: 60 CPUs / 512 GB / l40s:4",
                "1 node: 96 CPUs / 512 GB / l40s:4",
            ],
        )


def job(**fields):
    return {
        "state": "RUNNING",
        "start": "2026-10-14T08:42:05",
        "nodelist": "kn001",
        "sched_nodes": "(null)",
        "reason": "None",
        **fields,
    }


class JobTests(unittest.TestCase):
    def test_shows_when_a_job_started_or_will(self):
        self.assertEqual(views.job_start_time(job()), "2026-10-14 08:42")
        self.assertEqual(views.job_start_time(job(), compact=True), "10-14 08:42")
        pending = job(state="PENDING", start="N/A")
        self.assertEqual(views.job_start_time(pending), "-")

    def test_shows_where_a_job_runs_or_is_planned(self):
        self.assertEqual(views.job_nodes(job()), ("kn001", False))
        planned = job(state="PENDING", nodelist="", sched_nodes="kn134")
        self.assertEqual(views.job_nodes(planned), ("kn134", True))
        unplanned = job(state="PENDING", nodelist="", sched_nodes="(null)")
        self.assertEqual(views.job_nodes(unplanned), ("-", False))

    def test_shows_the_time_left(self):
        self.assertEqual(views.job_time_left(job(time_left="3:18")), "0-00:03:18")
        self.assertEqual(views.job_time_left(job(time_left="1-00:00:00")), "1-00:00:00")
        self.assertEqual(views.job_time_left(job(time_left="UNLIMITED")), "UNLIMITED")

    def test_gives_a_reason_only_while_waiting(self):
        self.assertEqual(views.job_reason(job()), "-")
        self.assertEqual(views.job_reason(job(reason="Priority")), "Priority")


if __name__ == "__main__":
    unittest.main()
