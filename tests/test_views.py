import unittest

from sqtop import snapshot, views

from test_snapshot import node, partition


def summary(*nodes):
    return snapshot.summarize_partitions([partition("gpu")], list(nodes), {}, [], None)[
        0
    ]


class ResourceTests(unittest.TestCase):
    """The rows of a single node's hardware table."""

    def rows(self, pending):
        nv2 = node("nv2", cfg_gpus={"nvidia_b200": 4}, alloc_gpus={"nvidia_b200": 1})
        return snapshot.summarize_resources([nv2], pending)

    def test_shows_free_and_pending_per_resource(self):
        cpus, mem, gpu = self.rows({None: 2})

        self.assertEqual(views.resource_free(cpus), "40/60")
        self.assertEqual(views.resource_free(mem), "384/512G")
        self.assertEqual(views.resource_free(gpu), "3/4")
        # Jobs asking for no GPU, no count for memory, jobs asking for the type.
        self.assertEqual(
            [views.resource_pending(row) for row in (cpus, mem, gpu)], ["2", "-", "0"]
        )
        self.assertEqual(
            views.resource_usage(mem),
            "384 free / 128 allocated / 0 unavailable of 512 GB",
        )
        self.assertEqual(
            views.resource_usage(gpu), "3 free / 1 allocated / 0 unavailable of 4"
        )
        self.assertEqual(
            [views.resource_pending(row) for row in self.rows(None)], ["?", "-", "?"]
        )

    def test_takes_the_longest_time_limit_of_usable_partitions(self):
        def limit(*partitions):
            return views.longest_time_limit(
                [{"max_time": time, "usable": usable} for time, usable in partitions]
            )

        self.assertEqual(limit((180, True), (1440, True), (10080, False)), "1-00:00:00")
        self.assertEqual(limit((180, True), (None, True)), "∞")
        # With none usable, the partitions there are.
        self.assertEqual(limit((180, False)), "0-03:00:00")
        self.assertEqual(limit(), "?")

    def test_names_the_node_and_its_partitions(self):
        self.assertEqual(views.node_state(node("nv2")), "nv2 (mixed)")
        self.assertEqual(
            views.node_state(node("nv2", state="IDLE+DRAIN")), "nv2 (unavailable)"
        )
        partitions = [
            {"name": "mig", "default": True, "usable": True, "access_note": ""},
            {
                "name": "full",
                "default": False,
                "usable": False,
                "access_note": "none of your QOS is allowed",
            },
        ]
        self.assertEqual(
            views.partition_names(partitions),
            "mig*, full (not usable: none of your QOS is allowed)",
        )


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
