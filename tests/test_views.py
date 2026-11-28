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


class SubRowTests(unittest.TestCase):
    """The rows below a partition's own: its kinds of node, then GPU types."""

    def partition(self, asked=None):
        # Like Narval's gpubase_interac: whole GPUs on nodes of two sizes and
        # MIG slices, cut two ways, on others.
        nodes = [
            node("w1", cfg_gpus={"a100": 4}, alloc_gpus={"a100": 1}),
            node("w2", cfg_gpus={"a100": 4}, alloc_gpus={"a100": 4}),
            node(
                "m1",
                cpu_efctv=48,
                cfg_gpus={"a100_1g": 7, "a100_2g": 4},
                alloc_gpus={"a100_1g": 5},
            ),
            node(
                "big",
                state="IDLE",
                cpu_efctv=64,
                mem_tot=1000 * 1024,
                cfg_gpus={"a100": 4},
                alloc_gpus={},
            ),
            node("m2", cfg_gpus={"a100_1g": 8, "a100_2g": 4}, alloc_gpus={}),
        ]
        return snapshot.summarize_partitions(
            [partition("gpu")], nodes, {}, [], None, asked
        )[0]

    def test_indents_each_row_under_its_parent(self):
        rows = list(views.sub_rows(self.partition()))

        self.assertEqual(
            [indent + label for indent, label, _, _ in rows],
            [
                "  ├ 2 nodes",
                "  ├ 1 node",
                "  │ ├ a100_1g",
                "  │ └ a100_2g",
                "  ├ 1 node",
                # Below the last kind there is no line to carry on.
                "  └ 1 node",
                "    ├ a100_1g",
                "    └ a100_2g",
            ],
        )
        # Each is named by its path, which tells apart the same type under
        # two kinds of node.
        self.assertEqual(
            [row["name"] for _, _, row, _ in rows][:3],
            [
                "gpu › 60 CPUs / 512 GB / a100:4",
                "gpu › 48 CPUs / 512 GB / a100_1g:7 / a100_2g:4",
                "gpu › 48 CPUs / 512 GB / a100_1g:7 / a100_2g:4 › a100_1g",
            ],
        )
        self.assertEqual(len({row["name"] for _, _, row, _ in rows}), len(rows))

    def test_shows_only_the_rows_below_an_opened_one(self):
        p = self.partition()
        slices = "gpu › 60 CPUs / 512 GB / a100_1g:8 / a100_2g:4"

        def labels(shown):
            return [
                indent + ("[] " if room else "") + label
                for indent, label, _, room in views.sub_rows(p, shown, slot=2)
            ]

        # Nothing until the partition is opened, then its kinds of node,
        # each with room for a button as some have rows below them.
        self.assertEqual(labels(set()), [])
        self.assertEqual(
            labels({"gpu"}),
            ["  ├ [] 2 nodes", "  ├ [] 1 node", "  ├ [] 1 node", "  └ [] 1 node"],
        )
        # An opened kind's GPU types are indented past its button; they
        # have nothing below them, so no room for one.
        self.assertEqual(
            labels({"gpu", slices})[3:],
            ["  └ [] 1 node", "      ├ a100_1g", "      └ a100_2g"],
        )
        other = "gpu › 48 CPUs / 512 GB / a100_1g:7 / a100_2g:4"
        self.assertEqual(
            labels({"gpu", other})[1:4],
            ["  ├ [] 1 node", "  │   ├ a100_1g", "  │   └ a100_2g"],
        )
        # A kind stays opened while its partition is closed.
        self.assertEqual(labels({slices}), [])

    def test_describes_a_kind_of_node_like_a_partition(self):
        p = self.partition({"gpu": {"a100": 6, "a100_2g": 2}})
        rows = [row for _, _, row, _ in views.sub_rows(p)]
        whole, slices, _, _, big, _, _, _ = rows

        # CPU/node, Mem/node, GPU/node, Free, Idle and Pend.
        self.assertEqual(
            views.sub_row_cells(whole), ("60", "512G", "a100:4", "3/8 a100", "0/2", "6")
        )
        self.assertEqual(
            views.sub_row_cells(big), ("64", "1000G", "a100:4", "4/4 a100", "1/1", "6")
        )
        # With several GPU types it stands for none: its types' rows count
        # the pending jobs.
        self.assertEqual(
            views.sub_row_cells(slices),
            (
                "48",
                "512G",
                "a100_1g:7 + a100_2g:4",
                "2/7 a100_1g, 4/4 a100_2g",
                "0/1",
                "",
            ),
        )
        self.assertEqual(views.main_resource(whole), (5, 3, 8))
        # What belongs to the partition stays, for the row's details.
        self.assertEqual(whole["max_time"], p["max_time"])
        self.assertTrue(whole["usable"])
        self.assertEqual(whole["names"], ["w1", "w2"])

    def test_gives_a_gpu_type_only_its_own_cells(self):
        p = self.partition({"gpu": {"a100": 6, "a100_2g": 2}})
        rows = [row for _, _, row, _ in views.sub_rows(p)]

        # Its nodes are those of the row above; the type is its label.
        self.assertEqual(views.sub_row_cells(rows[2]), ("", "", "7", "2/7", "", "0"))
        self.assertEqual(views.sub_row_cells(rows[3]), ("", "", "4", "4/4", "", "2"))
        self.assertEqual(views.sub_row_cells(rows[6]), ("", "", "8", "8/8", "", "0"))
        self.assertEqual(views.main_resource(rows[2]), (5, 2, 7))
        self.assertEqual(views.load_percent(rows[3]), 0)
        # squeue failed: how many ask for a type is unknown.
        unknown = [row for _, _, row, _ in views.sub_rows(self.partition())]
        self.assertEqual(views.sub_row_cells(unknown[0])[-1], "?")
        self.assertEqual(views.sub_row_cells(unknown[1])[-1], "")
        self.assertEqual(views.sub_row_cells(unknown[2])[-1], "?")

    def test_describes_kinds_of_node_without_gpus_by_their_cpus(self):
        # Like Narval's cpubase_interac: two sizes of memory.
        p = summary(
            node("c1", cfg_gpus={}, alloc_gpus={}),
            node("c2", state="IDLE", cfg_gpus={}, alloc_gpus={}, cpu_alloc=0),
            node("c3", cfg_gpus={}, alloc_gpus={}, mem_tot=1000 * 1024),
        )

        rows = list(views.sub_rows(p))

        # No kind has rows below it, so none needs room for a button.
        self.assertEqual(
            [(indent + label, room) for indent, label, _, room in rows],
            [("  ├ 2 nodes", False), ("  └ 1 node", False)],
        )
        self.assertEqual(
            [views.sub_row_cells(row) for _, _, row, _ in rows],
            [
                ("60", "512G", "-", "100/120 CPUs", "1/2", ""),
                ("60", "1000G", "-", "40/60 CPUs", "0/1", ""),
            ],
        )
        self.assertEqual(views.main_resource(rows[0][2]), (20, 100, 120))

    def test_lists_gpu_types_directly_below_a_single_kind(self):
        p = summary(node("a", cfg_gpus={"h100_1g": 8, "h100_2g": 4}, alloc_gpus={}))

        rows = list(views.sub_rows(p))

        self.assertEqual(
            [indent + label for indent, label, _, _ in rows],
            ["  ├ h100_1g", "  └ h100_2g"],
        )
        self.assertEqual(rows[1][2]["name"], "gpu › h100_2g")
        self.assertEqual(list(views.sub_rows(summary(node("a"), node("b")))), [])

    def test_lists_node_names_up_to_a_limit(self):
        self.assertEqual(views.node_names(["a", "b"]), "a, b")
        self.assertEqual(
            views.node_names(["a", "b", "c"], limit=2), "a, b, … (3 nodes)"
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
