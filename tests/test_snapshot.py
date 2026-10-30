import unittest
from unittest.mock import patch

from fakes import FakeHost

from sqtop import hosts, slurm, snapshot
from sqtop.snapshot import Limit, Scope

ASSOC_MGR = """Current Association Manager state

Association Records

ClusterName=c Account=root UserName= Partition= Priority=0 ID=1
    ParentAccount= Lineage=/ DefAssoc=No
    GrpJobs=N(5) GrpSubmitJobs=N(9)
    GrpTRES=cpu=N(64),gres/gpu=N(8)
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
ClusterName=c Account=lab UserName= Partition= Priority=0 ID=2
    ParentAccount=root(1) Lineage=/lab/ DefAssoc=No
    GrpJobs=N(4) GrpSubmitJobs=N(6)
    GrpTRES=cpu=N(48),gres/gpu=8(6)
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
ClusterName=c Account=lab UserName=me(123) Partition= Priority=0 ID=3
    SharesRaw/Norm/Level/Factor=1/0.03/30/0.08
    ParentAccount= Lineage=/lab/0-me/ DefAssoc=Yes
    GrpJobs=N(1) GrpSubmitJobs=N(2)
    GrpTRES=cpu=N(4),gres/gpu=N(1)
    MaxJobs=10(1) MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
    MaxTRESPJ=gres/gpu=4
ClusterName=c Account=guests UserName=me(123) Partition= Priority=0 ID=4
    ParentAccount= Lineage=/guests/0-me/ DefAssoc=No
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=

QOS Records

QOS=normal(1)
    MaxWallPJ=
    MaxTRESPJ=
    MaxTRESPN=
    Account Limits
      lab
        MaxJobsPA=N(4) MaxJobsAccruePA=N(0) MaxSubmitJobsPA=N(6)
        MaxTRESPA=cpu=N(48),gres/gpu=N(6)
    User Limits
      other(456)
        MaxJobsPU=N(2) MaxJobsAccruePU=N(0) MaxSubmitJobsPU=N(2)
        MaxTRESPU=cpu=N(8),gres/gpu=N(2)
QOS=limited(2)
    MaxWallPJ=180
    MaxTRESPJ=cpu=8,mem=65536,gres/gpu=1
    MaxTRESPN=
    Account Limits
      guests
        MaxJobsPA=2(1) MaxJobsAccruePA=N(0) MaxSubmitJobsPA=N(1)
        MaxTRESPA=gres/gpu=N(1)
    User Limits
      someone(789)
        MaxJobsPU=1(1) MaxJobsAccruePU=N(0) MaxSubmitJobsPU=N(1)
        MaxTRESPU=cpu=N(8),gres/gpu=1(1),gres/gpu:b200=1(1)
QOS=empty(3)
    MaxWallPJ=
    Account Limits
        No Accounts
    User Limits
        No Users
"""

QOS = slurm.parse_qos_records(ASSOC_MGR)
ASSOCIATIONS = slurm.parse_associations(ASSOC_MGR)


def node(name, state="MIXED", partitions=("gpu",), **overrides):
    return {
        "name": name,
        "state": state,
        "cpu_tot": 64,
        "cpu_efctv": 60,
        "cpu_alloc": 20,
        "mem_tot": 512 * 1024,
        "mem_alloc": 128 * 1024,
        "cfg_gpus": {"l40s": 4},
        "alloc_gpus": {"l40s": 1},
        "partitions": list(partitions),
        **overrides,
    }


def partition(name, **overrides):
    return {
        "name": name,
        "state": "UP",
        "default": False,
        "max_time": 180,
        "default_time": None,
        "allow_accounts": None,
        "deny_accounts": set(),
        "allow_qos": None,
        "deny_qos": set(),
        "allow_groups": None,
        "qos": None,
        "nodes": "",
        "total_nodes": 0,
        "total_cpus": 0,
        "max_nodes": None,
        "priority_tier": 1,
        "priority_job_factor": 1,
        **overrides,
    }


def by_name(limits):
    return {(limit.scope, limit.name): (limit.max, limit.used) for limit in limits}


class ScopeTests(unittest.TestCase):
    def test_leaves_out_what_slurm_would_pick_anyway(self):
        self.assertEqual(Scope("lab", "normal", True, True).directives(), [])
        self.assertEqual(Scope("lab", "high", True, False).directives(), ["--qos=high"])
        self.assertEqual(
            Scope("guests", "limited", False, True).directives(),
            ["--account=guests"],
        )
        self.assertEqual(Scope().directives(), [])

    def test_labels_the_pair(self):
        self.assertEqual(Scope("lab", "normal").label, "lab/normal")
        self.assertEqual(Scope().label, "default account/QOS")


class DefaultQosTests(unittest.TestCase):
    def test_follows_slurmctld(self):
        self.assertEqual(snapshot.default_qos(["high", "normal"], "high"), "high")
        self.assertEqual(snapshot.default_qos(["limited"]), "limited")
        self.assertEqual(snapshot.default_qos(["high", "normal"]), "normal")
        self.assertIsNone(snapshot.default_qos(["high", "low"]))


class DiscoverScopesTests(unittest.TestCase):
    def discover(self, assignments, accept=lambda pairs: set(pairs)):
        return snapshot.discover_scopes("me", assignments, ASSOC_MGR, QOS, accept)

    def test_uses_the_assignments_from_sacctmgr_without_probing(self):
        def accept(pairs):
            raise AssertionError("assigned pairs need no probe")

        scopes, notes = self.discover(
            {
                "lab": {"qos": ["high", "normal"], "default_qos": None},
                "guests": {"qos": ["limited"], "default_qos": None},
            },
            accept,
        )

        # With two accounts, commands always name the account.
        self.assertEqual(
            scopes,
            [
                Scope("guests", "limited", False, True, user_default=False),
                Scope("lab", "high", False, False, user_default=True),
                Scope("lab", "normal", False, True, user_default=True),
            ],
        )
        self.assertEqual(notes, [])

    def test_leaves_out_the_only_account(self):
        scopes, _ = self.discover({"lab": {"qos": ["normal"], "default_qos": None}})

        self.assertEqual(scopes, [Scope("lab", "normal", True, True, True)])
        self.assertEqual(scopes[0].directives(), [])

    def test_tests_each_cached_qos_for_each_account_when_sacctmgr_fails(self):
        probed = []

        def accept(pairs):
            probed.extend(pairs)
            return {("lab", "normal"), ("guests", "limited")}

        scopes, notes = self.discover(None, accept)

        self.assertEqual(
            sorted(probed),
            sorted(
                (account, qos)
                for account in ("guests", "lab")
                for qos in ("empty", "limited", "normal")
            ),
        )
        self.assertEqual(
            [scope.label for scope in scopes], ["guests/limited", "lab/normal"]
        )
        self.assertIn("test-submitting", notes[0])

    def test_notes_an_account_without_a_usable_qos(self):
        scopes, notes = self.discover({"lab": {"qos": [], "default_qos": None}})

        self.assertEqual(scopes, [])
        self.assertIn("No usable QOS found for account 'lab'.", notes)

    def test_does_not_fall_back_when_sacctmgr_reports_no_associations(self):
        def accept(pairs):
            raise AssertionError("an empty answer is authoritative")

        scopes, notes = self.discover({}, accept)

        self.assertEqual(scopes, [])
        self.assertIn("No Slurm association found for 'me'.", notes)

    def test_accepts_every_pair_slurm_does_not_call_invalid(self):
        replies = {
            "--qos=a": "sbatch: error: Invalid qos specification",
            "--qos=b": "Invalid account or account/partition combination specified",
            "--qos=c": "Job's QOS not permitted to use this partition",
            "--qos=d": "sbatch: Job 1 to start at 2030-01-01T00:00:00",
        }
        host = FakeHost(
            probe=lambda args: (1, next(v for k, v in replies.items() if k in args))
        )
        pairs = [("lab", qos) for qos in "abcd"]

        accepted = snapshot.accepted_pairs(host, pairs)

        self.assertEqual(accepted, {("lab", "c"), ("lab", "d")})
        self.assertEqual(len(host.streamed), 1)
        self.assertEqual(
            host.streamed[0][0],
            [
                "sbatch",
                "--test-only",
                "--account=lab",
                "--qos=a",
                "-t1:00:00",
                "--output=/dev/null",
                "--wrap=sleep infinity",
            ],
        )


class ScopeLimitTests(unittest.TestCase):
    def test_lists_qos_job_caps_and_borrows_caps_but_not_usage(self):
        jobs = [{"state": "RUNNING", "qos": "limited"}]
        limits = snapshot.scope_limits(
            Scope("guests", "limited"), QOS["limited"], ASSOCIATIONS, "me", jobs
        )

        rows = by_name(limits)
        self.assertEqual(rows["QOS per user", "running jobs"], (1, 1))
        self.assertEqual(rows["QOS per user", "submitted jobs"], (None, 1))
        # Borrowed from someone's entry; their usage of 1 is not ours.
        self.assertEqual(rows["QOS per user", "GPUs"], (1, "?"))
        self.assertEqual(rows["QOS per user", "b200 GPUs"], (1, "?"))
        self.assertEqual(rows["QOS per account", "running jobs"], (2, 1))
        self.assertEqual(rows["QOS per job", "CPUs"], (8, None))
        self.assertEqual(rows["QOS per job", "memory"], (65536, None))
        self.assertEqual(rows["QOS per job", "wall time"], (180, None))
        self.assertNotIn(("QOS per user", "CPUs"), rows)

    def test_counts_no_usage_when_nothing_runs_in_the_qos(self):
        limits = snapshot.scope_limits(
            Scope("guests", "limited"), QOS["limited"], ASSOCIATIONS, "me", []
        )

        rows = by_name(limits)
        self.assertEqual(rows["QOS per user", "GPUs"], (1, 0))
        self.assertEqual(rows["QOS per user", "running jobs"], (1, 0))

    def test_marks_job_caps_unknown_without_any_user_entry(self):
        limits = snapshot.scope_limits(
            Scope("lab", "empty"), QOS["empty"], ASSOCIATIONS, "me", None
        )

        self.assertEqual(by_name(limits)["QOS per user", "running jobs"], ("?", "?"))

    def test_adds_association_caps_for_the_user_and_its_accounts(self):
        limits = snapshot.scope_limits(
            Scope("lab", "normal"), QOS["normal"], ASSOCIATIONS, "me", []
        )

        rows = by_name(limits)
        self.assertEqual(rows["You in lab", "running jobs"], (10, 1))
        self.assertEqual(rows["You in lab, per job", "GPUs"], (4, None))
        self.assertEqual(rows["Account lab", "GPUs"], (8, 6))
        self.assertNotIn(("Account root", "GPUs"), rows)
        self.assertNotIn(("Account lab", "CPUs"), rows)

    def test_finds_the_tightest_cap(self):
        limits = [
            Limit("a", "GPUs", "tres:gres/gpu", 8),
            Limit("b", "GPUs", "tres:gres/gpu", 4),
            Limit("c", "GPUs", "tres:gres/gpu", None),
            Limit("d", "GPUs", "tres:gres/gpu", "?"),
        ]

        self.assertEqual(snapshot.tightest(limits, "tres:gres/gpu"), 4)
        self.assertIsNone(snapshot.tightest(limits, "wall"))

    def test_reads_the_callers_fairshare(self):
        self.assertEqual(snapshot.fairshare(ASSOCIATIONS, "lab", "me"), 0.08)
        self.assertIsNone(snapshot.fairshare(ASSOCIATIONS, "guests", "me"))


class ResourceUsageTests(unittest.TestCase):
    def test_separates_free_from_unavailable_capacity(self):
        usage = snapshot.resource_usage(
            [node("a"), node("b", state="IDLE+DRAIN", cpu_alloc=0, alloc_gpus={})]
        )

        self.assertEqual(
            dict(usage["cpus"]),
            {"total": 120, "allocated": 20, "free": 40, "unavailable": 60},
        )
        self.assertEqual(
            dict(usage["gpus"]["l40s"]),
            {"total": 8, "allocated": 1, "free": 3, "unavailable": 4},
        )
        self.assertEqual(dict(usage["states"]), {"mixed": 1, "unavailable": 1})
        self.assertEqual(usage["node_count"], 2)

    def test_uses_effective_cpus(self):
        usage = snapshot.resource_usage([node("a", cpu_efctv=48)])

        self.assertEqual(usage["cpus"]["total"], 48)

    def test_categorizes_sqtops(self):
        for state, category in [
            ("IDLE", "idle"),
            ("ALLOCATED", "allocated"),
            ("MIXED+PLANNED", "mixed"),
            ("COMPLETING", "mixed"),
            ("DOWN+NOT_RESPONDING", "unavailable"),
        ]:
            with self.subTest(state=state):
                self.assertEqual(snapshot.node_category({"state": state}), category)


class PartitionAccessTests(unittest.TestCase):
    def access(
        self, accounts=("lab",), qos=("normal",), groups=frozenset({"lab"}), **fields
    ):
        return snapshot.partition_access(
            partition("p", **fields), set(accounts), set(qos), groups
        )

    def test_allows_an_unrestricted_partition(self):
        self.assertEqual(self.access(), (True, ""))

    def test_checks_state_accounts_qos_and_groups(self):
        self.assertEqual(self.access(state="DOWN"), (False, "partition is DOWN"))
        self.assertFalse(self.access(allow_accounts={"other"})[0])
        self.assertFalse(self.access(deny_accounts={"lab"})[0])
        self.assertTrue(self.access(accounts=("lab", "x"), deny_accounts={"lab"})[0])
        self.assertFalse(self.access(allow_qos={"high"})[0])
        self.assertFalse(self.access(deny_qos={"normal"})[0])
        self.assertFalse(self.access(allow_groups={"admins"})[0])

    def test_skips_checks_it_cannot_make(self):
        self.assertTrue(self.access(accounts=(), allow_accounts={"other"})[0])
        self.assertTrue(self.access(qos=(), allow_qos={"high"})[0])
        self.assertTrue(self.access(groups=None, allow_groups={"admins"})[0])


class SummaryTests(unittest.TestCase):
    def test_summarizes_each_partition_from_its_member_nodes(self):
        nodes = [node("a"), node("b", partitions=("gpu", "long"))]
        summaries = snapshot.summarize_partitions(
            [partition("gpu"), partition("long", allow_accounts={"other"})],
            nodes,
            {"gpu": 7},
            [Scope("lab", "normal")],
            None,
        )

        gpu, long = summaries
        self.assertEqual(gpu["node_count"], 2)
        self.assertEqual(gpu["gpus"]["l40s"]["free"], 6)
        self.assertEqual(gpu["pending"], 7)
        self.assertTrue(gpu["usable"])
        self.assertEqual(long["node_count"], 1)
        self.assertEqual(long["pending"], 0)
        self.assertFalse(long["usable"])
        self.assertEqual(long["access_note"], "none of your accounts is allowed")

    def test_marks_pending_unknown_when_squeue_failed(self):
        summary = snapshot.summarize_partitions([partition("gpu")], [], None, [], None)

        self.assertIsNone(summary[0]["pending"])

    def test_groups_hardware_by_whole_gibibyte(self):
        nodes = [
            node("a", mem_tot=515472),
            node("b", mem_tot=515478),
            node("c", mem_tot=516096),
            node("d", cfg_gpus={}, alloc_gpus={}),
        ]

        groups = snapshot.summarize_hardware(nodes)

        self.assertEqual(
            [group["label"] for group in groups],
            [
                "60 CPUs / 512 GB",
                "60 CPUs / 503 GB / 4x l40s",
                "60 CPUs / 504 GB / 4x l40s",
            ],
        )
        self.assertEqual(groups[1]["names"], ["a", "b"])

    def test_takes_the_largest_gpu_count_per_node(self):
        nodes = [node("a"), node("b", cfg_gpus={"l40s": 2, "h100": 8})]

        self.assertEqual(snapshot.gpu_capacities(nodes), {"h100": 8, "l40s": 4})


NODE_TEXT = """NodeName=a Arch=x86_64
   CPUAlloc=20 CPUEfctv=60 CPUTot=64
   RealMemory=524288 AllocMem=131072
   State=MIXED
   Partitions=gpu
   CfgTRES=cpu=60,mem=512G,gres/gpu=4,gres/gpu:nvidia_l40s=4
   AllocTRES=cpu=20,gres/gpu=1,gres/gpu:nvidia_l40s=1
"""
PARENT_ASSOC = """Current Association Manager state

Association Records

ClusterName=c Account=dept UserName= Partition= Priority=0 ID=9
    ParentAccount=root(1) Lineage=/dept/ DefAssoc=No
    GrpJobs=N(9) GrpSubmitJobs=N(9)
    GrpTRES=gres/gpu=16(12)
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
"""


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.host = FakeHost(
            {
                "sacctmgr": "lab|normal|\n",
                "scontrol show node": NODE_TEXT,
                "scontrol show partition": "PartitionName=gpu State=UP MaxTime=03:00:00\n",
                "scontrol show assoc_mgr flags=assoc,qos": ASSOC_MGR,
                "squeue -h -u me": "",
                "squeue -h -t PD": "gpu\ngpu\n",
            }
        )
        self.info = snapshot.ClusterInfo("c", "25.05", "me", "login", None)

    def test_collects_every_panel_in_one_batch(self):
        snap = snapshot.Collector(self.info, self.host).collect()

        self.assertEqual(snap.scopes, [Scope("lab", "normal", True, True, True)])
        self.assertEqual(snap.partitions[0]["pending"], 2)
        self.assertEqual(snap.partitions[0]["gpus"]["nvidia_l40s"]["free"], 3)
        self.assertEqual(snap.gpu_names, {"nvidia_l40s": "l40s"})
        self.assertEqual(snap.hardware[0]["label"], "60 CPUs / 512 GB / 4x l40s")
        self.assertEqual(snap.jobs, [])
        self.assertEqual(snap.notes, [])
        # One sacctmgr call, then one batch with everything else.
        self.assertEqual(len(self.host.batches), 2)
        self.assertEqual(len(self.host.batches[1]), 5)

    def test_fetches_the_cache_only_for_the_callers_accounts_and_qos(self):
        collector = snapshot.Collector(self.info, self.host)
        collector.collect()
        collector.collect()

        commands = [
            args for batch in self.host.batches for args in batch if "assoc_mgr" in args
        ]
        self.assertTrue(all("accounts=lab" in args for args in commands))
        self.assertTrue(all("qos=normal" in args for args in commands))
        sacctmgr = [batch for batch in self.host.batches if batch[0][0] == "sacctmgr"]
        self.assertEqual(len(sacctmgr), 1)

    def test_adds_parent_account_limits(self):
        self.host.responses["scontrol show assoc_mgr flags=assoc,qos"] = (
            ASSOC_MGR.replace(
                "ParentAccount=root(1) Lineage=/lab/", "ParentAccount=dept(9)"
            )
        )
        self.host.responses["scontrol show assoc_mgr flags=assoc accounts=dept"] = (
            PARENT_ASSOC
        )
        collector = snapshot.Collector(self.info, self.host)

        snap = collector.collect()

        limits = by_name(snap.limits(snap.scopes[0], "me"))
        self.assertEqual(limits["Account dept", "GPUs"], (16, 12))
        self.assertEqual(collector.parents, {"dept"})
        # The QOS records still parse after the inserted associations.
        self.assertIn("normal", snap.qos_records)

    def test_fetches_the_whole_cache_when_sacctmgr_fails(self):
        self.host.responses["sacctmgr"] = None

        snap = snapshot.Collector(self.info, self.host).collect()

        self.assertIn(
            ["scontrol", "show", "assoc_mgr", "flags=assoc,qos"], self.host.batches[1]
        )
        self.assertEqual(len(self.host.streamed), 1)
        self.assertIn("test-submitting", snap.notes[0])

    def test_notes_what_failed_and_still_collects_the_rest(self):
        for prefix in (
            "scontrol show node",
            "squeue -h -u me",
            "scontrol show assoc_mgr flags=assoc,qos",
        ):
            self.host.responses[prefix] = None

        snap = snapshot.Collector(self.info, self.host).collect()

        self.assertEqual(snap.nodes, [])
        self.assertIsNone(snap.jobs)
        self.assertEqual(snap.qos_records, {})
        self.assertEqual(len(snap.notes), 3)

    def test_lets_a_lost_connection_through(self):
        with self.assertRaises(hosts.ConnectionLost):
            snapshot.Collector(self.info, FakeHost(lost="refused")).collect()


class DetectTests(unittest.TestCase):
    def test_reads_this_machine(self):
        host = FakeHost(
            {"scontrol show config": "ClusterName = c\nSLURM_VERSION = 25\n"}
        )
        with patch.object(slurm, "current_user", return_value="me"):
            info = snapshot.ClusterInfo.detect(host)

        self.assertEqual((info.name, info.version, info.user), ("c", "25", "me"))
        self.assertIsNone(info.ssh)

    def test_reads_a_remote_login_node_in_one_session(self):
        host = FakeHost(
            {
                "printenv PATH": "/opt/slurm/bin:/usr/bin\n",
                "printenv SLURM_CONF": "/cm/slurm/etc/fir/slurm.conf\n",
                "pwd": "/scratch/me\n",
                "id -un": "me\n",
                "id -Gn": "me lab\n",
                "hostname -s": "login2\n",
                "sh -c": "/bin/zsh\n",
                "scontrol show config": "ClusterName = fir\nSLURM_VERSION = 26.05\n",
            },
            ssh="fir.example.org",
        )

        info = snapshot.ClusterInfo.detect(host)

        self.assertEqual(len(host.batches), 1)
        self.assertEqual(
            host.env,
            {
                "PATH": "/opt/slurm/bin:/usr/bin",
                "SLURM_CONF": "/cm/slurm/etc/fir/slurm.conf",
            },
        )
        self.assertEqual(host.workdir, "/scratch/me")
        # Without SLURM_CONF in the login shell, only PATH is carried over.
        host.responses.pop("printenv SLURM_CONF")
        snapshot.ClusterInfo.detect(host)
        self.assertEqual(list(host.env), ["PATH"])
        self.assertEqual(
            (info.name, info.version, info.user, info.hostname, info.shell),
            ("fir", "26.05", "me", "login2", "zsh"),
        )
        self.assertEqual(info.groups, frozenset({"me", "lab"}))
        self.assertEqual(info.ssh, "fir.example.org")


if __name__ == "__main__":
    unittest.main()
