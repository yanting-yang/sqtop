import shlex
import unittest
from unittest.mock import patch

from sqtop import slurm

NODE_BLOCK = """NodeName=rcl-nv2.ece.ubc.ca Arch=x86_64 CoresPerSocket=56
   CPUAlloc=88 CPUEfctv=192 CPUTot=224 CPULoad=18.41
   RealMemory=2060000 AllocMem=706560 FreeMem=224124 Sockets=2 Boards=1
   CoreSpecCount=16 CPUSpecList=96-111,208-223 MemSpecLimit=65536
   State=MIXED ThreadsPerCore=2 TmpDisk=0 Weight=1
   CfgTRES=cpu=192,mem=2060000M,billing=448,gres/gpu=16,gres/gpu:nvidia_b200=4
   AllocTRES=cpu=88,mem=690G,gres/gpu=8,gres/gpu:nvidia_b200=3
"""


class ParseTresGpusTests(unittest.TestCase):
    def test_parses_typed_and_untyped_entries(self):
        tres = "cpu=192,mem=2060000M,gres/gpu=16,gres/gpu:nvidia_b200=4"

        self.assertEqual(slurm.parse_tres_gpus(tres), {"gpu": 16, "nvidia_b200": 4})

    def test_returns_empty_mapping_without_gpus(self):
        self.assertEqual(slurm.parse_tres_gpus("cpu=48,mem=257000M"), {})


class GpuRollupTests(unittest.TestCase):
    def test_drops_the_rollup_when_per_model_counts_exist(self):
        gpus = {
            "gpu": 16,
            "nvidia_b200": 4,
            "nvidia_b200_2g.45gb": 8,
            "nvidia_b200_3g.90gb": 4,
        }

        self.assertEqual(
            slurm.remove_untyped_gpu_rollup(gpus),
            {"nvidia_b200": 4, "nvidia_b200_2g.45gb": 8, "nvidia_b200_3g.90gb": 4},
        )

    def test_keeps_the_rollup_when_it_is_the_only_information(self):
        self.assertEqual(slurm.remove_untyped_gpu_rollup({"gpu": 4}), {"gpu": 4})

    def test_leaves_cpu_only_nodes_empty(self):
        self.assertEqual(slurm.remove_untyped_gpu_rollup({}), {})

    def test_normalizes_both_configured_and_allocated_gpus(self):
        node = slurm.normalize_gpu_rollups(
            {
                "cfg_gpus": {"gpu": 16, "nvidia_b200": 4},
                "alloc_gpus": {"gpu": 8, "nvidia_b200": 3},
            }
        )

        self.assertEqual(node["cfg_gpus"], {"nvidia_b200": 4})
        self.assertEqual(node["alloc_gpus"], {"nvidia_b200": 3})


class GresModelTests(unittest.TestCase):
    """Where accounting does not track GPU models, as on Trillium GPU,
    `CfgTRES` has only the total and `Gres=` names the model."""

    BLOCK = """NodeName=trig0002 Arch=x86_64 CoresPerSocket=48
   CPUAlloc=80 CPUEfctv=96 CPUTot=96 CPULoad=60.20
   AvailableFeatures=(null)
   Gres=gpu:h100:4(S:2-3)
   RealMemory=770000 AllocMem=768000 FreeMem=10000 Sockets=2 Boards=1
   State=MIXED ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=debug,compute,debug_full_node,compute_full_node
   CfgTRES=cpu=96,mem=770000M,billing=4,gres/gpu=4
   AllocTRES=cpu=80,mem=750G,gres/gpu=4
"""

    def test_reads_gpu_models_from_gres(self):
        self.assertEqual(slurm.parse_gres_gpus("gpu:h100:4(S:2-3)"), {"h100": 4})
        self.assertEqual(
            slurm.parse_gres_gpus("gpu:a100:2(S:0),gpu:v100:2(S:1),shard:8"),
            {"a100": 2, "v100": 2},
        )
        self.assertEqual(slurm.parse_gres_gpus("gpu:4(S:0-1)"), {})
        self.assertEqual(slurm.parse_gres_gpus("(null)"), {})

    def test_names_the_untyped_total_after_the_node_model(self):
        node = slurm.normalize_gpu_rollups(slurm.parse_nodes(self.BLOCK)[0])

        self.assertEqual(node["cfg_gpus"], {"h100": 4})
        self.assertEqual(node["alloc_gpus"], {"h100": 4})

    def test_keeps_the_total_untyped_with_several_models(self):
        block = self.BLOCK.replace("gpu:h100:4(S:2-3)", "gpu:a100:2,gpu:v100:2")
        node = slurm.normalize_gpu_rollups(slurm.parse_nodes(block)[0])

        self.assertEqual(node["cfg_gpus"], {"gpu": 4})
        self.assertEqual(node["alloc_gpus"], {"gpu": 4})

    def test_prefers_models_in_cfg_tres(self):
        node = slurm.normalize_gpu_rollups(slurm.parse_nodes(NODE_BLOCK)[0])

        self.assertEqual(node["cfg_gpus"], {"nvidia_b200": 4})


class ParseNodesTests(unittest.TestCase):
    def test_parses_core_fields(self):
        node = slurm.parse_nodes(NODE_BLOCK)[0]

        self.assertEqual(node["name"], "rcl-nv2.ece.ubc.ca")
        self.assertEqual(node["state"], "MIXED")
        self.assertEqual(node["cpu_tot"], 224)
        self.assertEqual(node["cpu_alloc"], 88)
        self.assertEqual(node["mem_tot"], 2060000)
        self.assertEqual(node["mem_alloc"], 706560)
        self.assertEqual(node["cfg_gpus"], {"gpu": 16, "nvidia_b200": 4})
        self.assertEqual(node["alloc_gpus"], {"gpu": 8, "nvidia_b200": 3})

    def test_records_effective_cpus_separately_from_total(self):
        node = slurm.parse_nodes(NODE_BLOCK)[0]

        self.assertEqual(node["cpu_efctv"], 192)

    def test_falls_back_to_cpu_tot_when_effective_count_is_absent(self):
        block = NODE_BLOCK.replace("CPUEfctv=192 ", "")

        self.assertEqual(slurm.parse_nodes(block)[0]["cpu_efctv"], 224)


ASSOC_MGR = """Current Association Manager state

Association Records

ClusterName=rcl Account=root UserName= Partition= Priority=0 ID=1
    ParentAccount= Lineage=/ DefAssoc=No
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
ClusterName=rcl Account=staff UserName=jordan.smith(12345) Partition= Priority=0 ID=9
    ParentAccount= Lineage=/staff/0-jordan.smith/ DefAssoc=No
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
ClusterName=rcl Account=guests UserName=jordan.smith(12345) Partition= Priority=0 ID=32
    ParentAccount= Lineage=/guests/0-jordan.smith/ DefAssoc=Yes
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=

QOS Records

QOS=normal(1)
    UsageRaw=311045818.638041
    MaxWallPJ=
    MaxTRESPJ=
    PreemptMode=OFF
    Account Limits
      rcl
        MaxJobsPA=N(8) MaxJobsAccruePA=N(0) MaxSubmitJobsPA=N(8)
    User Limits
      mahdik(18821)
        MaxJobsPU=16(5) MaxJobsAccruePU=4(0) MaxSubmitJobsPU=100(5)
        MaxTRESPU=cpu=64(56),mem=524288(290816),gres/gpu=4(4)
QOS=limited(2)
    UsageRaw=55306116.454286
    MaxWallPJ=
    MaxTRESPJ=cpu=8,mem=65536,gres/gpu=1,gres/gpu:nvidia_b200=1,gres/gpu:nvidia_b200_3g.90gb=1
    PreemptMode=OFF
    Account Limits
      guests
        MaxJobsPA=N(1) MaxJobsAccruePA=N(0) MaxSubmitJobsPA=N(1)
    User Limits
      yuxiang.fu(24958)
        MaxJobsPU=1(1) MaxJobsAccruePU=N(0) MaxSubmitJobsPU=N(1)
        MaxTRESPU=cpu=N(8),mem=N(65536),gres/gpu=1(1)
"""


class ParseTresValuesTests(unittest.TestCase):
    def test_keeps_tres_names_containing_colons_intact(self):
        values = slurm.parse_tres_values("cpu=8,mem=65536,gres/gpu:nvidia_b200=1")

        self.assertEqual(values, {"cpu": 8, "mem": 65536, "gres/gpu:nvidia_b200": 1})

    def test_ignores_empty_and_non_numeric_entries(self):
        self.assertEqual(slurm.parse_tres_values(""), {})
        self.assertEqual(slurm.parse_tres_values("cpu=unlimited,mem=64"), {"mem": 64})


class ParseTresLimitsTests(unittest.TestCase):
    def test_separates_resource_limits_from_current_usage(self):
        limits, usage = slurm.parse_tres_limits(
            "cpu=64(56),mem=524288(290816),gres/gpu=4(4)"
        )

        self.assertEqual(limits, {"cpu": 64, "mem": 524288, "gres/gpu": 4})
        self.assertEqual(usage, {"cpu": 56, "mem": 290816, "gres/gpu": 4})

    def test_keeps_unlimited_resources_and_zero_caps_distinct(self):
        limits, usage = slurm.parse_tres_limits("cpu=N(8),gres/gpu=0(0)")

        self.assertEqual(limits, {"cpu": None, "gres/gpu": 0})
        self.assertEqual(usage, {"cpu": 8, "gres/gpu": 0})

    def test_preserves_model_gpu_names_alongside_the_total_gpu_limit(self):
        limits, usage = slurm.parse_tres_limits(
            "gres/gpu=4(2),gres/gpu:nvidia_b200=1(0),gres/gpu:nvidia_b200_2g.45gb=3(2)"
        )

        self.assertEqual(
            limits,
            {
                "gres/gpu": 4,
                "gres/gpu:nvidia_b200": 1,
                "gres/gpu:nvidia_b200_2g.45gb": 3,
            },
        )
        self.assertEqual(
            usage,
            {
                "gres/gpu": 2,
                "gres/gpu:nvidia_b200": 0,
                "gres/gpu:nvidia_b200_2g.45gb": 2,
            },
        )

    def test_ignores_empty_and_malformed_entries_without_losing_valid_ones(self):
        self.assertEqual(slurm.parse_tres_limits(""), ({}, {}))
        limits, usage = slurm.parse_tres_limits(
            "garbage,cpu=bad(3),mem=64,gres/gpu=4(bad),node=3(1)trailing,,cpu=64(0)"
        )

        self.assertEqual(limits, {"cpu": 64})
        self.assertEqual(usage, {"cpu": 0})


class ParseLimitTests(unittest.TestCase):
    def test_reads_a_numeric_limit(self):
        self.assertEqual(slurm.parse_limit("16(5)"), 16)

    def test_reads_an_absent_limit_as_none(self):
        self.assertIsNone(slurm.parse_limit("N(0)"))

    def test_returns_none_for_unparseable_text(self):
        self.assertIsNone(slurm.parse_limit("junk"))


class ParseDefaultAccountTests(unittest.TestCase):
    def test_prefers_the_default_association(self):
        self.assertEqual(
            slurm.parse_default_account(ASSOC_MGR, "jordan.smith"), "guests"
        )

    def test_returns_none_for_an_unknown_user(self):
        self.assertIsNone(slurm.parse_default_account(ASSOC_MGR, "nobody"))


class ParseQosRecordsTests(unittest.TestCase):
    def setUp(self):
        self.records = slurm.parse_qos_records(ASSOC_MGR)

    def test_finds_every_qos(self):
        self.assertEqual(sorted(self.records), ["limited", "normal"])

    def test_reads_per_job_tres_caps(self):
        self.assertEqual(
            self.records["limited"]["max_tres_pj"],
            {
                "cpu": 8,
                "mem": 65536,
                "gres/gpu": 1,
                "gres/gpu:nvidia_b200": 1,
                "gres/gpu:nvidia_b200_3g.90gb": 1,
            },
        )

    def test_leaves_per_job_caps_empty_when_unset(self):
        self.assertEqual(self.records["normal"]["max_tres_pj"], {})

    def test_records_the_accounts_each_qos_governs(self):
        self.assertEqual(self.records["limited"]["accounts"], {"guests"})
        self.assertEqual(self.records["normal"]["accounts"], {"rcl"})

    def test_reads_per_user_job_and_resource_limits(self):
        self.assertEqual(
            self.records["limited"]["user_limits"]["yuxiang.fu"],
            {
                "max_jobs": 1,
                "max_submit_jobs": None,
                "max_tres_pu": {"cpu": None, "mem": None, "gres/gpu": 1},
            },
        )
        self.assertEqual(
            self.records["normal"]["user_limits"]["mahdik"],
            {
                "max_jobs": 16,
                "max_submit_jobs": 100,
                "max_tres_pu": {"cpu": 64, "mem": 524288, "gres/gpu": 4},
            },
        )

    def test_records_resource_usage_separately_from_borrowable_limits(self):
        self.assertEqual(
            self.records["normal"]["user_tres_usage"],
            {"mahdik": {"cpu": 56, "mem": 290816, "gres/gpu": 4}},
        )
        self.assertEqual(
            self.records["limited"]["user_tres_usage"],
            {"yuxiang.fu": {"cpu": 8, "mem": 65536, "gres/gpu": 1}},
        )

    def test_keeps_each_users_resource_usage_separate(self):
        output = ASSOC_MGR.replace(
            "QOS=limited(2)",
            "      me(12345)\n"
            "        MaxJobsPU=16(1) MaxSubmitJobsPU=100(1)\n"
            "        MaxTRESPU=cpu=64(8),mem=524288(65536),gres/gpu=4(0)\n"
            "QOS=limited(2)",
        )
        record = slurm.parse_qos_records(output)["normal"]

        self.assertEqual(record["user_limits"]["me"], record["user_limits"]["mahdik"])
        self.assertEqual(
            record["user_tres_usage"],
            {
                "mahdik": {"cpu": 56, "mem": 290816, "gres/gpu": 4},
                "me": {"cpu": 8, "mem": 65536, "gres/gpu": 0},
            },
        )

    def test_ignores_association_records(self):
        # Association blocks also contain MaxJobs=/MaxSubmitJobs= fields.
        self.assertNotIn("jordan.smith", self.records["limited"]["user_limits"])


class QosLookupTests(unittest.TestCase):
    def setUp(self):
        self.records = slurm.parse_qos_records(ASSOC_MGR)

    def test_prefers_the_users_own_limits(self):
        record = self.records["limited"]
        record["user_limits"]["me"] = {"max_jobs": 3, "max_submit_jobs": 9}

        self.assertEqual(
            slurm.qos_user_limits(record, "me"), {"max_jobs": 3, "max_submit_jobs": 9}
        )

    def test_borrows_limits_when_the_user_has_no_entry_yet(self):
        # QOS per-user limits are one value applied to every user, so another
        # user's entry carries the same numbers.
        self.assertEqual(
            slurm.qos_user_limits(self.records["limited"], "jordan.smith"),
            {
                "max_jobs": 1,
                "max_submit_jobs": None,
                "max_tres_pu": {"cpu": None, "mem": None, "gres/gpu": 1},
            },
        )

    def test_borrowing_resource_limits_does_not_create_usage_for_the_user(self):
        record = self.records["normal"]

        limits = slurm.qos_user_limits(record, "jordan.smith")

        self.assertEqual(
            limits["max_tres_pu"], {"cpu": 64, "mem": 524288, "gres/gpu": 4}
        )
        self.assertNotIn("jordan.smith", record["user_tres_usage"])
        self.assertEqual(
            record["user_tres_usage"]["mahdik"],
            {"cpu": 56, "mem": 290816, "gres/gpu": 4},
        )

    def test_returns_empty_limits_when_no_user_entries_exist(self):
        self.assertEqual(slurm.qos_user_limits({"user_limits": {}}, "me"), {})


class ParseUserAccountsTests(unittest.TestCase):
    def test_returns_every_current_user_account(self):
        self.assertEqual(
            slurm.parse_user_accounts(ASSOC_MGR, "jordan.smith"), ["guests", "staff"]
        )

    def test_matches_exact_user_and_deduplicates_partition_associations(self):
        output = """Association Records
ClusterName=killarney Account=project UserName=me(123) Partition= ID=1
ClusterName=killarney Account=project UserName=me(123) Partition=gpu ID=2
ClusterName=killarney Account=other UserName=someone(124) Partition= ID=3
ClusterName=killarney Account=prefix UserName=me.extra(125) Partition= ID=4
ClusterName=killarney Account=parent UserName= Partition= ID=5
QOS Records
ClusterName=killarney Account=outside UserName=me(123) Partition= ID=6
"""

        self.assertEqual(slurm.parse_user_accounts(output, "me"), ["project"])
        self.assertEqual(slurm.parse_user_accounts(output, "missing"), [])
        self.assertEqual(slurm.parse_user_accounts("", "me"), [])


class ParseAccountQosTests(unittest.TestCase):
    def test_combines_duplicate_associations_and_sorts_qos_names(self):
        output = (
            "project|normal,interac|\n"
            "project|normal,extra|\n"
            "other|normal|\n"
            "project|normal|\n"
        )

        self.assertEqual(
            slurm.parse_account_qos(output),
            {
                "other": {"qos": ["normal"], "default_qos": None},
                "project": {"qos": ["extra", "interac", "normal"], "default_qos": None},
            },
        )

    def test_reads_the_default_qos_column(self):
        self.assertEqual(
            slurm.parse_account_qos("project|normal,high|high\n"),
            {"project": {"qos": ["high", "normal"], "default_qos": "high"}},
        )

    def test_preserves_accounts_with_no_qos_and_ignores_invalid_rows(self):
        self.assertEqual(
            slurm.parse_account_qos(
                "empty||\n spaced | normal, interac |\n|normal|\ninvalid\n"
            ),
            {
                "empty": {"qos": [], "default_qos": None},
                "spaced": {"qos": ["interac", "normal"], "default_qos": None},
            },
        )
        self.assertEqual(slurm.parse_account_qos(""), {})


MULTI_ACCOUNT_QOS = """QOS Records
QOS=normal(1)
    MaxWallPJ=
    MaxTRESPJ=cpu=64
    MaxTRESPN=gres/gpu=4,gres/gpu:h100=2
    Account Limits
      project-a
        MaxJobsPA=16(3) MaxJobsAccruePA=N(0) MaxSubmitJobsPA=N(6)
        MaxTRESPA=cpu=128(24),mem=N(65536),gres/gpu=0(0),gres/gpu:h100=2(1)
      project-b
        MaxJobsPA=16(1) MaxSubmitJobsPA=N(2)
        MaxTRESPA=cpu=128(8),mem=N(16384),gres/gpu=0(0),gres/gpu:h100=2(0)
    User Limits
      other(456)
        MaxJobsPU=4(1) MaxSubmitJobsPU=8(2)
        MaxTRESPU=cpu=64(8),gres/gpu=2(1)
QOS=interac(2)
    MaxWallPJ=180
    MaxTRESPJ=gres/gpu=1
    Account Limits
      project-a
        MaxJobsPA=0(0) MaxSubmitJobsPA=bad(9)
        MaxTRESPA=cpu=bad(4),mem=0(0)
      project-c
    User Limits
"""


class QosAccountLimitTests(unittest.TestCase):
    def setUp(self):
        self.records = slurm.parse_qos_records(MULTI_ACCOUNT_QOS)

    def test_parses_per_account_caps_separately_from_account_usage(self):
        record = self.records["normal"]
        self.assertEqual(
            record["account_limits"]["project-a"],
            {
                "max_jobs": 16,
                "max_submit_jobs": None,
                "max_tres_pa": {
                    "cpu": 128,
                    "mem": None,
                    "gres/gpu": 0,
                    "gres/gpu:h100": 2,
                },
            },
        )
        self.assertEqual(record["accounts"], {"project-a", "project-b"})
        self.assertEqual(
            record["account_job_usage"],
            {
                "project-a": {"running": 3, "total": 6},
                "project-b": {"running": 1, "total": 2},
            },
        )
        self.assertEqual(
            record["account_tres_usage"],
            {
                "project-a": {
                    "cpu": 24,
                    "mem": 65536,
                    "gres/gpu": 0,
                    "gres/gpu:h100": 1,
                },
                "project-b": {
                    "cpu": 8,
                    "mem": 16384,
                    "gres/gpu": 0,
                    "gres/gpu:h100": 0,
                },
            },
        )
        self.assertEqual(
            record["user_tres_usage"], {"other": {"cpu": 8, "gres/gpu": 1}}
        )

    def test_keeps_qos_records_separate_and_preserves_zero_and_missing_caps(self):
        record = self.records["interac"]
        self.assertEqual(
            record["account_limits"],
            {"project-a": {"max_jobs": 0, "max_tres_pa": {"mem": 0}}, "project-c": {}},
        )
        self.assertEqual(record["account_job_usage"], {"project-a": {"running": 0}})
        self.assertEqual(record["account_tres_usage"], {"project-a": {"mem": 0}})

    def test_reads_per_node_caps_and_per_job_time_in_minutes(self):
        self.assertEqual(
            self.records["normal"]["max_tres_pn"], {"gres/gpu": 4, "gres/gpu:h100": 2}
        )
        self.assertIsNone(self.records["normal"]["max_wall_pj"])
        self.assertEqual(self.records["interac"]["max_tres_pn"], {})
        self.assertEqual(self.records["interac"]["max_wall_pj"], 180)
        self.assertEqual(
            slurm.parse_qos_records(
                MULTI_ACCOUNT_QOS.replace("MaxWallPJ=180", "MaxWallPJ=0")
            )["interac"]["max_wall_pj"],
            0,
        )

    def test_distinguishes_missing_or_malformed_time_from_unset_time(self):
        output = """QOS Records
QOS=missing(1)
    MaxTRESPJ=
QOS=malformed(2)
    MaxWallPJ=invalid
QOS=unset(3)
    MaxWallPJ=N
"""
        records = slurm.parse_qos_records(output)

        self.assertNotIn("max_wall_pj", records["missing"])
        self.assertNotIn("max_wall_pj", records["malformed"])
        self.assertIsNone(records["unset"]["max_wall_pj"])

    def test_prefers_requested_accounts_limits(self):
        record = self.records["normal"]
        record["account_limits"]["project-b"]["max_jobs"] = 7

        self.assertEqual(slurm.qos_account_limits(record, "project-b")["max_jobs"], 7)

    def test_borrows_only_caps_when_the_account_has_no_cached_usage(self):
        record = self.records["normal"]
        self.assertEqual(
            slurm.qos_account_limits(record, "inactive"),
            record["account_limits"]["project-a"],
        )
        self.assertNotIn("inactive", record["account_job_usage"])
        self.assertNotIn("inactive", record["account_tres_usage"])
        self.assertEqual(
            record["account_job_usage"]["project-a"], {"running": 3, "total": 6}
        )

    def test_returns_unknown_caps_if_no_account_entry_has_limits(self):
        self.assertEqual(
            slurm.qos_account_limits({"account_limits": {}}, "inactive"), {}
        )
        self.assertEqual(
            slurm.qos_account_limits({"account_limits": {"other": {}}}, "inactive"), {}
        )


class CommandTests(unittest.TestCase):
    def test_lists_the_users_accounts_qos_and_default_qos(self):
        self.assertEqual(
            slurm.account_qos_command("me", "killarney"),
            [
                "sacctmgr",
                "-nP",
                "show",
                "assoc",
                "where",
                "user=me",
                "cluster=killarney",
                "format=Account,QOS%1000,DefaultQOS%100",
            ],
        )

    def test_filters_the_association_cache_to_the_callers_accounts_and_qos(self):
        self.assertEqual(
            slurm.assoc_mgr_command(),
            ["scontrol", "show", "assoc_mgr", "flags=assoc,qos"],
        )
        self.assertEqual(
            slurm.assoc_mgr_command({"lab-b", "lab-a"}, {"normal", "high"}),
            [
                "scontrol",
                "show",
                "assoc_mgr",
                "flags=assoc,qos",
                "accounts=lab-a,lab-b",
                "qos=high,normal",
            ],
        )

    def test_lists_the_users_jobs_with_the_field_format(self):
        command = slurm.jobs_command("me")

        self.assertEqual(command[:5], ["squeue", "-h", "-u", "me", "-o"])
        self.assertTrue(command[5].endswith("|%j"))


class TestProbeTests(unittest.TestCase):
    def test_wraps_sleep_infinity_for_sbatch_instead_of_a_batch_script(self):
        self.assertEqual(
            slurm.test_command("sbatch", ["--test-only", "--mem=32G"]),
            [
                "sbatch",
                "--test-only",
                "--mem=32G",
                "--output=/dev/null",
                "--wrap=sleep infinity",
            ],
        )

    def test_runs_the_login_shell_after_srun_directives(self):
        # Tamia's submit filter only routes an srun that names a program.
        self.assertEqual(
            slurm.test_command("srun", ["--test-only", "-t3:00:00"], "zsh"),
            ["srun", "--test-only", "-t3:00:00", "zsh"],
        )
        with patch.dict(slurm.os.environ, {"SHELL": "/bin/bash"}):
            self.assertEqual(slurm.test_command("srun", [])[-1], "bash")

    def test_extracts_estimated_start_time_and_partition(self):
        result = slurm.parse_test_output(
            "sbatch: Job 5484 to start at 2026-07-25T19:06:59 a using "
            "16 processors on nodes rack15-12 in partition mig",
            0,
            "sbatch",
        )

        self.assertEqual(result["start_time"], "2026-07-25T19:06:59")
        self.assertEqual(result["partition"], "mig")
        self.assertEqual(result["result"], "Runnable")

    def test_reports_a_rejected_request(self):
        result = slurm.parse_test_output(
            "allocation failure: Requested node configuration is not available",
            1,
            "sbatch",
        )

        self.assertIsNone(result["start_time"])
        self.assertIsNone(result["partition"])
        self.assertEqual(
            result["result"],
            "allocation failure: Requested node configuration is not available",
        )

    def test_reports_a_silent_failure_by_exit_status(self):
        self.assertEqual(
            slurm.parse_test_output("", 1, "srun")["result"],
            "srun exited with status 1",
        )


class FormatRunCommandTests(unittest.TestCase):
    def test_quotes_shell_metacharacters_and_preserves_original_probe(self):
        directives = [
            "--test-only",
            "--comment=two words; echo $HOME",
            "--time=3:00:00",
        ]
        original = directives.copy()

        command = slurm.format_run_command("sbatch", directives)

        self.assertEqual(
            shlex.split(command),
            [
                "sbatch",
                "--test-only",
                "--comment=two words; echo $HOME",
                "--time=3:00:00",
                "--output=/dev/null",
                "--wrap=sleep infinity",
            ],
        )
        self.assertIn("'--comment=two words; echo $HOME'", command)
        self.assertEqual(directives, original)

    def test_wraps_sleep_infinity_for_sbatch_instead_of_a_script(self):
        command = slurm.format_run_command(
            "sbatch", ["--gpus-per-node=h100:2", "--time=3:00:00"]
        )

        self.assertEqual(
            command,
            "sbatch --gpus-per-node=h100:2 --time=3:00:00 --output=/dev/null "
            '--wrap="sleep infinity"',
        )
        self.assertNotIn("job.sh", command)

    def test_opens_an_interactive_shell_for_srun(self):
        command = slurm.format_run_command("srun", ["--time=3:00:00"], shell="zsh")

        self.assertEqual(command, "srun --time=3:00:00 --pty zsh")

    def test_opens_the_callers_login_shell_by_default(self):
        with patch.dict(slurm.os.environ, {"SHELL": "/usr/bin/fish"}):
            command = slurm.format_run_command("srun", [])

        self.assertEqual(command, "srun --pty fish")

    def test_submits_the_same_request_without_a_script(self):
        self.assertEqual(
            slurm.run_command("sbatch", ["-t3:00:00"]),
            ["sbatch", "-t3:00:00", "--output=/dev/null", "--wrap=sleep infinity"],
        )
        self.assertEqual(
            slurm.run_command("srun", ["-t3:00:00"], "zsh"),
            ["srun", "-t3:00:00", "--pty", "zsh"],
        )
        self.assertEqual(
            slurm.parse_submitted("Submitted batch job 6330123\n"), "6330123"
        )
        self.assertIsNone(slurm.parse_submitted("sbatch: error: no"))

    def test_shows_each_probe_as_it_runs(self):
        directives = ["--test-only", "-t3:00:00"]

        self.assertEqual(
            slurm.format_test_command("sbatch", directives),
            'sbatch --test-only -t3:00:00 --output=/dev/null --wrap="sleep infinity"',
        )
        self.assertEqual(
            slurm.format_test_command("srun", directives, "zsh"),
            "srun --test-only -t3:00:00 zsh",
        )


class NodeAvailabilityTests(unittest.TestCase):
    def test_reads_partitions_from_the_node_record(self):
        node = slurm.parse_nodes(NODE_BLOCK + "   Partitions=mig,full \n")[0]

        self.assertEqual(node["partitions"], ["mig", "full"])

    def test_defaults_to_no_partitions(self):
        self.assertEqual(slurm.parse_nodes(NODE_BLOCK)[0]["partitions"], [])

    def test_judges_every_state_word(self):
        for state, available in [
            ("IDLE", True),
            ("MIXED+PLANNED", True),
            ("ALLOCATED", True),
            ("IDLE+DRAIN", False),
            ("MIXED+DRAIN", False),
            ("DOWN+NOT_RESPONDING", False),
            ("IDLE+CLOUD+POWERED_DOWN", False),
            ("RESERVED", False),
        ]:
            with self.subTest(state=state):
                self.assertEqual(slurm.node_available({"state": state}), available)


class DurationTests(unittest.TestCase):
    def test_parses_every_slurm_time_format_in_minutes(self):
        for text, minutes in [
            ("30", 30),
            ("30:00", 30),
            ("03:00:00", 180),
            ("1-00:00:00", 1440),
            ("2-12", 3600),
            ("1-01:30", 1530),
            ("0:01:30", 2),
        ]:
            with self.subTest(text=text):
                self.assertEqual(slurm.parse_duration(text), minutes)

    def test_reads_unlimited_unset_and_garbage_as_none(self):
        for text in ("UNLIMITED", "INFINITE", "NONE", "", "1:2:3:4", "x-01:00"):
            with self.subTest(text=text):
                self.assertIsNone(slurm.parse_duration(text))

    def test_formats_minutes_for_sbatch(self):
        self.assertEqual(slurm.format_duration(180), "0-03:00:00")
        self.assertEqual(slurm.format_duration(1530), "1-01:30:00")
        self.assertEqual(slurm.format_duration(92160), "64-00:00:00")

    def test_keeps_seconds_when_normalizing_squeue_times(self):
        for text, normalized in [
            ("0:00", "0-00:00:00"),
            ("5:23", "0-00:05:23"),
            ("1:02:03", "0-01:02:03"),
            ("1-00:00:00", "1-00:00:00"),
            ("2-12", "2-12:00:00"),
            ("UNLIMITED", "UNLIMITED"),
            ("INVALID", "INVALID"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(slurm.normalize_time(text), normalized)

        self.assertEqual(slurm.parse_seconds("1-01:02:03"), 90123)


class ParseConfigTests(unittest.TestCase):
    def test_reads_key_value_lines(self):
        output = (
            "Configuration data as of 2026-10-07T13:00:00\n"
            "ClusterName             = killarney\n"
            "PriorityWeightTRES      = cpu=15000,gres/gpu=15000\n"
            "SLURM_VERSION           = 25.05.9\n"
            "Slurmctld(primary) at kctl01 is UP\n"
        )

        config = slurm.parse_config(output)

        self.assertEqual(config["ClusterName"], "killarney")
        self.assertEqual(config["SLURM_VERSION"], "25.05.9")
        self.assertEqual(config["PriorityWeightTRES"], "cpu=15000,gres/gpu=15000")
        self.assertEqual(len(config), 3)


PARTITIONS = (
    "PartitionName=gpubase_l40s_b1 AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL "
    "AllocNodes=ALL Default=NO QoS=N/A DefaultTime=NONE MaxNodes=UNLIMITED "
    "MaxTime=03:00:00 Nodes=kn[001-168] PriorityJobFactor=12 PriorityTier=1 "
    "State=UP TotalCPUs=10752 TotalNodes=168 "
    "TRES=cpu=10752,mem=86520000M,node=168,gres/gpu=672 "
    "TRESBillingWeights=cpu=1016.67,mem=48.41G\n"
    "PartitionName=private AllowGroups=lab,admins AllowAccounts=lab-a,lab-b "
    "DenyQos=low AllowQos=ALL Default=YES QoS=interac DefaultTime=01:00:00 "
    "MaxTime=UNLIMITED Nodes=rack1 State=DRAIN TotalCPUs=64 TotalNodes=1\n"
)


class ParsePartitionsTests(unittest.TestCase):
    def test_reads_limits_and_sizes(self):
        partition = slurm.parse_partitions(PARTITIONS)[0]

        self.assertEqual(partition["name"], "gpubase_l40s_b1")
        self.assertEqual(partition["state"], "UP")
        self.assertFalse(partition["default"])
        self.assertEqual(partition["max_time"], 180)
        self.assertIsNone(partition["default_time"])
        self.assertIsNone(partition["qos"])
        self.assertEqual(partition["total_nodes"], 168)
        self.assertEqual(partition["total_cpus"], 10752)
        self.assertIsNone(partition["max_nodes"])
        self.assertEqual(partition["priority_tier"], 1)
        self.assertEqual(partition["preempt_mode"], "OFF")

    def test_reads_all_as_unrestricted_and_lists_as_sets(self):
        public, private = slurm.parse_partitions(PARTITIONS)

        self.assertIsNone(public["allow_accounts"])
        self.assertIsNone(public["allow_groups"])
        self.assertEqual(public["deny_qos"], set())
        self.assertEqual(private["allow_accounts"], {"lab-a", "lab-b"})
        self.assertEqual(private["allow_groups"], {"lab", "admins"})
        self.assertEqual(private["deny_qos"], {"low"})
        self.assertIsNone(private["allow_qos"])

    def test_reads_default_partition_qos_and_unlimited_time(self):
        private = slurm.parse_partitions(PARTITIONS)[1]

        self.assertTrue(private["default"])
        self.assertEqual(private["qos"], "interac")
        self.assertIsNone(private["max_time"])
        self.assertEqual(private["default_time"], 60)
        self.assertEqual(private["state"], "DRAIN")


SQUEUE = (
    "101|RUNNING|gpu|lab|normal|1:02:03|3:00:00|2026-10-07T10:00:00|"
    "2026-10-07T09:59:00|None|4|32G|1|gres/gpu:l40s:1|kn001|(null)|1:57:57|train|a|b\n"
    "102|PENDING|gpu|lab|high|0:00|1-00:00:00|2026-10-08T06:00:00|"
    "2026-10-07T11:00:00|Priority|8|64G|1|gres/gpu:h100:2||kn005|1-00:00:00|eval\n"
    "103|PENDING|gpu,cpu|lab|normal|0:00|1:00:00|N/A|"
    "2026-10-07T11:00:00|Dependency|1|4G|1|N/A||(null)|1:00:00|post\n"
)


class JobTests(unittest.TestCase):
    def test_parses_each_field_and_keeps_pipes_in_names(self):
        jobs = slurm.parse_jobs(SQUEUE)

        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]["id"], "101")
        self.assertEqual(jobs[0]["state"], "RUNNING")
        self.assertEqual(jobs[0]["tres"], "gres/gpu:l40s:1")
        self.assertEqual(jobs[0]["name"], "train|a|b")
        self.assertEqual(jobs[1]["start"], "2026-10-08T06:00:00")
        self.assertEqual(jobs[1]["nodelist"], "")
        self.assertEqual(jobs[1]["sched_nodes"], "kn005")
        self.assertEqual(jobs[0]["sched_nodes"], "(null)")
        self.assertEqual(jobs[0]["time_left"], "1:57:57")

    def test_counts_jobs_overall_and_per_qos(self):
        jobs = slurm.parse_jobs(SQUEUE)

        self.assertEqual(
            slurm.job_counts(jobs), {"running": 1, "pending": 2, "total": 3}
        )
        self.assertEqual(
            slurm.job_counts(jobs, "normal"), {"running": 1, "pending": 1, "total": 2}
        )
        self.assertEqual(
            slurm.job_counts([], "normal"), {"running": 0, "pending": 0, "total": 0}
        )

    def test_reads_how_jobs_ended_from_sacct(self):
        output = "101|COMPLETED|0:0\n102|CANCELLED by 3134|0:15\n103_2|FAILED|1:0\n\n"

        self.assertEqual(
            slurm.parse_sacct_states(output),
            {
                "101": ("COMPLETED", "0:0"),
                "102": ("CANCELLED", "0:15"),
                "103_2": ("FAILED", "1:0"),
            },
        )
        self.assertEqual(
            slurm.sacct_command(["101", "102"])[:6],
            ["sacct", "-n", "-X", "-P", "-j", "101,102"],
        )

    def test_counts_pending_jobs_toward_every_partition_they_name(self):
        self.assertEqual(
            slurm.parse_pending("gpu\ngpu,cpu\ncpu\ngpu\n"), {"gpu": 3, "cpu": 2}
        )


ASSOCIATIONS = """Association Records

ClusterName=c Account=root UserName= Partition= Priority=0 ID=1
    SharesRaw/Norm/Level/Factor=1/0.00/1/0.00
    ParentAccount= Lineage=/ DefAssoc=No
    GrpJobs=N(550) GrpJobsAccrue=N(8398)
    GrpSubmitJobs=N(15624) GrpWall=N(10045957.45)
    GrpTRES=cpu=N(6548),gres/gpu=N(611)
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
    MaxTRESPJ=
    MaxTRESPN=
ClusterName=c Account=lab UserName= Partition= Priority=0 ID=2
    SharesRaw/Norm/Level/Factor=68209/0.01/8360303/0.00
    ParentAccount=root(1) Lineage=/lab/ DefAssoc=No
    GrpJobs=20(4) GrpJobsAccrue=N(15)
    GrpSubmitJobs=N(20) GrpWall=N(104967.65)
    GrpTRES=cpu=N(96),mem=N(589824),gres/gpu=8(6)
    GrpTRESMins=cpu=N(1),gres/gpu=N(2)
    MaxJobs= MaxJobsAccrue= MaxSubmitJobs= MaxWallPJ=
    MaxTRESPJ=
    MaxTRESPN=
ClusterName=c Account=lab UserName=me(123) Partition= Priority=0 ID=3
    SharesRaw/Norm/Level/Factor=1/0.03/30/0.08
    ParentAccount= Lineage=/lab/0-me/ DefAssoc=Yes
    GrpJobs=N(1) GrpJobsAccrue=N(0)
    GrpSubmitJobs=N(2) GrpWall=N(10295.06)
    GrpTRES=cpu=N(4),gres/gpu=N(1)
    MaxJobs=10(1) MaxJobsAccrue= MaxSubmitJobs=50(2) MaxWallPJ=2880
    MaxTRESPJ=gres/gpu=4
    MaxTRESPN=cpu=32

QOS Records

QOS=normal(1)
"""


class ParseAssociationsTests(unittest.TestCase):
    def setUp(self):
        self.root, self.lab, self.mine = slurm.parse_associations(ASSOCIATIONS)

    def test_identifies_each_association(self):
        self.assertEqual((self.lab["account"], self.lab["user"]), ("lab", ""))
        self.assertEqual((self.mine["account"], self.mine["user"]), ("lab", "me"))
        self.assertEqual(self.lab["parent"], "root")
        self.assertIsNone(self.mine["parent"])
        self.assertTrue(self.mine["default"])
        self.assertFalse(self.lab["default"])
        self.assertIsNone(self.mine["partition"])

    def test_reads_the_fairshare_factor(self):
        self.assertEqual(self.mine["fairshare"], 0.08)
        self.assertEqual(self.lab["fairshare"], 0.0)

    def test_reads_group_limits_with_usage(self):
        self.assertEqual(self.lab["grp_jobs"], 20)
        self.assertIsNone(self.lab["grp_submit_jobs"])
        self.assertEqual(self.lab["job_usage"]["grp_running"], 4)
        self.assertEqual(self.lab["grp_tres"]["gres/gpu"], 8)
        self.assertIsNone(self.lab["grp_tres"]["cpu"])
        self.assertEqual(self.lab["grp_tres_usage"]["gres/gpu"], 6)

    def test_reads_per_user_job_caps_with_usage(self):
        self.assertEqual(self.mine["max_jobs"], 10)
        self.assertEqual(self.mine["max_submit_jobs"], 50)
        self.assertEqual(self.mine["job_usage"]["running"], 1)
        self.assertEqual(self.mine["job_usage"]["total"], 2)
        self.assertIsNone(self.lab["max_jobs"])

    def test_reads_per_job_and_per_node_caps(self):
        self.assertEqual(self.mine["max_wall_pj"], 2880)
        self.assertEqual(self.mine["max_tres_pj"], {"gres/gpu": 4})
        self.assertEqual(self.mine["max_tres_pn"], {"cpu": 32})
        self.assertIsNone(self.lab["max_wall_pj"])

    def test_stops_at_the_qos_section(self):
        self.assertEqual(len(slurm.parse_associations(ASSOCIATIONS)), 3)


class GpuNameTests(unittest.TestCase):
    def test_shortens_model_and_keeps_mig_profiles(self):
        for gpu, name in [
            ("nvidia_h100_80gb_hbm3_1g.10gb", "h100 1g.10gb"),
            ("nvidia_b200_2g.45gb", "b200 2g.45gb"),
            ("nvidia_a100-sxm4-40gb", "a100"),
            ("nvidia_b200", "b200"),
            ("l40s", "l40s"),
        ]:
            with self.subTest(gpu=gpu):
                self.assertEqual(slurm.short_gpu_name(gpu), name)

    def test_keeps_full_names_that_would_collide(self):
        names = slurm.gpu_display_names(
            [
                "nvidia_a100-sxm4-40gb",
                "nvidia_a100-sxm4-80gb",
                "h100",
                "nvidia_h100_80gb_hbm3",
            ]
        )

        self.assertEqual(
            names,
            {
                "nvidia_a100-sxm4-40gb": "nvidia_a100-sxm4-40gb",
                "nvidia_a100-sxm4-80gb": "nvidia_a100-sxm4-80gb",
                "h100": "h100",
                "nvidia_h100_80gb_hbm3": "nvidia_h100_80gb_hbm3",
            },
        )


if __name__ == "__main__":
    unittest.main()
