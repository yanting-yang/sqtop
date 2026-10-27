import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from slurmtop import cli, hosts, probes, profiles, snapshot


def profile_tables(**tables):
    return patch.object(profiles, "load_profiles", return_value=tables)


class FindClustersTests(unittest.TestCase):
    def find(self, local_config, **tables):
        with (
            patch.object(hosts.Local, "run", return_value=local_config),
            profile_tables(**tables),
        ):
            return cli.find_clusters()

    def test_lists_this_cluster_then_each_one_with_an_ssh_host(self):
        clusters = self.find(
            "ClusterName = killarney\n",
            fir={"ssh": "fir.example.org"},
            rcl={"gpu_counts": "all"},
            tamia={"ssh": "tamia.example.org", "gpu_counts": "whole_node"},
        )

        self.assertEqual([c.name for c in clusters], ["killarney", "fir", "tamia"])
        self.assertIsInstance(clusters[0].host, hosts.Local)
        self.assertEqual(clusters[1].host.ssh, "fir.example.org")
        self.assertEqual(clusters[2].profile.gpu_counts, "whole_node")

    def test_skips_a_remote_entry_for_the_cluster_it_runs_on(self):
        clusters = self.find(
            "ClusterName = tamia\n", tamia={"ssh": "tamia.example.org"}
        )

        self.assertEqual([c.name for c in clusters], ["tamia"])
        self.assertIsNone(clusters[0].host.ssh)
        self.assertEqual(clusters[0].profile.ssh, "tamia.example.org")

    def test_shows_only_remote_clusters_where_slurm_is_missing(self):
        clusters = self.find(None, fir={"ssh": "fir.example.org"})

        self.assertEqual([c.name for c in clusters], ["fir"])


class MainTests(unittest.TestCase):
    def setUp(self):
        self.clusters = [
            snapshot.Cluster("rcl", profiles.Profile("rcl"), Mock(ssh=None))
        ]
        for target, name, value in [
            (cli, "find_clusters", self.clusters),
            (cli.report, "print_reports", None),
            (profiles, "load_mail", None),
        ]:
            patcher = patch.object(target, name, return_value=value)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def test_prints_reports_when_output_is_not_a_terminal(self):
        with patch.object(cli.sys.stdout, "isatty", return_value=False):
            cli.main([])

        self.print_reports.assert_called_once_with(self.clusters, probes.Settings())

    def test_reports_only_this_cluster_when_there_is_one(self):
        remote = snapshot.Cluster("fir", profiles.Profile("fir"), Mock(ssh="fir"))
        self.find_clusters.return_value = [*self.clusters, remote]
        with patch.object(cli.sys.stdout, "isatty", return_value=False):
            cli.main([])

        self.print_reports.assert_called_once_with(self.clusters, probes.Settings())

    def test_reports_remote_clusters_where_slurm_is_missing(self):
        remote = snapshot.Cluster("fir", profiles.Profile("fir"), Mock(ssh="fir"))
        self.find_clusters.return_value = [remote]
        with patch.object(cli.sys.stdout, "isatty", return_value=False):
            cli.main([])

        self.print_reports.assert_called_once_with([remote], probes.Settings())

    def test_adds_email_notification_from_the_config(self):
        self.load_mail.return_value = ("me@example.com", "END,FAIL")
        app = Mock()
        with (
            patch.object(cli.sys.stdout, "isatty", return_value=True),
            patch("slurmtop.tui.SlurmtopApp", return_value=app) as app_class,
        ):
            cli.main([])

        settings = app_class.call_args.args[1]
        self.assertEqual(settings.mail_user, "me@example.com")
        self.assertEqual(settings.mail_type, "END,FAIL")

    def test_opens_the_dashboard_on_a_terminal(self):
        app = Mock()
        with (
            patch.object(cli.sys.stdout, "isatty", return_value=True),
            patch("slurmtop.tui.SlurmtopApp", return_value=app) as app_class,
        ):
            cli.main([])

        self.print_reports.assert_not_called()
        # Remote clusters are contacted when their tab opens, not up front.
        app_class.assert_called_once_with(self.clusters, probes.Settings())
        app.run.assert_called_once_with()

    def test_accepts_no_arguments(self):
        for argv in (["rcl"], ["-c", "8"], ["--text"]):
            stderr = StringIO()
            with (
                self.subTest(argv=argv),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                cli.main(argv)

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("unrecognized arguments", stderr.getvalue())
        self.find_clusters.assert_not_called()

    def test_still_explains_itself_with_help(self):
        stdout = StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        text = " ".join(stdout.getvalue().split())
        self.assertIn(profiles.short_path(profiles.CONFIG_PATH), text)
        self.assertIn("Config edits that config", text)

    def test_exits_with_a_message_for_a_broken_config(self):
        self.find_clusters.side_effect = ValueError("bad setting")
        with self.assertRaises(SystemExit) as raised:
            cli.main([])

        self.assertIn("bad setting", str(raised.exception.code))

    def test_exits_when_there_is_nothing_to_show(self):
        self.find_clusters.return_value = []
        with self.assertRaises(SystemExit) as raised:
            cli.main([])

        self.assertIn(
            f"no cluster in {profiles.short_path(profiles.CONFIG_PATH)} has an ssh host",
            str(raised.exception.code),
        )


if __name__ == "__main__":
    unittest.main()
