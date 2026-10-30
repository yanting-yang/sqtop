import dataclasses
import datetime
import smtplib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fakes import FakeHost

from sqtop import mailer, probes, profiles, snapshot

from test_report import make_snapshot

NOW = datetime.datetime(2026, 10, 16, 15, 0)
HOUR = datetime.timedelta(hours=1)
SMTP = profiles.Smtp(
    "mail.example.org", 587, "me@example.org", "you@example.org", "secret"
)


def job(job_id, state="PENDING", **fields):
    return {
        "id": job_id,
        "name": "train",
        "state": state,
        "qos": "normal",
        "start": "N/A",
        "nodelist": "",
        "sched_nodes": "(null)",
        "reason": "Priority" if state == "PENDING" else "None",
        **fields,
    }


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)


class ComposeTests(unittest.TestCase):
    def test_sends_the_report_under_one_subject_as_text_and_html(self):
        message = mailer.compose(SMTP, "a | b <c>\n")

        self.assertEqual(message["Subject"], "sqtop")
        self.assertEqual(message["From"], "sqtop <me@example.org>")
        self.assertEqual(message["To"], "you@example.org")
        self.assertTrue(message["Message-ID"].endswith("@example.org>"))
        self.assertIsNone(message["In-Reply-To"])
        self.assertEqual(message.get_body(("plain",)).get_content(), "a | b <c>\n")
        html = message.get_body(("html",)).get_content()
        self.assertIn("<pre", html)
        self.assertIn("a | b &lt;c&gt;", html)

    def test_replies_to_the_first_email(self):
        message = mailer.compose(SMTP, "text", "<first@example.org>")

        self.assertEqual(message["In-Reply-To"], "<first@example.org>")
        self.assertEqual(message["References"], "<first@example.org>")


class SendTests(unittest.TestCase):
    def test_encrypts_with_starttls_before_logging_in(self):
        connection = MagicMock()
        connection.has_extn.return_value = True
        message = mailer.compose(SMTP, "text")
        with patch.object(mailer.smtplib, "SMTP", return_value=connection) as smtp:
            mailer.send(SMTP, message)

        smtp.assert_called_once_with(
            "mail.example.org", 587, timeout=mailer.SMTP_TIMEOUT
        )
        calls = [name for name, _, _ in connection.method_calls]
        self.assertLess(calls.index("starttls"), calls.index("login"))
        connection.login.assert_called_once_with("me@example.org", "secret")
        connection.send_message.assert_called_once_with(message)

    def test_uses_ssl_on_port_465(self):
        smtp = dataclasses.replace(SMTP, port=465)
        connection = MagicMock()
        with patch.object(mailer.smtplib, "SMTP_SSL", return_value=connection) as ssl:
            mailer.send(smtp, mailer.compose(smtp, "text"))

        self.assertEqual(ssl.call_args.args, ("mail.example.org", 465))
        connection.starttls.assert_not_called()
        connection.login.assert_called_once_with("me@example.org", "secret")

    def test_never_sends_the_password_unencrypted(self):
        connection = MagicMock()
        connection.has_extn.return_value = False
        with (
            patch.object(mailer.smtplib, "SMTP", return_value=connection),
            self.assertRaises(smtplib.SMTPException),
        ):
            mailer.send(SMTP, mailer.compose(SMTP, "text"))

        connection.login.assert_not_called()
        connection.send_message.assert_not_called()


class JobChangeTests(unittest.TestCase):
    def test_lists_each_change_since_the_last_email(self):
        previous = mailer.job_states(
            [job("1"), job("2", "RUNNING"), job("3"), job("4"), job("5"), job("7")]
        )
        jobs = [
            job("1", "RUNNING", nodelist="kn001"),
            job("2", "RUNNING", nodelist="kn002"),
            job("5", "SUSPENDED"),
            job("6", start="2026-10-16T18:30:00"),
        ]
        ended = {"3": ("CANCELLED", "0:0"), "4": ("FAILED", "1:0")}

        self.assertEqual(
            mailer.job_changes(previous, jobs, ended),
            [
                "1 (train) started on kn001",
                "5 (train) is now SUSPENDED",
                "6 (train) submitted (Priority), expected to start 10-16 18:30",
                "3 (train) ended: CANCELLED",
                "4 (train) ended: FAILED, exit code 1:0",
                "7 (train) left the queue",
            ],
        )

    def test_asks_sacct_how_jobs_ended(self):
        host = FakeHost({"sacct": "3|CANCELLED by 1234|0:0\n4|FAILED|1:0\n"})

        states = mailer.final_states(host, ["3", "4", "12_[1-9]"])

        self.assertEqual(states, {"3": ("CANCELLED", "0:0"), "4": ("FAILED", "1:0")})
        self.assertEqual(
            host.batches,
            [[["sacct", "-n", "-X", "-P", "-j", "3,4", "-o", "JobID,State,ExitCode"]]],
        )

    def test_does_without_sacct(self):
        self.assertEqual(mailer.final_states(FakeHost(), ["3"]), {})
        self.assertEqual(mailer.final_states(FakeHost(lost="gone"), ["3"]), {})
        host = FakeHost()
        self.assertEqual(mailer.final_states(host, []), {})
        self.assertEqual(host.batches, [])


class SendReportTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.directory / "state" / "mail.json"
        self.host = FakeHost({"sacct": "1|COMPLETED|0:0\n"})
        self.clusters = [snapshot.Cluster("here", profiles.Profile("here"), self.host)]
        patcher = patch.object(mailer, "send")
        self.send_mail = patcher.start()
        self.addCleanup(patcher.stop)

    def send(self, jobs, now, clusters=None):
        def print_cluster(cluster, settings, out):
            print(f"REPORT {cluster.name} -c{settings.cpus}", file=out)
            return make_snapshot(jobs)

        targets = [
            (cluster, probes.Settings()) for cluster in clusters or self.clusters
        ]
        with patch.object(mailer.report, "print_cluster", side_effect=print_cluster):
            return mailer.send_report(targets, SMTP, self.path, now)

    def text(self, message):
        return message.get_body(("plain",)).get_content()

    def test_threads_every_email_under_the_first(self):
        first = self.send([job("1")], NOW)
        second = self.send([job("1")], NOW + HOUR)
        third = self.send([job("1")], NOW + 2 * HOUR)

        self.assertIsNone(first["In-Reply-To"])
        for message in (second, third):
            self.assertEqual(message["Subject"], "sqtop")
            self.assertEqual(message["In-Reply-To"], first["Message-ID"])
            self.assertEqual(message["References"], first["Message-ID"])
        self.assertEqual(self.send_mail.call_count, 3)

    def test_leads_with_job_counts_then_what_changed(self):
        first = self.send([job("1", "RUNNING", nodelist="kn001"), job("2")], NOW)
        second = self.send([job("2", "RUNNING", nodelist="kn002")], NOW + HOUR)
        third = self.send([job("2", "RUNNING", nodelist="kn002")], NOW + 2 * HOUR)

        self.assertEqual(
            self.text(first), "here: 1 running, 1 pending\n\nREPORT here -c32\n"
        )
        self.assertEqual(
            self.text(second),
            "here: 1 running, 0 pending. Since 15:00:\n"
            "  2 (train) started on kn002\n"
            "  1 (train) ended: COMPLETED\n"
            "\nREPORT here -c32\n",
        )
        self.assertEqual(
            self.text(third),
            "here: 1 running, 0 pending; no changes since 16:00\n\nREPORT here -c32\n",
        )

    def test_dates_changes_since_an_earlier_day(self):
        self.send([job("1")], NOW)
        message = self.send([job("1")], NOW + 24 * HOUR)

        self.assertIn("no changes since 2026-10-16 15:00", self.text(message))

    def test_separates_the_reports_of_several_clusters(self):
        clusters = [
            snapshot.Cluster(name, profiles.Profile(name), FakeHost(ssh=name))
            for name in ("fir", "tamia")
        ]

        text = self.text(self.send([], NOW, clusters))

        self.assertTrue(text.startswith("fir: 0 running, 0 pending\n"))
        self.assertIn("tamia: 0 running, 0 pending\n", text)
        self.assertIn(f"REPORT fir -c32\n{'=' * 80}\n\nREPORT tamia -c32\n", text)

    def test_reports_each_cluster_with_its_own_request(self):
        def print_cluster(cluster, settings, out):
            print(f"REPORT {cluster.name} -c{settings.cpus}", file=out)
            return make_snapshot()

        targets = [(self.clusters[0], probes.Settings(cpus=8))]
        with patch.object(mailer.report, "print_cluster", side_effect=print_cluster):
            message = mailer.send_report(targets, SMTP, self.path, NOW)

        self.assertIn("REPORT here -c8", self.text(message))

    def test_says_when_a_cluster_cannot_be_read(self):
        with patch.object(mailer.report, "print_cluster", return_value=None):
            message = mailer.send_report(
                [(self.clusters[0], probes.Settings())], SMTP, self.path, NOW
            )

        self.assertTrue(self.text(message).startswith("here: cannot read your jobs"))

    def test_keeps_the_state_of_the_last_email_that_went_out(self):
        self.send([job("1")], NOW)
        saved = self.path.read_text()
        self.send_mail.side_effect = smtplib.SMTPException("down")

        with self.assertRaises(smtplib.SMTPException):
            self.send([job("2")], NOW + HOUR)

        self.assertEqual(self.path.read_text(), saved)


if __name__ == "__main__":
    unittest.main()
