import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from sqtop import profiles

REPO = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        # Never read your own config.
        self.default_path = profiles.CONFIG_PATH
        patcher = patch.object(profiles, "CONFIG_PATH", REPO / "missing.toml")
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_reads_your_config_from_your_config_directory(self):
        config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
        self.assertEqual(
            self.default_path, Path(config_home) / "sqtop" / "config.toml"
        )
        mine = Path(self.config('[mail]\nuser = "me@example.com"\n'))
        with patch.object(profiles, "CONFIG_PATH", mine):
            self.assertEqual(profiles.load_mail()[0], "me@example.com")
            # A cluster your file leaves out gets the generic profile.
            self.assertEqual(profiles.load_profile("tamia").gpu_counts, "pow2")

    def test_writes_a_config_as_toml_that_loads_back(self):
        config = {
            "mail": {"user": "me@example.com", "type": "END"},
            "smtp": {"server": "mail.example.org", "port": 465, "password": 'p"w\\'},
            "clusters": {
                "rcl": {"gpu_counts": "all"},
                "my cluster": {"ssh": "login.example.org", "gpu_counts": "all"},
            },
        }

        text = profiles.dump_config(config)

        self.assertEqual(tomllib.loads(text), config)
        self.assertIn('\n[clusters.rcl]\ngpu_counts = "all"\n', text)
        self.assertIn('[clusters."my cluster"]', text)
        self.assertTrue(text.startswith("# Settings for sqtop"))
        self.assertEqual(profiles.dump_config({}).count("["), 0)
        with self.assertRaises(ValueError):
            profiles.dump_config({"mail": {"user": 1.5}})

    def test_saves_your_config_privately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sqtop" / "config.toml"
            with patch.object(profiles, "CONFIG_PATH", path):
                profiles.save_config({"mail": {"user": "me@example.com"}})

                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(
                    profiles.load_mail(), ("me@example.com", "BEGIN,END,FAIL")
                )
                self.assertEqual(sorted(path.parent.iterdir()), [path])

    def test_checks_a_whole_config_with_the_loaders_rules(self):
        profiles.check_config({})
        profiles.check_config(
            {"mail": {"user": "me@example.com"}, "clusters": {"x": {"ssh": "x"}}}
        )
        for config in (
            {"mail": {"user": "me"}},
            {"smtp": {"server": "s", "sender": "me@example.org"}},
            {"clusters": {"x": {"gpu_counts": "some"}}},
            {"clusters": {"": {"ssh": "x"}}},
            {"clusters": "x"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                profiles.check_config(config)

    def test_shortens_paths_under_home_for_messages(self):
        self.assertEqual(
            profiles.short_path(Path.home() / ".config" / "x.toml"), "~/.config/x.toml"
        )
        self.assertEqual(profiles.short_path("/etc/x.toml"), "/etc/x.toml")

    def test_gives_an_unlisted_cluster_the_generic_settings(self):
        self.assertEqual(
            profiles.load_profile("newcluster"), profiles.Profile(name="newcluster")
        )
        self.assertEqual(
            profiles.load_profile("tamia", path="/nonexistent"),
            profiles.Profile(name="tamia"),
        )

    def test_reads_every_setting(self):
        path = self.config(
            """
[clusters.mycluster]
gpu_counts = "one_or_node"
resources = "by_gpu"
ssh = "login.example.org"
"""
        )

        mine = profiles.load_profile("mycluster", path=path)

        self.assertEqual(mine.gpu_counts, "one_or_node")
        self.assertEqual(mine.resources, "by_gpu")
        self.assertEqual(mine.ssh, "login.example.org")
        self.assertEqual(profiles.load_profile("other", path=path).resources, "request")

    def test_reads_the_mail_table(self):
        self.assertEqual(
            profiles.load_mail(self.config('[mail]\nuser = "me@example.com"\n')),
            ("me@example.com", "BEGIN,END,FAIL"),
        )
        self.assertEqual(
            profiles.load_mail(
                self.config('[mail]\nuser = "me@example.com"\ntype = "end, fail"\n')
            ),
            ("me@example.com", "END,FAIL"),
        )
        self.assertIsNone(profiles.load_mail(self.config("[clusters.x]\n")))
        self.assertIsNone(profiles.load_mail("/nonexistent"))

    def test_has_no_settings_without_a_config(self):
        self.assertEqual(profiles.load_config(), {})
        self.assertIsNone(profiles.load_mail())
        self.assertIsNone(profiles.load_smtp())
        self.assertEqual(profiles.remote_profiles(), [])
        self.assertEqual(profiles.load_profile("tamia"), profiles.Profile("tamia"))

    def test_reads_the_smtp_table(self):
        path = self.config(
            '[mail]\nuser = "me@example.com"\n'
            '[smtp]\nserver = "mail.example.org"\nsender = "me@example.org"\n'
            'password = "secret"\n'
        )

        smtp = profiles.load_smtp(path)

        self.assertEqual(
            smtp,
            profiles.Smtp(
                "mail.example.org", 587, "me@example.org", "me@example.com", "secret"
            ),
        )
        self.assertNotIn("secret", repr(smtp))
        path = self.config(
            '[smtp]\nserver = "mail.example.org"\nport = 465\n'
            'sender = "me@example.org"\nrecipient = "you@example.org"\n'
        )
        self.assertEqual(
            profiles.load_smtp(path),
            profiles.Smtp("mail.example.org", 465, "me@example.org", "you@example.org"),
        )
        self.assertIsNone(profiles.load_smtp(self.config("[mail]\n")))

    def test_rejects_a_broken_smtp_table(self):
        good = 'server = "mail.example.org"\nsender = "me@example.org"\n'
        for text in (
            'smtp = "mail.example.org"\n',
            '[smtp]\nsender = "me@example.org"\nrecipient = "you@example.org"\n',
            f"[smtp]\n{good}",
            f'[smtp]\n{good}recipient = "you@example.org"\nport = "587"\n',
            f'[smtp]\n{good}recipient = "you@example.org"\nuser = "me"\n',
            f'[smtp]\n{good}recipient = "you@example.org"\npassword = 1\n',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                profiles.load_smtp(self.config(text))

    def test_rejects_a_broken_mail_table(self):
        for text in (
            'mail = "me@example.com"\n',
            "[mail]\n",
            '[mail]\nuser = "me"\n',
            '[mail]\nuser = "me@example.com"\ntype = "SOMETIMES"\n',
            '[mail]\nuser = "me@example.com"\nto = "x"\n',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                profiles.load_mail(self.config(text))

    def test_rejects_unknown_or_invalid_settings(self):
        for text in (
            '[clusters.x]\nwalltimes = ["0-01:00:00"]\n',
            '[clusters.x]\nssh = ""\n',
            '[clusters.x]\ngpu_counts = "some"\n',
            '[clusters.x]\nresources = "none"\n',
            'clusters = "x"\n',
            "[clusters.x\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                profiles.load_profile("x", path=self.config(text))


if __name__ == "__main__":
    unittest.main()
