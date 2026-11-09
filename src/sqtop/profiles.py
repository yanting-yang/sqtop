"""Per-cluster settings, kept as data so a new cluster needs no code.

Everything else (hardware, partitions, time tiers, accounts, QOS limits) is
read from Slurm. A profile only records what Slurm cannot tell us: site
rules about GPU request sizes and about sizing CPUs and memory, and an ssh
host to reach the cluster from elsewhere.

Profiles live in the config file with the email settings (`[mail]` for
Slurm's job emails, `[smtp]` for the dashboard's hourly email): your own
`~/.config/sqtop/config.toml`, which the dashboard's config form (`Config`)
writes after checking the same rules (`check_config`, `save_config`).
Without it there are no profiles and no email.
"""

import dataclasses
import json
import os
import re
import tomllib
from pathlib import Path

# Job states `--mail-type` accepts (sbatch(1)).
MAIL_TYPES = frozenset(
    {
        "NONE",
        "BEGIN",
        "END",
        "FAIL",
        "REQUEUE",
        "ALL",
        "INVALID_DEPEND",
        "STAGE_OUT",
        "TIME_LIMIT",
        "TIME_LIMIT_90",
        "TIME_LIMIT_80",
        "TIME_LIMIT_50",
        "ARRAY_TASKS",
    }
)

# Your settings, in your config directory so a checkout and an installed copy
# share them and the SMTP password stays out of any repository. The
# dashboard's config form writes it; without it there are no settings at all.
CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    / "sqtop"
    / "config.toml"
)
SMTP_PORT = 587

GPU_COUNT_MODES = ("pow2", "all", "whole_node", "one_or_node")
RESOURCE_MODES = ("request", "by_gpu")


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str = "generic"
    # Which GPU counts to probe for each type, up to the per-job ceiling.
    gpu_counts: str = "pow2"
    # Whether GPU requests ask for the request's CPUs and memory, or leave
    # them to a site that sizes them by GPU and refuses --mem (`by_gpu`).
    resources: str = "request"
    # ssh destination of a login node, to show this cluster from elsewhere.
    ssh: str | None = None


def short_path(path):
    """`path` with your home directory as `~`, for messages."""
    try:
        return f"~/{Path(path).relative_to(Path.home())}"
    except ValueError:
        return str(path)


BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _toml_key(key):
    return key if BARE_KEY.fullmatch(key) else json.dumps(key)


def _toml_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, list)):
        raise ValueError(f"cannot write {value!r} to the config")
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    # TOML basic strings take JSON's escapes.
    return json.dumps(value) if isinstance(value, str) else str(value)


def dump_config(config):
    """`config` (as `load_config` returns it) as TOML text.

    Only the strings, integers and lists of strings the settings use are
    written; `[mail]` and `[smtp]` come first, then one table per cluster.
    """
    lines = [
        "# Settings for sqtop, written by its config form (Config). Each",
        "# setting is explained in the README under Configuration.",
    ]
    tables = [(name, config[name]) for name in ("mail", "smtp") if name in config]
    tables += [
        (f"clusters.{_toml_key(name)}", settings)
        for name, settings in config.get("clusters", {}).items()
    ]
    for name, settings in tables:
        lines += ["", f"[{name}]"]
        lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in settings.items()]
    return "\n".join(lines) + "\n"


def check_config(config):
    """Raise `ValueError` for anything the loaders would refuse in `config`."""
    parse_mail(config)
    parse_smtp(config)
    clusters = config.get("clusters", {})
    if not isinstance(clusters, dict):
        raise ValueError("[clusters] must be a table")
    for name, settings in clusters.items():
        if not name.strip():
            raise ValueError("a cluster needs a name (ClusterName in scontrol)")
        build_profile(name, settings)


def save_config(config):
    """Write `config` to your config file, privately (0600) and in one go."""
    text = dump_config(config)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(partial, CONFIG_PATH)


def load_config(path=None):
    """The whole config file (`CONFIG_PATH` by default), or {} if missing."""
    if path is None:
        path = CONFIG_PATH
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}


def load_mail(path=None):
    """Return (user, type) from the `[mail]` table, or None without one.

    Every cluster's commands then ask Slurm to email `user` on the job
    states listed in `type` (default BEGIN,END,FAIL).
    """
    return parse_mail(load_config(path))


def parse_mail(config):
    mail = config.get("mail")
    if mail is None:
        return None
    if not isinstance(mail, dict):
        raise ValueError("[mail] must be a table")
    unknown = set(mail) - {"user", "type"}
    if unknown:
        raise ValueError(f"[mail]: unknown settings {sorted(unknown)}")
    user = mail.get("user")
    if not isinstance(user, str) or "@" not in user or " " in user.strip():
        raise ValueError("[mail]: user must be an email address")
    kind = str(mail.get("type", "BEGIN,END,FAIL")).upper().replace(" ", "")
    invalid = set(kind.split(",")) - MAIL_TYPES
    if invalid:
        raise ValueError(f"[mail]: unknown type {sorted(invalid)}")
    return user.strip(), kind


@dataclasses.dataclass(frozen=True)
class Smtp:
    """The server the dashboard's hourly email goes through, and its reader."""

    server: str
    port: int
    sender: str
    recipient: str
    # Kept out of repr, so it never lands in a log or traceback.
    password: str | None = dataclasses.field(default=None, repr=False)


def load_smtp(path=None):
    """Return the `[smtp]` table as `Smtp`, or None without one.

    `port` defaults to 587 (STARTTLS; 465 means SSL), `sender` is also the
    login, and `recipient` defaults to the `[mail]` address.
    """
    return parse_smtp(load_config(path))


def parse_smtp(config):
    smtp = config.get("smtp")
    if smtp is None:
        return None
    if not isinstance(smtp, dict):
        raise ValueError("[smtp] must be a table")
    unknown = set(smtp) - {field.name for field in dataclasses.fields(Smtp)}
    if unknown:
        raise ValueError(f"[smtp]: unknown settings {sorted(unknown)}")
    server, sender = smtp.get("server"), smtp.get("sender")
    if not isinstance(server, str) or not server.strip():
        raise ValueError("[smtp]: server must be a host name")
    if not isinstance(sender, str) or "@" not in sender:
        raise ValueError("[smtp]: sender must be an email address")
    recipient = smtp.get("recipient")
    if recipient is None:
        mail = parse_mail(config)
        recipient = mail and mail[0]
    if not isinstance(recipient, str) or "@" not in recipient:
        raise ValueError("[smtp]: set recipient, or user in [mail]")
    port = smtp.get("port", SMTP_PORT)
    if type(port) is not int or not 0 < port < 65536:
        raise ValueError("[smtp]: port must be a port number")
    password = smtp.get("password")
    if password is not None and not isinstance(password, str):
        raise ValueError("[smtp]: password must be a string")
    return Smtp(
        server.strip(), port, sender.strip(), recipient.strip(), password or None
    )


def load_profiles(path=None):
    """Read the `[clusters.<name>]` tables, or none if the file is missing."""
    clusters = load_config(path).get("clusters", {})
    if not isinstance(clusters, dict):
        raise ValueError("[clusters] must be a table")
    return clusters


def build_profile(name, settings):
    """Validate raw settings and return a `Profile`."""
    known = {field.name for field in dataclasses.fields(Profile)} - {"name"}
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"cluster {name!r}: unknown settings {sorted(unknown)}")

    gpu_counts = settings.get("gpu_counts", "pow2")
    if gpu_counts not in GPU_COUNT_MODES:
        raise ValueError(
            f"cluster {name!r}: gpu_counts must be one of {', '.join(GPU_COUNT_MODES)}"
        )

    resources = settings.get("resources", "request")
    if resources not in RESOURCE_MODES:
        raise ValueError(
            f"cluster {name!r}: resources must be one of {', '.join(RESOURCE_MODES)}"
        )

    ssh = settings.get("ssh")
    if ssh is not None and (not isinstance(ssh, str) or not ssh.strip()):
        raise ValueError(f"cluster {name!r}: ssh must be a host name or alias")

    return Profile(name=name, gpu_counts=gpu_counts, resources=resources, ssh=ssh)


def load_profile(cluster, path=None):
    """Return the profile for `cluster`, or the generic one under its name."""
    settings = load_profiles(path).get(cluster, {})
    return build_profile(cluster or "generic", settings)


def remote_profiles(path=None):
    """Profiles of the clusters reachable over ssh, in file order."""
    profiles = [
        build_profile(name, settings) for name, settings in load_profiles(path).items()
    ]
    return [profile for profile in profiles if profile.ssh]
