"""Assemble what the caller can use right now from the parsers in `slurm`.

A `Collector` gathers partitions, hardware, account/QOS limits and the
caller's jobs into a `Snapshot`, which both the TUI and the text report
render. It runs Slurm through a host from `hosts`, so the cluster may be
this one or one reached over ssh. Nothing here is cluster-specific.
"""

import dataclasses
import datetime
import grp
import os
import platform
import re
from collections import Counter, defaultdict

from . import slurm

# The controller's reply when the caller may not use an account or QOS.
INVALID_SCOPE = re.compile(r"Invalid (?:account|qos)", re.IGNORECASE)

TRES_NAMES = {"cpu": "CPUs", "mem": "memory", "node": "nodes", "gres/gpu": "GPUs"}


@dataclasses.dataclass(frozen=True)
class Scope:
    """An account/QOS pair the caller can submit under.

    The default flags mark what commands may leave out because Slurm picks
    it anyway: the account when it is the user's only one, and the account's
    default QOS.
    """

    account: str | None = None
    qos: str | None = None
    default_account: bool = False
    default_qos: bool = False
    # The user's default account (DefAssoc), even when commands name it.
    user_default: bool = False

    @property
    def label(self):
        if self.account is None and self.qos is None:
            return "default account/QOS"
        return f"{self.account or '-'}/{self.qos or '-'}"

    def directives(self):
        """`--account`/`--qos`, left out when Slurm would pick them anyway."""
        flags = []
        if self.account and not self.default_account:
            flags.append(f"--account={self.account}")
        if self.qos and not self.default_qos:
            flags.append(f"--qos={self.qos}")
        return flags


@dataclasses.dataclass(frozen=True)
class Limit:
    """One cap and the usage it is measured against.

    `max` is an int, None for no cap, or "?" when unknown; `used` is an int,
    None when usage does not apply (per-job caps), or "?" when unknown.
    `key` is "jobs", "submit", "wall" or "tres:<name>"; memory is in MB and
    wall time in minutes.
    """

    scope: str
    name: str
    key: str
    max: object
    used: object = None
    per_user: bool = False


# What `ClusterInfo.detect` reads on a remote login node, in one session.
REMOTE_DETECT_COMMANDS = [
    ["printenv", "PATH"],
    ["id", "-un"],
    ["id", "-Gn"],
    ["hostname", "-s"],
    ["sh", "-c", 'getent passwd "$(id -un)" | cut -d: -f7'],
    slurm.CONFIG_COMMAND,
]


@dataclasses.dataclass
class ClusterInfo:
    name: str
    version: str
    user: str
    hostname: str
    groups: frozenset | None
    shell: str = "bash"
    # The ssh destination for a remote cluster; None for this machine.
    ssh: str | None = None

    @classmethod
    def detect(cls, host):
        """Read the cluster name, Slurm version and the caller's identity.

        On a remote host this also records the login shell's PATH on the host,
        which later commands need to find Slurm.
        """
        if host.ssh is None:
            output = host.run(slurm.CONFIG_COMMAND)
            config = {} if output is None else slurm.parse_config(output)
            user = slurm.current_user()
            return cls(
                name=config.get("ClusterName", ""),
                version=config.get("SLURM_VERSION", "?"),
                user=user,
                hostname=platform.node().split(".")[0],
                groups=user_groups(user),
                shell=slurm.login_shell(),
            )

        path, user, groups, hostname, shell, config = host.run_batch(
            REMOTE_DETECT_COMMANDS, login_path=True
        )
        host.path = (path or "").strip() or None
        config = {} if config is None else slurm.parse_config(config)
        return cls(
            name=config.get("ClusterName", ""),
            version=config.get("SLURM_VERSION", "?"),
            user=(user or "").strip(),
            hostname=(hostname or host.ssh).strip(),
            groups=None if groups is None else frozenset(groups.split()),
            shell=os.path.basename((shell or "").strip()) or "bash",
            ssh=host.ssh,
        )


@dataclasses.dataclass
class Cluster:
    """A cluster to show: its tab name, its profile and where Slurm runs."""

    name: str
    profile: object
    host: object


@dataclasses.dataclass
class Snapshot:
    taken_at: datetime.datetime
    nodes: list
    partitions: list
    hardware: list
    scopes: list
    qos_records: dict
    associations: list
    jobs: list | None
    notes: list
    # Display names for this cluster's GPU types (`slurm.gpu_display_names`).
    gpu_names: dict = dataclasses.field(default_factory=dict)

    def preferred_scope(self):
        """The pair Slurm uses by default, or the first one there is."""
        return next(
            (s for s in self.scopes if s.user_default and s.default_qos),
            self.scopes[0] if self.scopes else Scope(),
        )

    def limits(self, scope, user):
        record = self.qos_records.get(scope.qos) if scope.qos else None
        return scope_limits(scope, record, self.associations, user, self.jobs)


def user_groups(user):
    """Names of the user's Unix groups, or None when they cannot be read."""
    try:
        gids = os.getgrouplist(user, os.getgid())
    except (KeyError, OSError):
        return None
    names = set()
    for gid in gids:
        try:
            names.add(grp.getgrgid(gid).gr_name)
        except KeyError:
            continue
    return frozenset(names)


# --- Accounts and QOS -----------------------------------------------------------


def accepted_pairs(host, pairs):
    """The account/QOS pairs Slurm does not reject as invalid.

    Submits a minimal CPU-only `sbatch --test-only` per pair, all in one
    session; an invalid account or QOS reply means the caller's associations
    lack the pair. Any other outcome, including a failed probe or a partition
    that refuses the QOS, keeps it.
    """
    commands = [
        slurm.test_command(
            "sbatch",
            ["--test-only", f"--account={account}", f"--qos={qos}", "-t1:00:00"],
        )
        for account, qos in pairs
    ]
    rejected = set()

    def landed(index, returncode, output):
        if INVALID_SCOPE.search(output):
            rejected.add(pairs[index])

    host.run_stream(commands, landed)
    return set(pairs) - rejected


def default_qos(qos_names, configured=None):
    """The QOS Slurm assigns when a job names none.

    Mirrors slurmctld: the association's default if set, else its only QOS,
    else `normal`. Returns None when `normal` is not among the choices, so
    `--qos` is always passed.
    """
    if configured:
        return configured
    if len(qos_names) == 1:
        return qos_names[0]
    return "normal" if "normal" in qos_names else None


def discover_scopes(user, assignments, assoc_output, qos_records, accept):
    """Return the caller's account/QOS pairs and notes on how they were found.

    `assignments` come from `sacctmgr` (`slurm.parse_account_qos`). When it
    failed (None; slurmdbd unreachable from the login node), every cached QOS
    is a candidate for each of the user's cached accounts, and `accept(pairs)`
    keeps those Slurm does not reject as invalid: QOS `Account Limits`
    entries track usage, not permission.
    """
    notes = []
    if assignments is None:
        accounts = slurm.parse_user_accounts(assoc_output or "", user)
        candidates = [(a, q) for a in accounts for q in sorted(qos_records)]
        accepted = accept(candidates) if candidates else set()
        assignments = {
            account: {
                "qos": [q for q in sorted(qos_records) if (account, q) in accepted],
                "default_qos": None,
            }
            for account in accounts
        }
        if accounts:
            notes.append(
                "sacctmgr unavailable: QOS access inferred by test-submitting "
                "each cached QOS."
            )
    if not assignments:
        notes.append(f"No Slurm association found for '{user}'.")

    # With several accounts, commands always name one: sites such as Fir
    # reject GPU jobs that leave the choice to the default account.
    default_account = next(iter(assignments)) if len(assignments) == 1 else None
    user_default = slurm.parse_default_account(assoc_output or "", user)
    scopes = []
    for account, entry in sorted(assignments.items()):
        if not entry["qos"]:
            notes.append(f"No usable QOS found for account '{account}'.")
        qos_default = default_qos(entry["qos"], entry["default_qos"])
        scopes += [
            Scope(
                account,
                name,
                default_account=account == default_account,
                default_qos=name == qos_default,
                user_default=account in (user_default, default_account),
            )
            for name in entry["qos"]
        ]
    return scopes, notes


def association_chain(associations, account, user):
    """The caller's associations in `account`, then the account and its parents.

    Limits on every level apply to the caller's jobs; `root` is left out.
    """
    chain = [
        record
        for record in associations
        if record["user"] == user and record["account"] == account
    ]
    by_account = {
        record["account"]: record
        for record in associations
        if not record["user"] and record["partition"] is None
    }
    seen = set()
    current = account
    while current and current != "root" and current not in seen:
        seen.add(current)
        record = by_account.get(current)
        if record is None:
            break
        chain.append(record)
        current = record["parent"]
    return chain


def fairshare(associations, account, user):
    for record in associations:
        if record["user"] == user and record["account"] == account:
            return record["fairshare"]
    return None


def tres_name(tres):
    if tres in TRES_NAMES:
        return TRES_NAMES[tres]
    if tres.startswith("gres/gpu:"):
        return f"{tres.removeprefix('gres/gpu:')} GPUs"
    return tres


def _tres_limits(scope, caps, usage=None, per_user=False):
    return [
        Limit(
            scope,
            tres_name(tres),
            f"tres:{tres}",
            value,
            None if usage is None else usage.get(tres, "?"),
            per_user,
        )
        for tres, value in sorted(caps.items())
        if value is not None
    ]


def _job_limits(scope, limits, usage, *, always=False, per_user=False):
    rows = []
    for key, limit_key, usage_key, name in (
        ("jobs", "max_jobs", "running", "running jobs"),
        ("submit", "max_submit_jobs", "total", "submitted jobs"),
    ):
        value = limits.get(limit_key, "?")
        if value is None and not always:
            continue
        if value == "?" and not always:
            continue
        rows.append(Limit(scope, name, key, value, usage.get(usage_key, "?"), per_user))
    return rows


def scope_limits(scope, record, associations, user, jobs):
    """Every cap on jobs submitted under `scope`, from its QOS and associations.

    The QOS per-user job caps are always listed (`?` when no user entry
    reveals them); other caps only when set. Per-user usage comes from the
    caller's queued jobs in that QOS, across all accounts.
    """
    limits = []
    if record is not None:
        counts = slurm.job_counts(jobs, scope.qos) if jobs is not None else {}
        user_limits = slurm.qos_user_limits(record, user)
        account_limits = slurm.qos_account_limits(record, scope.account)
        account_usage = record["account_job_usage"].get(scope.account, {})
        user_caps = user_limits.get("max_tres_pu", {})
        user_usage = record["user_tres_usage"].get(user)
        if user_usage is None and counts.get("running") == 0:
            # No tracked entry and nothing running: the caller uses none of it.
            user_usage = dict.fromkeys(user_caps, 0)
        limits += _job_limits(
            "QOS per user", user_limits, counts, always=True, per_user=True
        )
        limits += _tres_limits(
            "QOS per user", user_caps, user_usage or {}, per_user=True
        )
        limits += _job_limits("QOS per account", account_limits, account_usage)
        limits += _tres_limits(
            "QOS per account",
            account_limits.get("max_tres_pa", {}),
            record["account_tres_usage"].get(scope.account, {}),
        )
        limits += _tres_limits("QOS per job", record["max_tres_pj"])
        limits += _tres_limits("QOS per node", record["max_tres_pn"])
        if record.get("max_wall_pj") is not None:
            limits.append(
                Limit("QOS per job", "wall time", "wall", record["max_wall_pj"])
            )

    # Max* limits are inherited down to the caller's own association, so only
    # the Grp* totals are read from the account and its parents.
    for assoc in association_chain(associations, scope.account, user):
        usage = assoc["job_usage"]
        grp_usage = {
            "running": usage.get("grp_running", "?"),
            "total": usage.get("grp_total", "?"),
        }
        grp_jobs = {
            "max_jobs": assoc["grp_jobs"],
            "max_submit_jobs": assoc["grp_submit_jobs"],
        }
        if not assoc["user"]:
            where = f"Account {assoc['account']}"
            limits += _job_limits(where, grp_jobs, grp_usage)
            limits += _tres_limits(where, assoc["grp_tres"], assoc["grp_tres_usage"])
            continue
        where = f"You in {assoc['account']}"
        max_jobs = {
            "max_jobs": assoc["max_jobs"],
            "max_submit_jobs": assoc["max_submit_jobs"],
        }
        limits += _job_limits(where, max_jobs, usage, per_user=True)
        limits += _job_limits(where, grp_jobs, grp_usage, per_user=True)
        limits += _tres_limits(
            where, assoc["grp_tres"], assoc["grp_tres_usage"], per_user=True
        )
        limits += _tres_limits(f"{where}, per job", assoc["max_tres_pj"])
        limits += _tres_limits(f"{where}, per node", assoc["max_tres_pn"])
        if assoc["max_wall_pj"] is not None:
            limits.append(
                Limit(f"{where}, per job", "wall time", "wall", assoc["max_wall_pj"])
            )
    return limits


def tightest(limits, key):
    """The smallest numeric cap with this key, or None when nothing caps it."""
    values = [
        limit.max for limit in limits if limit.key == key and isinstance(limit.max, int)
    ]
    return min(values) if values else None


# --- Partitions and hardware -------------------------------------------------------


def node_category(node):
    if not slurm.node_available(node):
        return "unavailable"
    base = node["state"].split("+")[0].rstrip("*~#!%$@^-").upper()
    if base == "IDLE":
        return "idle"
    if base == "ALLOCATED":
        return "allocated"
    return "mixed"


def resource_usage(nodes):
    """Total, allocated, free and unavailable CPUs, memory (MB) and GPUs.

    Unallocated resources on nodes that cannot take jobs (down, drained,
    reserved...) count as unavailable rather than free. CPUs are the
    effective count, excluding cores reserved with CoreSpecCount.
    """
    cpus = Counter()
    mem = Counter()
    gpus = defaultdict(Counter)
    states = Counter()
    for node in nodes:
        spare = "free" if slurm.node_available(node) else "unavailable"
        states[node_category(node)] += 1

        total = node["cpu_efctv"]
        allocated = min(node["cpu_alloc"], total)
        cpus.update(total=total, allocated=allocated)
        cpus[spare] += total - allocated

        allocated = min(node["mem_alloc"], node["mem_tot"])
        mem.update(total=node["mem_tot"], allocated=allocated)
        mem[spare] += node["mem_tot"] - allocated

        for gpu, total in node["cfg_gpus"].items():
            allocated = min(node["alloc_gpus"].get(gpu, 0), total)
            gpus[gpu].update(total=total, allocated=allocated)
            gpus[gpu][spare] += total - allocated
    return {
        "cpus": cpus,
        "mem": mem,
        "gpus": {gpu: gpus[gpu] for gpu in sorted(gpus)},
        "states": states,
        "node_count": len(nodes),
    }


def partition_access(partition, accounts, qos_names, groups):
    """Return (usable, reason) for the caller; unknown inputs skip their check."""
    if partition["state"] != "UP":
        return False, f"partition is {partition['state']}"
    if accounts:
        allow = partition["allow_accounts"]
        if not any(
            (allow is None or account in allow)
            and account not in partition["deny_accounts"]
            for account in accounts
        ):
            return False, "none of your accounts is allowed"
    if qos_names:
        allow = partition["allow_qos"]
        if not any(
            (allow is None or name in allow) and name not in partition["deny_qos"]
            for name in qos_names
        ):
            return False, "none of your QOS is allowed"
    allow = partition["allow_groups"]
    if groups is not None and allow is not None and not groups & allow:
        return False, "none of your groups is allowed"
    return True, ""


def summarize_partitions(partitions, nodes, pending, scopes, groups):
    members = defaultdict(list)
    for node in nodes:
        for name in node["partitions"]:
            members[name].append(node)
    accounts = {scope.account for scope in scopes if scope.account}
    qos_names = {scope.qos for scope in scopes if scope.qos}
    summaries = []
    for partition in partitions:
        usable, reason = partition_access(partition, accounts, qos_names, groups)
        summaries.append(
            {
                **partition,
                **resource_usage(members.get(partition["name"], [])),
                # How many nodes of each hardware kind (`node_hw_key`).
                "node_kinds": Counter(
                    node_hw_key(node) for node in members.get(partition["name"], [])
                ),
                "pending": None
                if pending is None
                else pending.get(partition["name"], 0),
                "usable": usable,
                "access_note": reason,
            }
        )
    return summaries


def node_hw_key(node):
    return (
        node["cpu_efctv"],
        node["mem_tot"] // 1024,
        frozenset(node["cfg_gpus"].items()),
    )


def hw_key_label(key, names=None):
    cpus, mem_gb, gpu_frozenset = key
    names = names or {}
    parts = [f"{cpus} CPUs", f"{mem_gb} GB"]
    parts += [f"{count}x {names.get(gpu, gpu)}" for gpu, count in sorted(gpu_frozenset)]
    return " / ".join(parts)


def summarize_hardware(nodes, names=None):
    """Group nodes by CPUs, whole GiB of memory and GPUs, with usage per group."""
    groups = defaultdict(list)
    for node in nodes:
        groups[node_hw_key(node)].append(node)
    return [
        {
            "label": hw_key_label(key, names),
            "names": [node["name"] for node in groups[key]],
            **resource_usage(groups[key]),
        }
        for key in sorted(groups, key=lambda key: (sorted(key[2]), key[0], key[1]))
    ]


def gpu_capacities(nodes):
    """The most GPUs of each type that one node offers."""
    capacities = {}
    for node in nodes:
        for gpu, count in node["cfg_gpus"].items():
            capacities[gpu] = max(capacities.get(gpu, 0), count)
    return dict(sorted(capacities.items()))


# --- Collection -------------------------------------------------------------------


class Collector:
    """Fetch snapshots through `host`, discovering account/QOS pairs once.

    Each refresh is one batch of commands, so a remote cluster costs one ssh
    session. Once the caller's accounts and QOS are known, the association
    cache is fetched only for them and their parent accounts.
    """

    # Parent accounts deeper than this are not followed.
    MAX_ACCOUNT_DEPTH = 5

    def __init__(self, info, host):
        self.info = info
        self.host = host
        self.scopes = None
        self.scope_notes = []
        self.parents = set()

    def filters(self, assignments):
        """Accounts and QOS to fetch from the cache; None fetches everything."""
        if self.scopes:
            accounts = {scope.account for scope in self.scopes} | self.parents
            return accounts, {scope.qos for scope in self.scopes}
        if assignments:
            qos = {name for entry in assignments.values() for name in entry["qos"]}
            return set(assignments), qos or None
        return None, None

    def with_parent_associations(self, output, accounts):
        """Add the associations of parent accounts missing from `output`.

        Their Grp limits apply to the caller's jobs too. The records go before
        the QOS section, where the association parser stops reading.
        """
        known = set(accounts)
        for _ in range(self.MAX_ACCOUNT_DEPTH):
            missing = {
                record["parent"]
                for record in slurm.parse_associations(output)
                if not record["user"]
                and record["parent"] not in (None, "root")
                and record["parent"] not in known
            }
            if not missing:
                break
            known |= missing
            self.parents |= missing
            command = ["scontrol", "show", "assoc_mgr", "flags=assoc"]
            text = self.host.run([*command, f"accounts={','.join(sorted(missing))}"])
            if text is None:
                break
            records = text.partition("Association Records")[2].strip("\n")
            head, qos_header, qos_section = output.partition("QOS Records")
            output = f"{head.rstrip()}\n{records}\n\n{qos_header}{qos_section}"
        return output

    def collect(self):
        """Return a `Snapshot`; raises `hosts.ConnectionLost` if ssh fails."""
        user = self.info.user
        notes = []

        assignments = None
        if self.scopes is None and self.info.name:
            output = self.host.run(slurm.account_qos_command(user, self.info.name))
            assignments = None if output is None else slurm.parse_account_qos(output)
        accounts, qos = self.filters(assignments)

        nodes, partitions, assoc_output, jobs, pending = self.host.run_batch(
            [
                slurm.NODES_COMMAND,
                slurm.PARTITIONS_COMMAND,
                slurm.assoc_mgr_command(accounts, qos),
                slurm.jobs_command(user),
                slurm.PENDING_COMMAND,
            ]
        )

        if nodes is None:
            notes.append("'scontrol show node' failed: is this a Slurm login node?")
        nodes = [
            slurm.normalize_gpu_rollups(node) for node in slurm.parse_nodes(nodes or "")
        ]
        if partitions is None:
            notes.append("'scontrol show partition' failed.")
        partitions = slurm.parse_partitions(partitions or "")
        if assoc_output is None:
            notes.append("Limits unavailable: 'scontrol show assoc_mgr' failed.")
        elif accounts:
            assoc_output = self.with_parent_associations(assoc_output, accounts)
        qos_records = slurm.parse_qos_records(assoc_output or "")
        associations = slurm.parse_associations(assoc_output or "")
        if jobs is None:
            notes.append("'squeue' failed: your jobs and usage are unknown.")
        else:
            jobs = slurm.parse_jobs(jobs)

        if self.scopes is None:
            scopes, self.scope_notes = discover_scopes(
                user,
                assignments,
                assoc_output,
                qos_records,
                lambda pairs: accepted_pairs(self.host, pairs),
            )
            if scopes or assoc_output is not None:
                self.scopes = scopes
        scopes = self.scopes or []
        gpu_names = slurm.gpu_display_names(
            sorted({gpu for node in nodes for gpu in node["cfg_gpus"]})
        )

        return Snapshot(
            taken_at=datetime.datetime.now().replace(microsecond=0),
            nodes=nodes,
            partitions=summarize_partitions(
                partitions,
                nodes,
                None if pending is None else slurm.parse_pending(pending),
                scopes,
                self.info.groups,
            ),
            hardware=summarize_hardware(nodes, gpu_names),
            scopes=scopes,
            qos_records=qos_records,
            associations=associations,
            jobs=jobs,
            notes=self.scope_notes + notes,
            gpu_names=gpu_names,
        )
