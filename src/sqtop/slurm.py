"""Slurm commands and parsers for their output.

The `*_command` helpers build argument lists that a host (`hosts.Local` or
`hosts.Remote`) runs; the parsers are pure functions over captured output.
"""

import getpass
import os
import re
import shlex

SBATCH_START_PATTERN = re.compile(
    r"\bto start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\b"
)
SBATCH_PARTITION_PATTERN = re.compile(r"\bin partition (\S+)")
# sbatch runs this in place of a batch script, holding the allocation until
# the time limit or `scancel`.
SBATCH_WRAP = "sleep infinity"
# Submitted jobs send their output here: the placeholder prints nothing worth
# keeping, so no slurm-<jobid>.out is left where sbatch ran.
SBATCH_OUTPUT = "--output=/dev/null"

# Node state words (the base state or a `+FLAG`) that stop a node from taking
# new jobs; its unallocated resources count as unavailable rather than free.
UNAVAILABLE_NODE_STATES = frozenset(
    {
        "DOWN",
        "DRAIN",
        "DRAINED",
        "DRAINING",
        "FAIL",
        "FAILING",
        "FUTURE",
        "INVAL",
        "MAINT",
        "NOT_RESPONDING",
        "POWERED_DOWN",
        "POWERING_DOWN",
        "POWERING_UP",
        "REBOOT_ISSUED",
        "REBOOT_REQUESTED",
        "RESERVED",
        "UNKNOWN",
    }
)

# `squeue -o` fields for the caller's jobs. The name goes last so a `|` in it
# cannot shift the other columns.
JOB_FIELDS = (
    ("id", "%i"),
    ("state", "%T"),
    ("partition", "%P"),
    ("account", "%a"),
    ("qos", "%q"),
    ("elapsed", "%M"),
    ("time_limit", "%l"),
    ("start", "%S"),
    ("submit", "%V"),
    ("reason", "%r"),
    ("cpus", "%C"),
    ("mem", "%m"),
    ("nodes", "%D"),
    ("tres", "%b"),
    ("nodelist", "%N"),
    # Nodes the backfill scheduler plans for a pending job, or "(null)".
    ("sched_nodes", "%Y"),
    # Time left before the limit; the whole limit while pending.
    ("time_left", "%L"),
    ("name", "%j"),
)


def current_user():
    return getpass.getuser()


def login_shell():
    """The caller's shell name, used for interactive `srun --pty` commands."""
    return os.path.basename(os.environ.get("SHELL") or "bash")


# --- Durations ---------------------------------------------------------------


def parse_seconds(text):
    """Return the seconds in a Slurm time string, or None when unlimited/unset.

    Accepts `MM`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM` and `D-HH:MM:SS`.
    """
    text = text.strip()
    days = 0
    if "-" in text:
        day_text, _, text = text.partition("-")
        if not day_text.isdigit():
            return None
        days = int(day_text)
        parts = text.split(":")
        parts += ["0"] * (3 - len(parts))
    else:
        parts = text.split(":")
        if len(parts) == 1:
            parts = ["0", parts[0], "0"]
        elif len(parts) == 2:
            parts = ["0", *parts]
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    hours, minutes, seconds = (int(part) for part in parts)
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def parse_duration(text):
    """Return the minutes in a Slurm time string, or None when unlimited/unset.

    Leftover seconds round up, so a limit is never understated.
    """
    seconds = parse_seconds(text)
    return None if seconds is None else -(-seconds // 60)


def format_seconds(seconds):
    """Format seconds as Slurm's `D-HH:MM:SS`, the form every panel shows."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_duration(minutes):
    """Format minutes as the `D-HH:MM:SS` form `sbatch -t` accepts."""
    return format_seconds(minutes * 60)


def normalize_time(text):
    """Rewrite a Slurm time string as `D-HH:MM:SS`; leave other text alone."""
    seconds = parse_seconds(text)
    return text if seconds is None else format_seconds(seconds)


# --- Cluster configuration ---------------------------------------------------


def parse_config(output):
    """Parse `scontrol show config` `Key = Value` lines into a dict."""
    config = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key and " " not in key:
            config[key] = value.strip()
    return config


CONFIG_COMMAND = ["scontrol", "show", "config"]


# --- Nodes -------------------------------------------------------------------


def parse_tres_gpus(tres_str):
    gpus = {}
    for match in re.finditer(r"gres/gpu(?::([^=,]+))?=(\d+)", tres_str):
        label = match.group(1) or "gpu"
        gpus[label] = int(match.group(2))
    return gpus


def remove_untyped_gpu_rollup(gpu_counts):
    """Drop the untyped GPU total when per-model counts are available."""
    typed = {label: count for label, count in gpu_counts.items() if label != "gpu"}
    return typed if typed else gpu_counts


def short_gpu_name(gpu):
    """A readable GPU type: `nvidia_h100_80gb_hbm3_1g.10gb` -> `h100 1g.10gb`.

    Drops the vendor prefix and memory/form-factor qualifiers, and keeps a MIG
    profile apart from the model.
    """
    name = gpu.removeprefix("nvidia_")
    mig = re.search(r"[_-](\d+g\.\d+gb)$", name)
    if mig:
        name = name[: mig.start()]
    name = re.sub(r"[_-]\d+gb(?:[_-]hbm\w*)?$", "", name)
    name = re.sub(r"[_-](?:sxm\d*|pcie)(?=[_-]|$)", "", name)
    return f"{name} {mig.group(1)}" if mig else name


def gpu_display_names(gpus):
    """Map GPU types to `short_gpu_name`, keeping any that would collide whole."""
    short = {gpu: short_gpu_name(gpu) for gpu in gpus}
    taken = [name for name in short.values()]
    return {gpu: name if taken.count(name) == 1 else gpu for gpu, name in short.items()}


def normalize_gpu_rollups(node):
    node["cfg_gpus"] = remove_untyped_gpu_rollup(node["cfg_gpus"])
    node["alloc_gpus"] = remove_untyped_gpu_rollup(node["alloc_gpus"])
    return node


def parse_nodes(output):
    """Parse `scontrol show node` into dicts.

    `CfgTRES` usually lists an untyped `gres/gpu=N` beside per-model entries
    summing to the same N; `parse_tres_gpus` keeps both, and
    `normalize_gpu_rollups` drops the rollup.
    """
    nodes = []
    for block in re.split(r"\n(?=NodeName=)", output.strip()):
        if not block.strip():
            continue
        node = {}
        for pattern, key, cast in [
            (r"NodeName=(\S+)", "name", str),
            (r"State=(\S+)", "state", str),
            (r"CPUTot=(\d+)", "cpu_tot", int),
            (r"CPUAlloc=(\d+)", "cpu_alloc", int),
            (r"RealMemory=(\d+)", "mem_tot", int),
            (r"AllocMem=(\d+)", "mem_alloc", int),
        ]:
            match = re.search(pattern, block)
            node[key] = (
                cast(match.group(1)) if match else ("?" if cast is str else cast())
            )

        # Cores held back by CoreSpecCount are counted in CPUTot but cannot be
        # allocated; CPUEfctv is what the scheduler actually offers to jobs.
        match = re.search(r"CPUEfctv=(\d+)", block)
        node["cpu_efctv"] = int(match.group(1)) if match else node["cpu_tot"]

        match = re.search(r"CfgTRES=(\S+)", block)
        node["cfg_gpus"] = parse_tres_gpus(match.group(1) if match else "")
        match = re.search(r"AllocTRES=(\S*)", block)
        node["alloc_gpus"] = parse_tres_gpus(match.group(1) if match else "")

        match = re.search(r"\bPartitions=(\S+)", block)
        node["partitions"] = match.group(1).split(",") if match else []

        nodes.append(node)
    return nodes


def node_available(node):
    """Whether the node can take new jobs, judged from every state word."""
    words = re.split(r"[+*~#!%$@^-]", node["state"].upper())
    return not UNAVAILABLE_NODE_STATES.intersection(words)


NODES_COMMAND = ["scontrol", "show", "node"]


# --- Partitions --------------------------------------------------------------


def _name_set(value):
    """`ALL` (or unset) means no restriction (None); otherwise a set of names."""
    if value in (None, "", "ALL"):
        return None
    return {name for name in value.split(",") if name}


def _count(value):
    return int(value) if value and value.isdigit() else None


def parse_partitions(output):
    """Parse `scontrol show partition -o`, one partition per line.

    Allow lists are None when unrestricted; deny lists are sets. Times are in
    minutes, with None meaning unlimited or unset.
    """
    partitions = []
    for line in output.splitlines():
        fields = dict(re.findall(r"(\S+?)=(\S*)", line))
        if "PartitionName" not in fields:
            continue
        qos = fields.get("QoS", "N/A")
        partitions.append(
            {
                "name": fields["PartitionName"],
                "state": fields.get("State", "?"),
                "default": fields.get("Default") == "YES",
                "max_time": parse_duration(fields.get("MaxTime", "UNLIMITED")),
                "default_time": parse_duration(fields.get("DefaultTime", "NONE")),
                "allow_accounts": _name_set(fields.get("AllowAccounts")),
                "deny_accounts": _name_set(fields.get("DenyAccounts")) or set(),
                "allow_qos": _name_set(fields.get("AllowQos")),
                "deny_qos": _name_set(fields.get("DenyQos")) or set(),
                "allow_groups": _name_set(fields.get("AllowGroups")),
                "qos": None if qos in ("", "N/A") else qos,
                "nodes": fields.get("Nodes", ""),
                "total_nodes": _count(fields.get("TotalNodes")) or 0,
                "total_cpus": _count(fields.get("TotalCPUs")) or 0,
                "max_nodes": _count(fields.get("MaxNodes")),
                "priority_tier": _count(fields.get("PriorityTier")) or 0,
                "priority_job_factor": _count(fields.get("PriorityJobFactor")) or 0,
                "preempt_mode": fields.get("PreemptMode", "OFF"),
            }
        )
    return partitions


PARTITIONS_COMMAND = ["scontrol", "show", "partition", "-o"]


# --- Jobs --------------------------------------------------------------------


def parse_jobs(output):
    """Parse `squeue -h -o` output produced with the `JOB_FIELDS` format."""
    keys = [key for key, _ in JOB_FIELDS]
    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = line.split("|", len(keys) - 1)
        if len(values) == len(keys):
            jobs.append(dict(zip(keys, values)))
    return jobs


def jobs_command(user):
    """`squeue` for the user's queued and running jobs, in `JOB_FIELDS` order."""
    return ["squeue", "-h", "-u", user, "-o", "|".join(c for _, c in JOB_FIELDS)]


PENDING_COMMAND = ["squeue", "-h", "-t", "PD", "-o", "%P"]


def parse_pending(output):
    """Count pending jobs per partition across all users.

    A job submitted to several partitions counts toward each of them.
    """
    counts = {}
    for line in output.split():
        for partition in line.split(","):
            counts[partition] = counts.get(partition, 0) + 1
    return counts


def job_counts(jobs, qos=None):
    """Running/pending/total counts of `jobs`, optionally within one QOS."""
    states = [job["state"] for job in jobs if qos is None or job["qos"] == qos]
    return {
        "running": states.count("RUNNING"),
        "pending": states.count("PENDING"),
        "total": len(states),
    }


def sacct_command(ids):
    """`sacct` for how jobs that left the queue ended, one line per job."""
    fields = "JobID,State,ExitCode"
    return ["sacct", "-n", "-X", "-P", "-j", ",".join(ids), "-o", fields]


def parse_sacct_states(output):
    """Map job ids to (state, exit code) from `sacct_command` output.

    `CANCELLED by 1234` keeps only the state.
    """
    states = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) == 3 and fields[0] and fields[1].strip():
            states[fields[0]] = (fields[1].split()[0], fields[2])
    return states


# --- Accounting ----------------------------------------------------------------


def parse_tres_values(tres_str):
    """Parse a plain `cpu=8,mem=65536,gres/gpu=1` TRES list into a dict."""
    values = {}
    for item in tres_str.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        # TRES names contain '=' only as the final separator, but do contain
        # ':' (gres/gpu:nvidia_b200), so split from the right.
        name, _, raw = item.rpartition("=")
        try:
            values[name] = int(raw)
        except ValueError:
            continue
    return values


def parse_tres_limits(tres_str):
    """Parse `cpu=64(8),mem=N(65536)` into separate limit and usage dicts."""
    limits = {}
    usage = {}
    for item in tres_str.split(","):
        name, separator, raw = item.strip().rpartition("=")
        if not name or not separator:
            continue
        match = re.fullmatch(r"(N|\d+)\((\d+)\)", raw)
        if match:
            limits[name] = None if match.group(1) == "N" else int(match.group(1))
            usage[name] = int(match.group(2))
    return limits, usage


def parse_limit(text):
    """Turn a `16(5)` / `N(0)` limit token into its numeric limit or None."""
    match = re.match(r"(\S+?)\((\d+)\)", text)
    if not match:
        return None
    return None if match.group(1) == "N" else int(match.group(1))


def parse_default_account(output, user):
    """Find `user`'s default account in `scontrol show assoc_mgr` output."""
    fallback = None
    for block in re.split(r"\n(?=ClusterName=)", output):
        if f"UserName={user}(" not in block:
            continue
        match = re.search(r"Account=(\S+)", block)
        if not match:
            continue
        if "DefAssoc=Yes" in block:
            return match.group(1)
        fallback = fallback or match.group(1)
    return fallback


def parse_user_accounts(output, user):
    """Return all accounts associated with exactly `user` in the cache."""
    return sorted(
        {
            record["account"]
            for record in parse_associations(output)
            if record["user"] == user
        }
    )


def _strip_id(name):
    return re.sub(r"\(\d+\)$", "", name)


def parse_associations(output):
    """Parse the `Association Records` section of `scontrol show assoc_mgr`.

    Each record has its `account`, `user` ("" for an account's own
    association), `parent` account and `partition`, whether it is the user's
    default (`default`), the fair-share `fairshare` factor, and its limits:
    `max_jobs` / `max_submit_jobs` / `grp_jobs` / `grp_submit_jobs` (None when
    unset, usage in `job_usage`), `grp_tres` with usage in `grp_tres_usage`,
    `max_tres_pj`, `max_tres_pn` and `max_wall_pj` (minutes).
    """
    records = []
    record = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "QOS Records":
            break
        if line.startswith("ClusterName="):
            fields = dict(re.findall(r"(\S+?)=(\S*)", line))
            record = {
                "account": fields.get("Account", ""),
                "user": _strip_id(fields.get("UserName", "")),
                "partition": fields.get("Partition") or None,
                "parent": None,
                "default": False,
                "fairshare": None,
                "max_jobs": None,
                "max_submit_jobs": None,
                "grp_jobs": None,
                "grp_submit_jobs": None,
                "job_usage": {},
                "grp_tres": {},
                "grp_tres_usage": {},
                "max_tres_pj": {},
                "max_tres_pn": {},
                "max_wall_pj": None,
            }
            records.append(record)
            continue
        if record is None:
            continue

        match = re.search(r"SharesRaw/Norm/Level/Factor=(?:[^/\s]*/){3}(\S+)", line)
        if match:
            try:
                record["fairshare"] = float(match.group(1))
            except ValueError:
                pass
        match = re.search(r"\bParentAccount=(\S*)", line)
        if match:
            record["parent"] = _strip_id(match.group(1)) or None
        if re.search(r"\bDefAssoc=Yes\b", line):
            record["default"] = True
        for field, key, usage_key in (
            ("GrpJobs", "grp_jobs", "grp_running"),
            ("GrpSubmitJobs", "grp_submit_jobs", "grp_total"),
            ("MaxJobs", "max_jobs", "running"),
            ("MaxSubmitJobs", "max_submit_jobs", "total"),
        ):
            match = re.search(rf"\b{field}=(N|\d+)?(?:\((\d+)\))?(?=\s|$)", line)
            if match:
                limit = match.group(1)
                record[key] = int(limit) if limit and limit != "N" else None
                if match.group(2) is not None:
                    record["job_usage"][usage_key] = int(match.group(2))
        match = re.search(r"\bGrpTRES=(\S*)", line)
        if match:
            record["grp_tres"], record["grp_tres_usage"] = parse_tres_limits(
                match.group(1)
            )
        for field, key in (("MaxTRESPJ", "max_tres_pj"), ("MaxTRESPN", "max_tres_pn")):
            match = re.search(rf"\b{field}=(\S*)", line)
            if match:
                record[key] = parse_tres_values(match.group(1))
        match = re.search(r"\bMaxWallPJ=(\d+)", line)
        if match:
            record["max_wall_pj"] = int(match.group(1))
    return records


def parse_account_qos(output):
    """Parse `sacctmgr -nP ... format=Account,QOS,DefaultQOS` association rows.

    Returns `{account: {"qos": [names], "default_qos": name or None}}`.
    """
    accounts = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 2 or not fields[0].strip():
            continue
        account = fields[0].strip()
        entry = accounts.setdefault(account, {"qos": set(), "default_qos": None})
        entry["qos"].update(
            name.strip() for name in fields[1].split(",") if name.strip()
        )
        if len(fields) > 2 and fields[2].strip():
            entry["default_qos"] = fields[2].strip()
    return {
        account: {"qos": sorted(entry["qos"]), "default_qos": entry["default_qos"]}
        for account, entry in sorted(accounts.items())
    }


def parse_qos_records(output):
    """Parse the `QOS Records` section of `scontrol show assoc_mgr` output.

    Each QOS record contains per-job `max_tres_pj`, `accounts`, and
    `user_limits` keyed by user with `max_jobs`, `max_submit_jobs`, and
    per-user `max_tres_pu` caps. A limit of None means no QOS cap is set.
    Resource usage is kept separately in `user_tres_usage`, keyed by user,
    so borrowing another user's limits never borrows their usage. The parallel
    `account_limits`, `account_job_usage`, and `account_tres_usage` mappings
    hold per-account caps and each account's own usage. `max_tres_pn` holds
    per-node caps, `max_wall_pj` is a per-job time cap in minutes, and
    `priority` is the QOS priority.
    """
    records = {}
    record = None
    subsection = None
    user = None
    account = None
    in_qos_section = False

    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "QOS Records":
            in_qos_section = True
            continue
        if not in_qos_section:
            continue

        match = re.match(r"^QOS=(\S+?)\(\d+\)\s*$", line)
        if match:
            record = {
                "max_tres_pj": {},
                "max_tres_pn": {},
                "accounts": set(),
                "account_limits": {},
                "account_job_usage": {},
                "account_tres_usage": {},
                "user_limits": {},
                "user_tres_usage": {},
            }
            records[match.group(1)] = record
            subsection = None
            user = None
            account = None
            continue
        if record is None:
            continue

        if stripped == "Account Limits":
            subsection = "accounts"
            continue
        if stripped == "User Limits":
            subsection = "users"
            continue
        if line.startswith("    MaxTRESPJ="):
            record["max_tres_pj"] = parse_tres_values(stripped.partition("=")[2])
            continue
        if line.startswith("    MaxTRESPN="):
            record["max_tres_pn"] = parse_tres_values(stripped.partition("=")[2])
            continue
        if line.startswith("    MaxWallPJ="):
            value = stripped.partition("=")[2]
            if value.isdigit():
                record["max_wall_pj"] = int(value)
            elif value in ("", "N"):
                record["max_wall_pj"] = None
            continue
        if line.startswith("    Priority="):
            value = stripped.partition("=")[2]
            if value.isdigit():
                record["priority"] = int(value)
            continue

        indent = len(line) - len(line.lstrip())
        if subsection == "accounts" and indent == 6:
            account = stripped
            record["accounts"].add(account)
            record["account_limits"].setdefault(account, {})
        elif subsection == "accounts" and indent == 8 and account:
            limits = record["account_limits"][account]
            for field, key, usage_key in (
                ("MaxJobsPA", "max_jobs", "running"),
                ("MaxSubmitJobsPA", "max_submit_jobs", "total"),
            ):
                match = re.search(rf"\b{field}=(N|\d+)\((\d+)\)(?=\s|$)", line)
                if match:
                    limits[key] = None if match.group(1) == "N" else int(match.group(1))
                    record["account_job_usage"].setdefault(account, {})[usage_key] = (
                        int(match.group(2))
                    )
            match = re.search(r"\bMaxTRESPA=(\S*)", line)
            if match:
                tres_limits, usage = parse_tres_limits(match.group(1))
                limits["max_tres_pa"] = tres_limits
                record["account_tres_usage"][account] = usage
        elif subsection == "users" and indent == 6:
            user = _strip_id(stripped)
            record["user_limits"].setdefault(user, {})
        elif subsection == "users" and indent == 8 and user:
            for field, key in (
                ("MaxJobsPU", "max_jobs"),
                ("MaxSubmitJobsPU", "max_submit_jobs"),
            ):
                match = re.search(rf"\b{field}=(\S+)", line)
                if match:
                    record["user_limits"][user][key] = parse_limit(match.group(1))
            match = re.search(r"\bMaxTRESPU=(\S*)", line)
            if match:
                limits, usage = parse_tres_limits(match.group(1))
                record["user_limits"][user]["max_tres_pu"] = limits
                record["user_tres_usage"][user] = usage
    return records


def qos_user_limits(record, user):
    """Per-user job and resource limits for `record`.

    QOS per-user limits are a single value applied to every user, so when the
    caller has no entry yet (they have never had a job tracked under this QOS)
    any other user's entry carries the same limits.
    """
    limits = record["user_limits"].get(user)
    if limits:
        return limits
    for other in record["user_limits"].values():
        if other:
            return other
    return {}


def qos_account_limits(record, account):
    """Return QOS per-account caps, borrowing only caps from a tracked account."""
    limits = record["account_limits"].get(account)
    if limits:
        return limits
    for other in record["account_limits"].values():
        if other:
            return other
    return {}


def account_qos_command(user, cluster):
    """`sacctmgr` listing the user's accounts, QOS and default QOS."""
    return [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        "where",
        f"user={user}",
        f"cluster={cluster}",
        "format=Account,QOS%1000,DefaultQOS%100",
    ]


def assoc_mgr_command(accounts=None, qos=None):
    """`scontrol show assoc_mgr`, optionally only for some accounts and QOS.

    The unfiltered cache can be tens of MB on a large site; filtering to the
    caller's accounts and QOS keeps a refresh small. `users=` does not filter.
    """
    command = ["scontrol", "show", "assoc_mgr", "flags=assoc,qos"]
    if accounts:
        command.append(f"accounts={','.join(sorted(accounts))}")
    if qos:
        command.append(f"qos={','.join(sorted(qos))}")
    return command


# --- Scheduling probes ---------------------------------------------------------


def run_command(command, directives, shell=None):
    """The command that submits a request: an `sbatch` job holding the
    allocation with `SBATCH_WRAP`, or an interactive shell with `srun`.

    `test_command` probes the same request with `--test-only`.
    """
    if command == "srun":
        return ["srun", *directives, "--pty", shell or login_shell()]
    return ["sbatch", *directives, SBATCH_OUTPUT, f"--wrap={SBATCH_WRAP}"]


def format_run_command(command, directives, shell=None):
    """`run_command` as one line for copying."""
    if command == "srun":
        return shlex.join(run_command(command, directives, shell))
    # Double quotes are safe because SBATCH_WRAP has no shell metacharacters.
    sbatch = shlex.join(["sbatch", *directives, SBATCH_OUTPUT])
    return f'{sbatch} --wrap="{SBATCH_WRAP}"'


SUBMITTED_PATTERN = re.compile(r"\bSubmitted batch job (\d+)")


def parse_submitted(output):
    """The job id in sbatch's "Submitted batch job N", or None."""
    match = SUBMITTED_PATTERN.search(output)
    return match.group(1) if match else None


def test_command(command, directives, shell=None):
    """The `--test-only` probe for `directives` with sbatch or srun.

    Each names the program its `format_run_command` runs: sbatch wraps the
    same placeholder, so no batch script is needed, and srun runs the login
    shell. Some submit filters (Tamia's) only route an srun that names one.
    """
    if command == "srun":
        return ["srun", *directives, shell or login_shell()]
    return ["sbatch", *directives, f"--wrap={SBATCH_WRAP}"]


def format_test_command(command, directives, shell=None):
    """`test_command` as one line, quoted like `format_run_command`."""
    if command == "srun":
        return shlex.join(test_command(command, directives, shell))
    return f'{shlex.join(["sbatch", *directives])} --wrap="{SBATCH_WRAP}"'


def parse_test_output(output, returncode, command):
    """Read a `--test-only` reply.

    Returns `start_time` (None when the request was rejected), `partition`
    (as resolved by the scheduler) and the `result` text.
    """
    start_match = SBATCH_START_PATTERN.search(output)
    partition_match = SBATCH_PARTITION_PATTERN.search(output)
    partition = partition_match.group(1) if partition_match else None

    if start_match:
        return {
            "start_time": start_match.group(1),
            "partition": partition,
            "result": "Runnable",
        }

    result = " ".join(output.splitlines())
    if not result:
        result = f"{command} exited with status {returncode}"
    return {"start_time": None, "partition": partition, "result": result}
