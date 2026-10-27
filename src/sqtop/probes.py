"""Start-time estimates from `sbatch`/`srun --test-only` probes.

A `Grid` holds one probe per request size (CPU only, then 1..N GPUs of each
type) and walltime. Walltimes default to the distinct partition time limits,
which is where routing tiers change, and requests stop at the tightest GPU cap
in the account/QOS limits. No probe ever starts a job.
"""

import dataclasses
import datetime
import re
import shlex

from . import slurm, snapshot

# Probed when no partition sets a time limit and the profile names none.
DEFAULT_WALLTIMES = (60, 1440)
MAX_DERIVED_WALLTIMES = 6

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
REPLY_PREFIX = re.compile(r"^(?:sbatch|srun): (?:error: )?")
# Noise every site's replies share. Site banners are learned instead, from
# the lines accepted replies print too (`Grid.banner`).
GENERIC_NOISE = (
    r"\b(?:sbatch|srun): (?:error: )?",
    r"[-=]{8,}",
    r"allocation failure: Unspecified error",
)


@dataclasses.dataclass(frozen=True)
class Request:
    gpu: str | None = None
    count: int = 0
    # How to show the GPU type; the full type still goes into --gres.
    name: str | None = dataclasses.field(default=None, compare=False)
    # Ask for every CPU and all the memory of the node, as GPU requests do
    # where GPUs are only allocated by node (`gpu_counts = "whole_node"`).
    whole_node: bool = False

    @property
    def label(self):
        return f"{self.count}x {self.name or self.gpu}" if self.gpu else "CPU only"

    def directives(self):
        return [f"--gres=gpu:{self.gpu}:{self.count}"] if self.gpu else []


@dataclasses.dataclass(frozen=True)
class Settings:
    """What every probe asks for besides its GPUs, walltime and scope."""

    command: str = "sbatch"
    cpus: int = 4
    mem: str = "32G"
    # Extra options for every probe, e.g. ("--partition=x", "--constraint=y").
    extra: tuple[str, ...] = ()
    # Overrides the walltimes derived from partition limits, in minutes.
    walltimes: tuple[int, ...] = ()
    # Email for job notifications (`[mail]` in the config); None for none.
    mail_user: str | None = None
    mail_type: str = "BEGIN,END,FAIL"
    # Ask for whole nodes: every CPU, all memory and every GPU on the node.
    whole_node: bool = False

    def resource_directives(self, whole_node=False):
        """CPU and memory options: the request's, or a whole node's.

        `whole_node` asks for the whole node for one request even when the
        request itself does not.
        """
        if self.whole_node or whole_node:
            # --exclusive gives the job every CPU; --mem=0 all the memory.
            return ["--exclusive", "--mem=0"]
        # -c is --cpus-per-task, kept short for copying.
        return [f"-c{self.cpus}", f"--mem={self.mem}"]

    def describe(self):
        """The request in one line, e.g. `sbatch -c4 --mem=32G`."""
        return shlex.join([self.command, *self.resource_directives(), *self.extra])

    def mail_directives(self):
        """`--mail-type`/`--mail-user`, when notifications are on."""
        if not self.mail_user:
            return []
        return [f"--mail-type={self.mail_type}", f"--mail-user={self.mail_user}"]


@dataclasses.dataclass(frozen=True)
class Result:
    start: datetime.datetime | None
    partition: str | None
    message: str
    # The command to submit the request: no --test-only, with any email;
    # `run` is the same command as arguments, which Submit runs.
    command: str
    run: tuple[str, ...] = ()
    # The probe as it ran (minimal: no email) and its reply, which a
    # rejection's message is cleaned from again once more banner is known.
    probe: str = ""
    output: str = dataclasses.field(default="", repr=False)

    @property
    def ok(self):
        return self.start is not None


@dataclasses.dataclass
class Grid:
    scope: snapshot.Scope
    settings: Settings
    requests: list
    walltimes: list
    # Walltimes above this cap (minutes) are not probed.
    max_wall: int | None = None
    # The cluster user's login shell, for `srun --pty` run commands.
    shell: str = "bash"
    results: dict = dataclasses.field(default_factory=dict)
    # Lines accepted replies print too: a site's banner or notes on every
    # submission (RCL's storage policy, Fir's memory units), which
    # `clean_message` leaves out of reasons. Grids of one cluster may share it.
    banner: set = dataclasses.field(default_factory=set)

    @property
    def gpus_by_node(self):
        """Whether GPU rows ask for whole nodes though the request does not."""
        return not self.settings.whole_node and any(
            request.whole_node for request in self.requests
        )

    def skipped(self, column):
        return self.max_wall is not None and self.walltimes[column] > self.max_wall

    def cells(self):
        return [
            (row, column)
            for row in range(len(self.requests))
            for column in range(len(self.walltimes))
            if not self.skipped(column)
        ]

    def request_directives(self, row, column):
        """The cell's request: account/QOS, GPUs, CPUs and memory, walltime."""
        request = self.requests[row]
        return [
            *self.scope.directives(),
            *request.directives(),
            *self.settings.resource_directives(request.whole_node),
            # -t is --time, kept short for copying.
            f"-t{slurm.format_duration(self.walltimes[column])}",
        ]

    def probe_directives(self, row, column):
        """The probe: the request and extra options under --test-only.

        Only what the estimate depends on, so no email options.
        """
        return [
            "--test-only",
            *self.request_directives(row, column),
            *self.settings.extra,
        ]

    def run_directives(self, row, column):
        """The same request to submit, with the email options."""
        return [
            *self.request_directives(row, column),
            *self.settings.mail_directives(),
            *self.settings.extra,
        ]


def parse_walltime(text):
    """Minutes in a compact (`90m`, `3h`, `1d12h`) or Slurm time string."""
    text = text.strip().lower()
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?", text)
    if text and match:
        days, hours, minutes = (int(group or 0) for group in match.groups())
        return days * 1440 + hours * 60 + minutes or None
    return slurm.parse_duration(text)


def parse_walltimes(text):
    """Sorted distinct minutes from a comma- or space-separated list."""
    values = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        minutes = parse_walltime(part)
        if not minutes:
            raise ValueError(f"invalid walltime {part!r}")
        values.add(minutes)
    return tuple(sorted(values))


def format_wait(start, now):
    """How long until `start`: "now" within a minute, else `D-HH:MM:SS`.

    Rounded up to the minute, so the wait never reads shorter than it is.
    """
    seconds = (start - now).total_seconds()
    if seconds <= 60:
        return "now"
    return slurm.format_duration(int(-(-seconds // 60)))


def counts_for(ceiling, capacity, mode):
    """GPU counts worth probing: powers of two to the ceiling, all, or whole node."""
    if mode == "whole_node":
        return [capacity]
    if mode == "all":
        return list(range(1, ceiling + 1))
    counts = []
    count = 1
    while count < ceiling:
        counts.append(count)
        count *= 2
    return [*counts, ceiling]


def max_gpu_counts(capacities, limits):
    """Return the largest count of each GPU type worth probing in one job.

    The tightest cap on the model or on all GPUs combined, at any scope (per
    job, user, account or node), bounds the count along with one node's
    capacity. A zero cap still probes one GPU to surface the reason.
    """
    counts = {}
    for gpu, capacity in capacities.items():
        caps = [
            cap
            for key in (f"tres:gres/gpu:{gpu}", "tres:gres/gpu")
            if (cap := snapshot.tightest(limits, key)) is not None
        ]
        counts[gpu] = max(1, min([capacity, *caps]))
    return counts


def plan_requests(capacities, ceilings, mode="pow2", names=None):
    """Requests for each GPU type up to its ceiling, then CPU only.

    In `whole_node` mode each GPU request asks for its whole node: all of its
    GPUs, CPUs and memory.
    """
    names = names or {}
    whole_node = mode == "whole_node"
    requests = [
        Request(gpu, count, names.get(gpu), whole_node)
        for gpu, capacity in capacities.items()
        for count in counts_for(ceilings.get(gpu, capacity), capacity, mode)
    ]
    requests.append(Request())
    return requests


def plan_walltimes(partitions, override=()):
    """Walltimes to probe: the override, or the partitions' time tiers.

    Distinct time limits of the partitions the caller can use mark where
    routing changes; more than `MAX_DERIVED_WALLTIMES` are thinned evenly,
    keeping the shortest and longest. Preemptible partitions (such as Fir's
    122-day `cpupreempt`) are not routing tiers, so they are left out.
    """
    if override:
        return sorted(set(override))
    tiers = sorted(
        {
            partition["max_time"]
            for partition in partitions
            if partition["max_time"]
            and partition.get("usable", True)
            and partition.get("preempt_mode", "OFF") == "OFF"
        }
    )
    if not tiers:
        return list(DEFAULT_WALLTIMES)
    if len(tiers) <= MAX_DERIVED_WALLTIMES:
        return tiers
    last = len(tiers) - 1
    picks = {
        round(index * last / (MAX_DERIVED_WALLTIMES - 1))
        for index in range(MAX_DERIVED_WALLTIMES)
    }
    return [tiers[index] for index in sorted(picks)]


def usable_nodes(snap):
    """Nodes in partitions the caller can use, or every node if none are known."""
    usable = {partition["name"] for partition in snap.partitions if partition["usable"]}
    nodes = [node for node in snap.nodes if usable.intersection(node["partitions"])]
    return nodes or snap.nodes


def build_grid(snap, scope, settings, profile, info):
    """Size the grid for `scope` on the cluster `info` describes."""
    limits = snap.limits(scope, info.user)
    capacities = snapshot.gpu_capacities(usable_nodes(snap))
    usable = [partition for partition in snap.partitions if partition["usable"]]
    return Grid(
        scope=scope,
        settings=settings,
        requests=plan_requests(
            capacities,
            max_gpu_counts(capacities, limits),
            # A whole node means all of its GPUs, whatever the profile probes.
            "whole_node" if settings.whole_node else profile.gpu_counts,
            snap.gpu_names,
        ),
        walltimes=plan_walltimes(usable or snap.partitions, settings.walltimes),
        max_wall=snapshot.tightest(limits, "wall"),
        shell=info.shell,
    )


def reply_lines(text):
    """A reply's lines, without colour codes, command prefixes or padding."""
    lines = (
        REPLY_PREFIX.sub("", line.strip()).strip()
        for line in ANSI_ESCAPE.sub("", text).splitlines()
    )
    return [line for line in lines if line]


def clean_message(text, banner=frozenset()):
    """The reason in a rejection, without colour codes, prefixes or banners.

    Lines in `banner` were printed by an accepted reply too, so they cannot
    be the reason.
    """
    text = " ".join(line for line in reply_lines(text) if line not in banner)
    for pattern in GENERIC_NOISE:
        text = re.sub(pattern, " ", text)
    return " ".join(text.split()).strip(" .") or "rejected without a reason"


def learn_banner(grid, output):
    """Add an accepted reply's lines to the banner and reclean the reasons.

    Rejections that landed first get their banner lines removed now.
    """
    lines = {
        line
        for line in reply_lines(output)
        if not slurm.SBATCH_START_PATTERN.search(line)
    }
    if lines <= grid.banner:
        return
    grid.banner |= lines
    for cell, result in grid.results.items():
        if not result.ok and result.output:
            message = clean_message(result.output, grid.banner)
            grid.results[cell] = dataclasses.replace(result, message=message)


def make_result(grid, row, column, returncode, output):
    """Turn one probe's reply into a `Result` with its probe and run commands."""
    command = grid.settings.command
    output = output.strip()
    raw = slurm.parse_test_output(output, returncode, command)
    start = raw["start_time"]
    run = slurm.run_command(command, grid.run_directives(row, column), grid.shell)
    return Result(
        start=datetime.datetime.fromisoformat(start) if start else None,
        partition=raw["partition"],
        message="" if start else clean_message(output or raw["result"], grid.banner),
        command=slurm.format_run_command(
            command, grid.run_directives(row, column), grid.shell
        ),
        run=tuple(run),
        probe=slurm.format_test_command(
            command, grid.probe_directives(row, column), grid.shell
        ),
        output=output,
    )


def run_grid(grid, host, on_result=None, should_stop=lambda: False):
    """Run every probe in `grid` on `host`, reporting each result as it lands.

    `should_stop` is polled between results; once it returns True, probes not
    yet started are dropped. Raises `hosts.ConnectionLost` if ssh fails.
    """
    cells = grid.cells()
    commands = [
        slurm.test_command(
            grid.settings.command, grid.probe_directives(row, column), grid.shell
        )
        for row, column in cells
    ]

    def landed(index, returncode, output):
        row, column = cells[index]
        result = make_result(grid, row, column, returncode, output)
        if result.ok:
            learn_banner(grid, output)
        grid.results[row, column] = result
        if on_result is not None:
            on_result(row, column, result)

    host.run_stream(commands, landed, should_stop)
    return grid
