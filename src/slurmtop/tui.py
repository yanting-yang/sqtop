"""Interactive dashboard: partitions, account/QOS limits, start estimates, jobs.

Each cluster gets a tab holding a `ClusterView`. A cluster is contacted only
while its tab is shown: at startup that is the local one, and a remote cluster
is first reached over ssh (see `hosts`) when its tab is opened. The visible
view collects data on a background thread every `REFRESH_SECONDS`, and probes
run on their own worker, filling the estimates grid as replies arrive.

Everything is a click; slurmtop binds no keys. The action bar at the bottom
refreshes, re-probes, edits the request, turns the hourly email (`mailer`) of
the clusters shown so far on or off (it stops when the dashboard exits), opens
the config form (`ConfigScreen`, which saves the config file and applies it)
and quits.
"""

import dataclasses
import datetime
import re
import shlex

from rich.cells import cell_len
from rich.console import Group
from rich.measure import Measurement
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import get_current_worker

from . import hosts, mailer, probes, profiles, slurm, snapshot, views
from .profiles import Profile

REFRESH_SECONDS = 60
# A tab shown again refreshes once its data is this old.
STALE_AFTER = datetime.timedelta(seconds=REFRESH_SECONDS)
# Re-probe on a refresh once the estimates are this old.
PROBE_MAX_AGE = datetime.timedelta(minutes=5)
# Below this width the 2x2 grid cannot fit D-HH:MM:SS columns, so panels stack.
NARROW_WIDTH = 135

CELL_STYLES = {
    "pending": "dim",
    "skipped": "dim",
    "blocked": "red",
    "now": "bold green",
    "hours": "green",
    "days": "yellow",
    "weeks": "dark_orange",
}
JOB_STATE_STYLES = {"RUNNING": "green", "PENDING": "yellow"}


def load_bar(summary, width=8):
    """`███▒▒░░` bar of allocated, unavailable and free capacity, with a %."""
    allocated, free, total = views.main_resource(summary)
    if not total:
        return Text("-", style="dim")
    unavailable = total - allocated - free
    cells = [round(width * value / total) for value in (allocated, unavailable)]
    cells.append(max(0, width - sum(cells)))
    percent = views.load_percent(summary)
    style = "red" if percent >= 90 else "yellow" if percent >= 70 else "green"
    bar = Text()
    bar.append("█" * cells[0], style=style)
    bar.append("▒" * cells[1], style="grey42")
    bar.append("░" * cells[2], style="grey35")
    bar.append(f" {percent:>3}%")
    return bar


class CommandLine:
    """A command line that wraps between arguments, never inside a quoted one.

    Rich wraps at any space, which would split `--wrap="sleep infinity"`.
    """

    WORD = re.compile(r"""(?:[^\s"']+|"[^"]*"|'[^']*')+""")

    GAP = 2

    def __init__(self, text, style="bold", suffix=None):
        self.words = self.WORD.findall(text)
        self.style = style
        # Text kept whole after the last word, such as a Copy link.
        self.suffix = suffix

    def __rich_console__(self, console, options):
        line = ""
        for word in self.words:
            if line and cell_len(line) + 1 + cell_len(word) > options.max_width:
                yield Text(line, self.style)
                line = word
            else:
                line = f"{line} {word}" if line else word
        last = Text(line, self.style)
        if self.suffix is not None:
            if cell_len(line) + self.GAP + self.suffix.cell_len > options.max_width:
                yield last
                last = Text()
            else:
                last.append(" " * self.GAP)
            last.append_text(self.suffix)
        yield last

    def __rich_measure__(self, console, options):
        longest = max((cell_len(word) for word in self.words), default=0)
        whole = cell_len(" ".join(self.words))
        if self.suffix is not None:
            longest = max(longest, self.suffix.cell_len)
            whole += self.GAP + self.suffix.cell_len
        return Measurement(longest, whole).clamp(max_width=options.max_width)


def copy_link(command):
    """A Copy link: a click copies `command` (`SlurmtopApp.action_copy_command`)."""
    meta = {"@click": ("app.copy_command", (command,))}
    return Text("Copy", Style(color="cyan", underline=True, meta=meta))


class Dashboard(Grid):
    """The four panels: a 2x2 grid, or a scrolling stack when narrow."""

    def on_resize(self, event):
        self.set_class(event.size.width < NARROW_WIDTH, "narrow")


class RequestScreen(ModalScreen):
    """Edit the resources every probe asks for."""

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        settings = self.settings
        walltimes = ", ".join(slurm.format_duration(m) for m in settings.walltimes)
        with Vertical(id="request-dialog"):
            yield Label("Probe request", classes="dialog-title")
            yield Checkbox(
                "Whole node (--exclusive --mem=0, all GPUs)",
                settings.whole_node,
                id="whole-node",
            )
            yield Label("CPUs per task")
            yield Input(
                str(settings.cpus),
                id="cpus",
                type="integer",
                disabled=settings.whole_node,
            )
            yield Label("Memory")
            yield Input(
                settings.mem, id="mem", placeholder="32G", disabled=settings.whole_node
            )
            yield Label("Walltimes (blank: one per partition time limit)")
            yield Input(walltimes, id="walltimes", placeholder="0-03:00:00, 1-00:00:00")
            yield Label("Extra options for every probe")
            yield Input(
                shlex.join(settings.extra),
                id="extra",
                placeholder="--partition=name --constraint=feature",
            )
            yield Label(f"Email on {settings.mail_type} (blank: no email)")
            yield Input(
                settings.mail_user or "", id="mail", placeholder="you@example.com"
            )
            yield Label("", id="request-error")
            with Horizontal(id="request-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Probe", id="apply", variant="primary")

    @on(Button.Pressed, "#cancel")
    def cancel(self):
        self.dismiss(None)

    @on(Checkbox.Changed, "#whole-node")
    def whole_node_changed(self, event):
        # A whole node brings its own CPUs and memory.
        for field in ("#cpus", "#mem"):
            self.query_one(field, Input).disabled = event.value

    @on(Button.Pressed, "#apply")
    def apply(self):
        try:
            cpus = int(self.query_one("#cpus", Input).value)
            if cpus < 1:
                raise ValueError("CPUs must be at least 1")
            mail = self.query_one("#mail", Input).value.strip()
            if mail and ("@" not in mail or " " in mail):
                raise ValueError("Email must be an address, or blank for none")
            settings = dataclasses.replace(
                self.settings,
                cpus=cpus,
                mem=self.query_one("#mem", Input).value.strip() or "32G",
                extra=tuple(shlex.split(self.query_one("#extra", Input).value)),
                walltimes=probes.parse_walltimes(
                    self.query_one("#walltimes", Input).value
                ),
                mail_user=mail or None,
                whole_node=self.query_one("#whole-node", Checkbox).value,
            )
        except ValueError as error:
            self.query_one("#request-error", Label).update(Text(str(error), "red"))
            return
        self.dismiss(settings)


class ClusterRow(Horizontal):
    """One cluster's profile in the config form: name, rules and ssh host."""

    def __init__(self, cluster="", settings=None):
        super().__init__()
        self.cluster = cluster
        self.settings = settings or {}

    def compose(self) -> ComposeResult:
        settings = self.settings
        yield Input(self.cluster, classes="name", placeholder="ClusterName")
        # Only the modes the loader accepts can be picked.
        yield Select(
            [(mode, mode) for mode in profiles.GPU_COUNT_MODES],
            value=settings.get("gpu_counts", profiles.Profile.gpu_counts),
            allow_blank=False,
            compact=True,
            classes="gpu-counts",
        )
        yield Input(str(settings.get("ssh", "")), classes="ssh", placeholder="host")
        yield Button("Remove", classes="remove")

    def field(self, name):
        return self.query_one(f".{name}", Input).value.strip()

    @property
    def gpu_counts(self):
        return self.query_one(".gpu-counts", Select).value

    def values(self):
        """(name, settings) from the row, or None for an empty one.

        The default GPU counts are left out of the file.
        """
        name, ssh, gpu_counts = self.field("name"), self.field("ssh"), self.gpu_counts
        default = gpu_counts == profiles.Profile.gpu_counts
        if not (name or ssh) and default:
            return None
        settings = {}
        if not default:
            settings["gpu_counts"] = gpu_counts
        if ssh:
            settings["ssh"] = ssh
        return name, settings


class ConfigScreen(ModalScreen):
    """Edit the config in a form and save it; dismisses with True once saved.

    The form starts from the settings in use (none without a config) and
    checks what it saves with the loaders' own rules, so a saved file always
    loads.
    """

    MAIL_FIELDS = ("user", "type")
    SMTP_FIELDS = ("server", "port", "sender", "password", "recipient")

    def __init__(self, config, path, note=None):
        super().__init__()
        self.config = config
        self.path = path
        self.note = note

    def compose(self) -> ComposeResult:
        mail = self.config.get("mail") or {}
        smtp = self.config.get("smtp") or {}
        # Labels are Text: Rich would read "[mail]" as markup.
        # The scrolling dialog must not take focus itself, or the first Enter
        # and Tab would go to it instead of the first field.
        with VerticalScroll(id="config-dialog", can_focus=False):
            yield Label("Config", classes="dialog-title")
            yield Label(Text(f"Saved to {self.path}", style="dim"), id="config-path")
            yield Label(
                Text("Job emails [mail] (blank address: none)"), classes="section"
            )
            yield from self.fields(
                "mail",
                {
                    "user": ("Address", mail.get("user", ""), "you@example.org"),
                    "type": ("On", mail.get("type", "BEGIN,END,FAIL"), ""),
                },
            )
            yield Label(
                Text("Hourly email [smtp] (blank server: none)"), classes="section"
            )
            yield from self.fields(
                "smtp",
                {
                    "server": ("Server", smtp.get("server", ""), "mail.example.org"),
                    "port": ("Port", smtp.get("port", ""), "587 (STARTTLS), 465 (SSL)"),
                    "sender": ("Sender", smtp.get("sender", ""), "also the login"),
                    "password": ("Password", smtp.get("password", ""), ""),
                    "recipient": (
                        "Recipient",
                        smtp.get("recipient", ""),
                        "the address above",
                    ),
                },
            )
            yield Label(Text("Clusters [clusters.<name>]"), classes="section")
            with Horizontal(classes="cluster-heading"):
                yield Label("Name", classes="name")
                yield Label("GPU counts", classes="gpu-counts")
                yield Label("ssh host", classes="ssh")
            with Vertical(id="clusters"):
                for name, settings in (self.config.get("clusters") or {}).items():
                    yield ClusterRow(name, settings)
            yield Label("", id="config-error")
            with Horizontal(id="config-buttons"):
                yield Button("Add cluster", id="config-add")
                yield Button("Cancel", id="config-cancel")
                yield Button("Save", id="config-save", variant="primary")

    @staticmethod
    def fields(table, fields):
        for name, (label, value, placeholder) in fields.items():
            with Horizontal(classes="field"):
                yield Label(label)
                yield Input(
                    str(value),
                    id=f"{table}-{name}",
                    placeholder=placeholder,
                    password=name == "password",
                )

    def on_mount(self):
        if self.note:
            self.query_one("#config-error", Label).update(Text(self.note, "red"))

    @on(Button.Pressed, "#config-cancel")
    def cancel(self):
        self.dismiss(False)

    @on(Button.Pressed, "#config-add")
    async def add_cluster(self):
        row = ClusterRow()
        await self.query_one("#clusters").mount(row)
        row.query_one(".name", Input).focus()

    @on(Button.Pressed, ".remove")
    def remove_cluster(self, event):
        event.button.parent.remove()

    def field(self, table, name):
        return self.query_one(f"#{table}-{name}", Input).value.strip()

    def values(self):
        """The config the form describes, as `load_config` would return it."""
        config = {}
        mail = {name: self.field("mail", name) for name in self.MAIL_FIELDS}
        if mail["user"]:
            config["mail"] = {k: v for k, v in mail.items() if v}
        smtp = {name: self.field("smtp", name) for name in self.SMTP_FIELDS}
        if any(smtp.values()):
            if smtp["port"].isdigit():
                smtp["port"] = int(smtp["port"])
            config["smtp"] = {k: v for k, v in smtp.items() if v != ""}
        clusters = {}
        for row in self.query(ClusterRow):
            values = row.values()
            if values is None:
                continue
            name, settings = values
            if name in clusters:
                raise ValueError(f"cluster {name!r} is listed twice")
            clusters[name] = settings
        if clusters:
            config["clusters"] = clusters
        return config

    @on(Button.Pressed, "#config-save")
    def save(self):
        try:
            config = self.values()
            profiles.check_config(config)
            profiles.save_config(config)
        except (ValueError, OSError) as error:
            self.query_one("#config-error", Label).update(Text(str(error), "red"))
            return
        self.dismiss(True)


class ConfirmScreen(ModalScreen):
    """Ask before a command that acts on the cluster (submitting or releasing a
    job): the exact command and what happens next. Dismisses with True to go
    ahead."""

    # Cancel has focus, so going ahead always takes a deliberate click.
    AUTO_FOCUS = "#confirm-cancel"

    def __init__(self, question, command, note, verb):
        super().__init__()
        self.question = question
        self.command = command
        self.note = note
        self.verb = verb

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.question, classes="dialog-title")
            yield Static(CommandLine(self.command), id="confirm-command")
            yield Static(Text(self.note, style="dim"))
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="confirm-cancel")
                yield Button(self.verb, id="confirm-ok", variant="warning")

    @on(Button.Pressed, "#confirm-cancel")
    def cancel(self):
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-ok")
    def confirm(self):
        self.dismiss(True)


class ClusterView(Vertical):
    """One cluster's dashboard: identity line, four panels and details.

    The view owns its cluster's state (snapshot, probe request, grid), so
    each tab keeps its own selection and request. The app's action bar runs
    its actions on the visible tab.
    """

    def __init__(self, cluster, settings, pane_id, **kwargs):
        super().__init__(**kwargs)
        self.cluster = cluster
        # The id of the tab pane holding this view (`SlurmtopApp.view`).
        self.pane_id = pane_id
        self.host = cluster.host
        self.profile = cluster.profile
        self.settings = settings
        self.info = None
        self.collector = None
        self.snap = None
        self.scope = None
        self.grid = None
        # Banner lines learned from accepted probes, kept across re-probes.
        self.banner = set()
        self.probed_at = None
        self.error = None
        self.detail_from = "partitions"
        # True from starting a collection until its outcome is shown. Kept
        # here rather than read from the worker, which is still winding down
        # when its outcome arrives, so a refresh asked for then is not dropped.
        self.collecting = False

    # --- Layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(self.identity(), id="identity")
            yield Static("loading…", id="status")
        # Partitions span the top; estimates sit below on the left, beside
        # accounts and jobs (the grid fills cells in this order).
        with Dashboard(id="dashboard"):
            with Container(id="left-top"):
                # Clicking either title switches to the other (`app.toggle_nodes`).
                toggle = Style(underline=True, meta={"@click": "app.toggle_nodes"})
                yield self.table("partitions", Text("Partitions", toggle), "row")
                yield self.table("hardware", Text("Nodes by hardware", toggle), "row")
            # Titled with the probed account/QOS once there is one.
            yield self.table("estimates", None, "cell")
            yield self.table("scopes", "Accounts & QOS", "row")
            yield self.table("jobs", "My jobs", "row")
        yield Static(id="detail")

    @staticmethod
    def table(table_id, title, cursor):
        table = DataTable(id=table_id, cursor_type=cursor, zebra_stripes=True)
        table.border_title = title
        return table

    def identity(self):
        text = Text()
        info = self.info
        if info is None:
            text.append(self.cluster.name, style="bold cyan")
        else:
            text.append(f"{info.user}@{info.hostname}", style="bold")
            text.append(f"  {info.name or self.cluster.name}", style="bold cyan")
            text.append(f"  Slurm {info.version}", style="dim")
        if self.host.ssh:
            text.append(f"  via ssh {self.host.ssh}", style="dim")
        if self.profile != Profile(name=self.profile.name, ssh=self.profile.ssh):
            text.append(f"  profile {self.profile.name}", style="dim")
        return text

    def on_mount(self):
        self.tables = {table.id: table for table in self.query(DataTable)}
        self.status = self.query_one("#status", Static)
        self.detail = self.query_one("#detail", Static)
        self.tables["hardware"].display = False
        self.detail.border_title = "Details"
        self.tables["partitions"].add_columns(
            "Partition",
            "Use",
            "CPU/node",
            "Mem/node",
            "GPU/node",
            "Load",
            "Free",
            "Idle",
            "Pend",
            "Max",
        )
        self.tables["hardware"].add_columns(
            "Hardware", "Nodes", "States", "CPUs free", "Mem free (GB)", "GPUs free"
        )
        self.tables["scopes"].add_columns(
            "Account", "QOS", "Jobs", "GPUs", "Wall", "FS"
        )
        self.tables["jobs"].add_columns(
            "Job", "Name", "St", "Partition", "Start", "Time", "Left", "Nodes", "Reason"
        )
        # Nothing is fetched until the tab is opened (`activate`).
        if self.host.ssh:
            self.status.update("not checked yet")
            self.detail.update(
                Text(
                    f"{self.cluster.name} is checked over ssh ({self.host.ssh}) "
                    "when you open this tab.",
                    style="dim",
                )
            )

    def activate(self):
        """The tab was opened: fetch what is missing or old, then take focus."""
        # The pane is shown on the next refresh; a hidden table cannot take focus.
        self.app.call_after_refresh(self.focus_main)
        if self.stale or self.error is not None:
            self.refresh_snapshot()
        elif self.probes_stale:
            self.start_probes()

    def focus_main(self):
        """Focus the panel the details pane follows."""
        table = self.tables.get(self.detail_from)
        if table is not None and table.display:
            table.focus()

    def loaded(self):
        """Clear the loading state; tables that were loading could not take focus."""
        for table in self.tables.values():
            table.loading = False
        if self.app.view is self and not self.has_focus_within:
            self.focus_main()

    # --- Data ----------------------------------------------------------------

    @property
    def stale(self):
        return self.snap is None or (
            datetime.datetime.now() - self.snap.taken_at >= STALE_AFTER
        )

    @property
    def probes_stale(self):
        return self.probed_at is None or (
            datetime.datetime.now() - self.probed_at > PROBE_MAX_AGE
        )

    def refresh_snapshot(self):
        """Collect in the background, unless a collection is still running."""
        if self.collecting:
            return
        self.collecting = True
        if self.snap is None:
            self.status.update("loading…")
            for table in self.tables.values():
                table.loading = True
        self.collect()

    @work(thread=True, group="snapshot")
    def collect(self):
        try:
            if self.collector is None:
                info = snapshot.ClusterInfo.detect(self.host)
                self.collector = snapshot.Collector(info, self.host)
                self.app.call_from_thread(self.detected, info)
            snap = self.collector.collect()
        except hosts.ConnectionLost as error:
            self.app.call_from_thread(self.disconnected, str(error))
            return
        except Exception as error:  # Keep the dashboard up; say what broke.
            self.app.call_from_thread(self.collect_failed, error)
            return
        self.app.call_from_thread(self.apply_snapshot, snap)

    def collect_failed(self, error):
        self.collecting = False
        self.loaded()
        self.notify(f"{self.cluster.name}: refresh failed: {error!r}", severity="error")

    def detected(self, info):
        self.info = info
        self.query_one("#identity", Static).update(self.identity())

    def disconnected(self, message):
        """Show why ssh failed; the last data, if any, stays on screen."""
        self.collecting = False
        self.error = message
        self.loaded()
        self.update_status()
        self.show_detail()

    def apply_snapshot(self, snap):
        first = self.snap is None
        self.collecting = False
        self.snap = snap
        self.error = None
        self.fill_partitions()
        self.fill_hardware()
        self.fill_jobs()
        if self.scope not in snap.scopes:
            self.scope = snap.preferred_scope()
        self.fill_scopes()
        if self.probes_stale:
            self.start_probes()
        elif self.grid is not None:
            self.fill_estimates()
        self.loaded()
        if first:
            for note in snap.notes:
                self.notify(
                    f"{self.cluster.name}: {note}", severity="warning", timeout=10
                )
        self.show_detail()
        self.update_status()

    def update_status(self):
        if self.error is not None:
            status = Text("not connected", style="bold red")
            if self.snap is not None:
                status.append(f" · last data {self.snap.taken_at:%H:%M:%S}", "dim")
        elif self.snap is None:
            return
        else:
            age = int((datetime.datetime.now() - self.snap.taken_at).total_seconds())
            status = Text(f"updated {self.snap.taken_at:%H:%M:%S} ({age}s ago)")
            if self.probed_at is not None:
                status.append(f" · probed {self.probed_at:%H:%M:%S}")
        status.append_text(self.app.mail_status())
        self.status.update(status)

    # --- Tables --------------------------------------------------------------

    @staticmethod
    def refill(table, rows):
        """Replace the rows of `table`, keeping the cursor on the same row."""
        row = table.cursor_row
        table.clear()
        for key, cells in rows:
            table.add_row(*cells, key=key)
        if table.row_count:
            table.move_cursor(row=min(row, table.row_count - 1), scroll=False)

    def fill_partitions(self):
        rows = []
        for summary in self.snap.partitions:
            usable = Text("yes", "green") if summary["usable"] else Text("no", "red")
            pending = summary["pending"]
            rows.append(
                (
                    summary["name"],
                    (
                        Text(summary["name"] + ("*" if summary["default"] else "")),
                        usable,
                        views.node_cpus(summary),
                        views.node_mem(summary),
                        views.node_gpus(summary, self.snap.gpu_names),
                        load_bar(summary),
                        views.free_resources(summary, self.snap.gpu_names),
                        views.idle_nodes(summary),
                        "?" if pending is None else str(pending),
                        views.max_time(summary),
                    ),
                )
            )
        self.refill(self.tables["partitions"], rows)

    def fill_hardware(self):
        rows = []
        for index, group in enumerate(self.snap.hardware):
            names = self.snap.gpu_names
            gpus = ", ".join(
                f"{counts['free']}/{counts['total']} {names.get(gpu, gpu)}"
                for gpu, counts in group["gpus"].items()
            )
            mem = group["mem"]
            rows.append(
                (
                    str(index),
                    (
                        group["label"],
                        str(group["node_count"]),
                        views.state_counts(group["states"]),
                        f"{group['cpus']['free']}/{group['cpus']['total']}",
                        f"{views.gb(mem['free'])}/{views.gb(mem['total'])}",
                        gpus or "-",
                    ),
                )
            )
        self.refill(self.tables["hardware"], rows)

    def fill_scopes(self):
        rows = []
        for index, scope in enumerate(self.snap.scopes):
            summary = views.scope_summary(self.snap, scope, self.info.user)
            # The pair being probed stands out; the others are plain.
            style = "bold cyan" if scope == self.scope else ""
            rows.append(
                (
                    str(index),
                    (
                        Text(summary["account"], style),
                        Text(summary["qos"], style),
                        summary["jobs"],
                        summary["gpus"],
                        summary["wall"],
                        summary["fairshare"],
                    ),
                )
            )
        table = self.tables["scopes"]
        self.refill(table, rows)
        table.border_subtitle = "* left out of commands · click a row twice: probe"

    def fill_jobs(self):
        table = self.tables["jobs"]
        jobs = self.snap.jobs
        if not jobs:
            message = "squeue failed" if jobs is None else "No jobs in the queue"
            self.refill(table, [("none", (Text(message, "dim"), *[""] * 8))])
            return

        def clip(text, width=16):
            return text if len(text) <= width else text[: width - 1] + "…"

        rows = []
        for job in jobs:
            state = job["state"]
            elapsed = (
                slurm.normalize_time(job["elapsed"]) if state != "PENDING" else "-"
            )
            rows.append(
                (
                    job["id"],
                    (
                        job["id"],
                        clip(job["name"]),
                        Text(views.short_state(state), JOB_STATE_STYLES.get(state, "")),
                        # A pending job may list several; details show them all.
                        clip(job["partition"]),
                        views.job_start_time(job, compact=True),
                        elapsed,
                        self.time_left_cell(job),
                        self.job_nodes_cell(job),
                        views.job_reason(job),
                    ),
                )
            )
        self.refill(table, rows)

    @staticmethod
    def time_left_cell(job):
        """Time left; a running job near its limit stands out, a pending
        job's whole limit is dimmed."""
        text = views.job_time_left(job)
        if job["state"] != "RUNNING":
            return Text(text, "dim")
        minutes = slurm.parse_duration(job.get("time_left", ""))
        if minutes is not None and minutes < 10:
            return Text(text, "bold red")
        if minutes is not None and minutes < 60:
            return Text(text, "yellow")
        return Text(text)

    @staticmethod
    def job_nodes_cell(job):
        """Where the job runs; nodes only planned for it are dimmed."""
        nodes, planned = views.job_nodes(job)
        return Text(nodes, "dim italic" if planned else "")

    def fill_estimates(self):
        table = self.tables["estimates"]
        grid = self.grid
        cursor = table.cursor_coordinate
        table.clear(columns=True)
        table.fixed_columns = 1
        table.add_column("Request", key="request")
        for column, minutes in enumerate(grid.walltimes):
            # Wide enough for any wait ("365-08:37:00") before results arrive.
            table.add_column(slurm.format_duration(minutes), key=f"w{column}", width=12)
        now = datetime.datetime.now()
        for row, request in enumerate(grid.requests):
            cells = [
                self.estimate_cell(row, column, now)
                for column in range(len(grid.walltimes))
            ]
            table.add_row(request.label, *cells, key=f"r{row}")
        if table.row_count:
            table.move_cursor(
                row=min(cursor.row, table.row_count - 1),
                column=max(1, min(cursor.column, len(grid.walltimes))),
                scroll=False,
            )
        settings = grid.settings
        # Clicking the account/QOS probes the next one, when there is another.
        title = Text(grid.scope.label)
        if self.snap is not None and len(self.snap.scopes) > 1:
            title.stylize(Style(underline=True, meta={"@click": "app.next_scope"}))
        table.border_title = title
        notes = ["whole node"] if settings.whole_node else []
        if grid.gpus_by_node:
            notes.append("GPUs by whole node")
        # Clicking the command switches sbatch/srun.
        toggle = Style(underline=True, meta={"@click": "app.toggle_command"})
        subtitle = Text()
        subtitle.append(settings.command, toggle)
        subtitle.append(settings.describe().removeprefix(settings.command))
        for note in notes:
            subtitle.append(f" · {note}")
        table.border_subtitle = subtitle

    def estimate_cell(self, row, column, now):
        text, kind = views.cell(self.grid, row, column, now)
        style = CELL_STYLES[kind]
        if column in views.earliest_columns(self.grid, row):
            style += " underline"
        return Text(text, style, justify="right")

    # --- Probes --------------------------------------------------------------

    def start_probes(self):
        if self.snap is None:
            return
        self.grid = probes.build_grid(
            self.snap, self.scope, self.settings, self.profile, self.info
        )
        self.grid.banner = self.banner
        self.probed_at = datetime.datetime.now()
        self.fill_estimates()
        self.fill_scopes()
        self.run_probes(self.grid)

    @work(thread=True, exclusive=True, group="probes")
    def run_probes(self, grid):
        worker = get_current_worker()

        def report(row, column, result):
            self.app.call_from_thread(self.probe_landed, grid, row, column)

        try:
            probes.run_grid(grid, self.host, report, lambda: worker.is_cancelled)
        except hosts.ConnectionLost as error:
            self.app.call_from_thread(self.disconnected, str(error))
        except Exception as error:  # Keep the dashboard up; say what broke.
            self.app.call_from_thread(
                self.notify,
                f"{self.cluster.name}: probes failed: {error!r}",
                severity="error",
            )

    def probe_landed(self, grid, row, column):
        if grid is not self.grid:
            return
        table = self.tables["estimates"]
        now = datetime.datetime.now()
        # The earliest start in the row may move, so restyle the whole row.
        for index in range(len(grid.walltimes)):
            table.update_cell(
                f"r{row}", f"w{index}", self.estimate_cell(row, index, now)
            )
        selected = table.cursor_coordinate == (row, column + 1)
        if self.detail_from == "estimates" and selected:
            self.show_detail()

    # --- Details -------------------------------------------------------------

    def on_descendant_focus(self, event):
        if isinstance(event.widget, DataTable):
            self.detail_from = event.widget.id
            self.show_detail()

    @on(DataTable.RowHighlighted)
    @on(DataTable.CellHighlighted)
    def highlighted(self, event):
        if event.data_table.id == self.detail_from:
            self.show_detail()

    # A click on the highlighted row selects it (the first click highlights).
    @on(DataTable.RowSelected, "#scopes")
    def scope_selected(self, event):
        index = int(event.row_key.value)
        self.select_scope(self.snap.scopes[index])

    def show_detail(self):
        detail = self.detail
        if self.error is not None:
            detail.update(self.connection_detail())
            return
        if self.snap is None:
            return
        source = self.detail_from
        table = self.tables[source]
        if not table.row_count:
            detail.update(self.notes_text())
            return
        row = table.cursor_row
        if source == "partitions":
            detail.update(self.partition_detail(self.snap.partitions[row]))
        elif source == "hardware":
            detail.update(self.hardware_detail(self.snap.hardware[row]))
        elif source == "scopes":
            detail.update(self.scope_detail(self.snap.scopes[row]))
        elif source == "estimates" and self.grid is not None:
            detail.update(self.estimate_detail(row, table.cursor_column - 1))
        elif source == "jobs" and self.snap.jobs:
            detail.update(self.job_detail(self.snap.jobs[row]))
        else:
            detail.update(self.notes_text())

    def connection_detail(self):
        title = Text(f"Cannot reach {self.cluster.name} over ssh ", style="bold red")
        title.append(f"({self.host.ssh})", style="red")
        return Group(
            title,
            Text(self.error),
            Text(
                f"{hosts.connect_hint(self.host.ssh)} Then click Refresh.", style="dim"
            ),
        )

    def notes_text(self):
        notes = self.snap.notes if self.snap else []
        return Text("\n".join(notes) or "", style="yellow")

    def partition_detail(self, summary):
        cpus = summary["cpus"]
        mem = summary["mem"]

        def allow(names):
            return "ALL" if names is None else ", ".join(sorted(names))

        header = Text()
        header.append(summary["name"], style="bold")
        header.append(f"  {summary['state']}")
        if summary["usable"]:
            header.append("  usable", style="green")
        else:
            header.append(f"  not usable: {summary['access_note']}", style="red")
        header.append(f"  · max time {views.max_time(summary)}")
        if summary["default_time"]:
            default = slurm.format_duration(summary["default_time"])
            header.append(f" · default {default}")
        if summary["qos"]:
            header.append(f" · partition QOS {summary['qos']}")
        header.append(f" · priority tier {summary['priority_tier']}")
        if summary.get("preempt_mode", "OFF") != "OFF":
            header.append(f" · preemptible ({summary['preempt_mode']})", "yellow")
        pending = "?" if summary["pending"] is None else summary["pending"]
        lines = [
            header,
            Text(
                f"Nodes {summary['node_count']} "
                f"({views.state_counts(summary['states'])}) · pending jobs {pending}"
            ),
            Text(" · ".join(views.node_kinds(summary, self.snap.gpu_names))),
            Text(
                f"CPUs {cpus['free']} free / {cpus['allocated']} allocated / "
                f"{cpus.get('unavailable', 0)} unavailable of {cpus['total']} · "
                f"Memory {views.gb(mem['free'])} GB free of {views.gb(mem['total'])} GB"
            ),
        ]
        for gpu, counts in summary["gpus"].items():
            lines.append(
                Text(
                    f"GPUs {self.snap.gpu_names.get(gpu, gpu)}: {counts['free']} free / {counts['allocated']} "
                    f"allocated / {counts.get('unavailable', 0)} unavailable "
                    f"of {counts['total']}"
                )
            )
        lines.append(
            Text(
                f"Allowed accounts {allow(summary['allow_accounts'])} · QOS "
                f"{allow(summary['allow_qos'])} · groups "
                f"{allow(summary['allow_groups'])}",
                style="dim",
            )
        )
        return Group(*lines)

    def hardware_detail(self, group):
        names = ", ".join(group["names"][:40])
        if len(group["names"]) > 40:
            names += f", … ({len(group['names'])} nodes)"
        return Group(
            Text(group["label"], style="bold"),
            Text(views.state_counts(group["states"])),
            Text(names, style="dim"),
        )

    def scope_detail(self, scope):
        table = Table(box=None, pad_edge=False, expand=False, header_style="bold")
        for column in ("Scope", "Limit", "Max", "Used"):
            table.add_column(
                column, justify="right" if column in ("Max", "Used") else "left"
            )
        for row in views.limit_rows(self.snap.limits(scope, self.info.user)):
            table.add_row(*row)
        summary = views.scope_summary(self.snap, scope, self.info.user)
        caption = Text(f"{scope.label}", style="bold")
        caption.append(
            f"  your jobs: {summary['jobs']} running/cap, {summary['pending']} pending"
        )
        flags = " ".join(scope.directives()) or "no --account/--qos needed"
        caption.append(f" · {flags} · ∞ no cap · ? unknown", style="dim")
        if not table.row_count:
            return Group(caption, Text("No limits found for this QOS.", "dim"))
        return Group(caption, table)

    def estimate_detail(self, row, column):
        grid = self.grid
        if column < 0:
            return Text(
                "Rows are request sizes; columns are walltimes. Cells show how long "
                "until the scheduler expects the job to start.",
                style="dim",
            )
        request = grid.requests[row]
        walltime = slurm.format_duration(grid.walltimes[column])
        title = Text(f"{request.label} for {walltime}", style="bold")
        title.append(f"  {grid.scope.label}", style="dim")
        if grid.skipped(column):
            cap = slurm.format_duration(grid.max_wall)
            return Group(
                title,
                Text(
                    f"Not probed: over the {cap} wall-time cap of this account/QOS.",
                    "dim",
                ),
            )
        result = grid.results.get((row, column))
        if result is None:
            return Group(title, Text("Probing…", "dim"))
        lines = [title]
        if result.ok:
            now = datetime.datetime.now()
            line = Text("Estimated start ", style="green")
            line.append(f"{result.start:%Y-%m-%d %H:%M}", style="bold green")
            line.append(f" ({probes.format_wait(result.start, now)})")
            if result.partition:
                line.append(f" in partition {result.partition}")
            lines.append(line)
        else:
            lines.append(Text(f"Cannot run: {result.message}", style="red"))
        lines.append(Text())
        lines.append(self.commands_detail(result))
        return Group(*lines)

    def commands_detail(self, result):
        """The probe as it ran with its reply, then the command to submit.

        Labels sit in their own column, so wrapped commands stay aligned.
        """
        prompt = f"{self.host.ssh}$" if self.host.ssh else "$"
        output = probes.ANSI_ESCAPE.sub("", result.output) or "(no output)"
        batch = result.run[0] == "sbatch"
        hints = []
        if self.host.ssh:
            hints.append(f"run it on {self.cluster.name}")
        if not batch:
            # An interactive shell needs a terminal of its own.
            hints.append("run it in a terminal for a shell on the allocation")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        probe = CommandLine(f"{prompt} {result.probe}", suffix=copy_link(result.probe))
        table.add_row("Probe", probe)
        table.add_row("", Text(output, style="dim"))
        table.add_row("", "")
        # Clicking the label runs the app's action for the visible tab; only
        # sbatch jobs are submitted from here.
        submit = Text("Submit")
        if batch:
            submit.stylize(Style(underline=True, meta={"@click": "app.submit"}))
        command = CommandLine(
            f"{prompt} {result.command}", suffix=copy_link(result.command)
        )
        table.add_row(submit, command)
        if hints:
            table.add_row("", Text(" · ".join(hints), style="dim"))
        return table

    def job_detail(self, job):
        state = Text(job["state"], JOB_STATE_STYLES.get(job["state"], ""))
        reason = views.job_reason(job)
        header = Text.assemble(
            (f"{job['id']} ", "bold"),
            (job["name"], "bold"),
            "  ",
            state,
            f" ({reason})" if reason != "-" else "",
        )
        nodes, planned = views.job_nodes(job)
        where = f"planned on {nodes}" if planned else f"on {nodes}"
        started = "starts" if job["state"] == "PENDING" else "started"
        return Group(
            header,
            Text(
                f"{job['account']}/{job['qos']} in {job['partition']} · "
                f"submitted {job['submit']} · {started} {views.job_start_time(job)} "
                f"{where}"
            ),
            Text(
                f"{job['cpus']} CPUs · {job['mem']} · {job['nodes']} node(s) · "
                f"{job['tres'] if job['tres'] != 'N/A' else 'no GRES'} · "
                f"elapsed {slurm.normalize_time(job['elapsed'])} of "
                f"{slurm.normalize_time(job['time_limit'])}, "
                f"{views.job_time_left(job)} left"
            ),
            *self.job_commands(job),
        )

    def job_commands(self, job):
        """Commands for a queued job: a shell attached to it, and releasing it."""
        if job["state"] not in ("RUNNING", "PENDING"):
            return []
        prompt = f"{self.host.ssh}$" if self.host.ssh else "$"
        shell = self.info.shell if self.info else slurm.login_shell()
        attach = shlex.join(
            ["srun", f"--jobid={job['id']}", "--overlap", "--pty", shell]
        )
        suffix = copy_link(attach)
        if job["state"] == "PENDING":
            suffix = Text.assemble(("(once it starts)  ", "dim"), suffix)
        release = f"scancel {job['id']}"
        # Clicking Release cancels the job once a dialog confirms it.
        label = Text("Release")
        label.stylize(
            Style(underline=True, meta={"@click": ("app.release_job", (job["id"],))})
        )
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("Attach", CommandLine(f"{prompt} {attach}", suffix=suffix))
        table.add_row(
            label, CommandLine(f"{prompt} {release}", suffix=copy_link(release))
        )
        return [Text(), table]

    # --- Actions -------------------------------------------------------------

    def action_refresh(self):
        self.probed_at = None
        self.refresh_snapshot()

    def action_probe(self):
        self.start_probes()

    def select_scope(self, scope):
        self.scope = scope
        self.start_probes()
        self.notify(f"Probing {self.cluster.name} as {scope.label}")

    def action_next_scope(self):
        """Probe with the next account/QOS (a click on the estimates' title)."""
        scopes = self.snap.scopes if self.snap else []
        if len(scopes) < 2:
            self.notify("No other account/QOS to probe with.")
            return
        index = scopes.index(self.scope) if self.scope in scopes else -1
        self.select_scope(scopes[(index + 1) % len(scopes)])

    def action_toggle_command(self):
        command = "srun" if self.settings.command == "sbatch" else "sbatch"
        self.settings = dataclasses.replace(self.settings, command=command)
        self.start_probes()

    def action_edit_request(self):
        def done(settings):
            if settings is not None:
                self.set_settings(settings)

        self.app.push_screen(RequestScreen(self.settings), done)

    def set_settings(self, settings):
        """Probe with a new request (edited, or with a reloaded `[mail]`)."""
        self.settings = settings
        self.start_probes()

    def set_profile(self, profile):
        """Use a reloaded profile; True if its rules changed (then re-probed)."""
        changed = profile.gpu_counts != self.profile.gpu_counts
        self.profile = self.cluster.profile = profile
        if changed:
            self.query_one("#identity", Static).update(self.identity())
            self.start_probes()
        return changed

    def selected_result(self):
        """The probe result under the estimates cursor, or None."""
        table = self.tables["estimates"]
        if self.grid is None or table.cursor_column < 1:
            return None
        return self.grid.results.get((table.cursor_row, table.cursor_column - 1))

    def action_submit(self):
        """Submit the selected estimate's request once the dialog confirms it."""
        result = self.selected_result()
        if result is None:
            self.notify("Select a probed estimate first.")
            return
        if not result.ok:
            self.notify(
                f"Slurm would refuse this request: {result.message}",
                severity="warning",
            )
            return
        if result.run[0] != "sbatch":
            self.notify("Only sbatch jobs are submitted from here; click Copy instead.")
            return
        prompt = f"{self.host.ssh}$" if self.host.ssh else "$"
        note = (
            "The job holds the allocation with 'sleep infinity' until its time "
            "limit; release it early with scancel <id>."
        )

        def done(confirmed):
            if confirmed:
                self.submit_batch(result.run)

        self.app.push_screen(
            ConfirmScreen(
                f"Submit this job on {self.cluster.name}?",
                f"{prompt} {result.command}",
                note,
                "Submit",
            ),
            done,
        )

    @work(thread=True, group="submit")
    def submit_batch(self, args):
        try:
            code, output = self.host.act(list(args))
        except hosts.ConnectionLost as error:
            self.app.call_from_thread(self.disconnected, str(error))
            return
        self.app.call_from_thread(self.submitted, code, output)

    def submitted(self, code, output):
        job = slurm.parse_submitted(output)
        if code != 0 or job is None:
            reason = probes.clean_message(
                output or f"sbatch exited with status {code}", self.banner
            )
            self.notify(f"sbatch failed: {reason}", severity="error", timeout=15)
            return
        self.notify(
            f"Batch job {job} on {self.cluster.name}; scancel {job} releases it.",
            title="Submitted",
            timeout=10,
        )
        self.refresh_snapshot()

    def release_job(self, job_id):
        """Release a job with scancel once the dialog confirms it."""
        jobs = (self.snap.jobs if self.snap else None) or []
        job = next((job for job in jobs if job["id"] == job_id), None)
        if job is None:
            self.notify(f"Job {job_id} is no longer in the queue.")
            return
        prompt = f"{self.host.ssh}$" if self.host.ssh else "$"
        if job["state"] == "RUNNING":
            note = "Whatever runs in it is killed and its allocation is released."
        else:
            note = "It leaves the queue without running."

        def done(confirmed):
            if confirmed:
                self.cancel_job(job_id)

        self.app.push_screen(
            ConfirmScreen(
                f"Release job {job_id} ({job['name']}) on {self.cluster.name}?",
                f"{prompt} scancel {job_id}",
                note,
                "Release",
            ),
            done,
        )

    @work(thread=True, group="release")
    def cancel_job(self, job_id):
        try:
            code, output = self.host.act(["scancel", job_id])
        except hosts.ConnectionLost as error:
            self.app.call_from_thread(self.disconnected, str(error))
            return
        self.app.call_from_thread(self.released, job_id, code, output)

    def released(self, job_id, code, output):
        if code != 0:
            reason = probes.clean_message(
                output or f"scancel exited with status {code}"
            )
            self.notify(f"scancel failed: {reason}", severity="error", timeout=15)
            return
        self.notify(f"Released job {job_id} on {self.cluster.name}.", title="Released")
        self.refresh_snapshot()

    def action_toggle_nodes(self):
        """Show Nodes by hardware in place of Partitions, or back."""
        partitions = self.tables["partitions"]
        hardware = self.tables["hardware"]
        showing = partitions.display
        partitions.display = not showing
        hardware.display = showing
        (hardware if showing else partitions).focus()

class SlurmtopApp(App):
    TITLE = "slurmtop"
    # Every control is a click (the action bar, titles and links); the command
    # palette would only add themes.
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    TabbedContent, ContentSwitcher, TabPane, ClusterView { height: 1fr; }
    TabPane { padding: 0; }
    #topbar { height: 1; background: $panel; padding: 0 1; }
    #actions { dock: bottom; height: 1; background: $panel; padding: 0 1; }
    #identity { width: 1fr; }
    #status { width: auto; color: $text-muted; }
    #dashboard {
        grid-size: 2 3;
        grid-columns: 3fr 2fr;
        grid-rows: 3fr 2fr 2fr;
        height: 1fr;
    }
    #left-top { column-span: 2; }
    #estimates { row-span: 2; }
    #dashboard.narrow {
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: auto;
        overflow-y: auto;
    }
    #dashboard.narrow #left-top { column-span: 1; }
    #dashboard.narrow #estimates { row-span: 1; }
    #dashboard.narrow #left-top { height: auto; }
    #dashboard.narrow DataTable { height: auto; max-height: 16; }
    #left-top { height: 1fr; }
    DataTable {
        height: 1fr;
        border: round $border-blurred;
        border-title-color: $text-muted;
        scrollbar-size: 1 1;
    }
    DataTable:focus { border: round $border; border-title-color: $text; }
    #detail {
        height: 12;
        border: round $border-blurred;
        border-title-color: $text-muted;
        padding: 0 1;
        overflow-y: auto;
    }
    RequestScreen { align: center middle; }
    #request-dialog {
        width: 64;
        height: auto;
        border: round $border;
        background: $surface;
        padding: 1 2;
    }
    .dialog-title { text-style: bold; margin-bottom: 1; }
    #whole-node { margin-bottom: 1; }
    #request-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #request-buttons Button { margin-left: 1; }
    ConfirmScreen { align: center middle; }
    #confirm-dialog {
        width: 100;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    #confirm-command { margin-bottom: 1; }
    #confirm-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #confirm-buttons Button { margin-left: 1; }
    ConfigScreen { align: center middle; }
    #config-dialog {
        width: 110;
        height: auto;
        max-height: 90%;
        border: round $border;
        background: $surface;
        padding: 1 2;
    }
    #config-dialog .section { margin-top: 1; text-style: bold; }
    #config-dialog Input { height: 1; border: none; padding: 0 1; background: $boost; }
    #config-dialog Input:focus { background: $primary 40%; }
    #config-dialog .field { height: 1; }
    #config-dialog .field Label { width: 12; }
    #config-dialog .cluster-heading { height: 1; color: $text-muted; }
    #config-dialog .cluster-heading Label { padding: 0 1; }
    #clusters { height: auto; }
    ClusterRow { height: 1; }
    ClusterRow Input, .cluster-heading Label { width: 1fr; margin-right: 1; }
    ClusterRow .name, .cluster-heading .name { width: 20; }
    ClusterRow .gpu-counts, .cluster-heading .gpu-counts { width: 14; margin-right: 1; }
    ClusterRow Select > SelectCurrent { background: $boost; }
    ClusterRow Button { height: 1; min-width: 8; border: none; margin: 0; }
    #config-error { margin-top: 1; }
    #config-buttons { height: auto; align-horizontal: right; margin-top: 1; }
    #config-buttons Button { margin-left: 1; }
    #config-add { margin-right: 4; }
    """

    def __init__(self, clusters, settings):
        super().__init__()
        self.clusters = clusters
        self.settings = settings
        self.theme = "tokyo-night"
        # Tab panes are cluster-<n>; n is never reused when tabs come and go.
        self.pane_count = len(clusters)
        # The hourly email, while it is on: its server, its timer, when the
        # next one goes out and why the last one failed (None if it did not).
        self.smtp = None
        self.mail_timer = None
        self.mail_next = None
        self.mail_error = None

    def compose(self) -> ComposeResult:
        with TabbedContent(id="clusters"):
            for index, cluster in enumerate(self.clusters):
                with TabPane(cluster.name, id=f"cluster-{index}"):
                    yield ClusterView(cluster, self.settings, f"cluster-{index}")
        yield Static(self.actions(), id="actions")

    def on_mount(self):
        # App.query_one searches the screen on top, which is the request dialog
        # while it is open, so keep direct references to the dashboard widgets.
        self.tabs = self.query_one(TabbedContent)
        self.action_bar = self.query_one("#actions", Static)
        self.views = list(self.query(ClusterView))
        self.set_interval(REFRESH_SECONDS, self.refresh_visible)
        self.set_interval(1, self.update_status)
        self.view.activate()

    @property
    def view(self):
        """The cluster view on the visible tab."""
        active = self.tabs.active
        return next((v for v in self.views if v.pane_id == active), self.views[0])

    def refresh_visible(self):
        """Only the visible cluster refreshes, probing when its estimates are old.

        Hidden tabs are left alone; opening one refreshes it if needed.
        """
        if self.view.snap is not None:
            self.view.refresh_snapshot()

    def update_status(self):
        self.view.update_status()

    @on(TabbedContent.TabActivated)
    def tab_activated(self, event):
        # The first tab is activated in on_mount, once the views are known.
        if getattr(self, "views", None):
            self.view.activate()

    def actions(self):
        """The action bar: a click on a label runs its action, which acts on
        the visible tab for the first three. The email label says whether the
        hourly email is on, and a click flips it."""
        on = self.mail_timer is not None
        controls = [
            ("Refresh", "app.refresh", None),
            ("Re-probe", "app.probe", None),
            ("Edit request", "app.edit_request", None),
            (f"Email: {'on' if on else 'off'}", ("app.set_mail", (not on,)), on),
            ("Config", "app.edit_config", None),
            ("Quit", "app.quit", None),
        ]
        text = Text()
        for label, action, highlight in controls:
            if text:
                text.append("   ")
            style = Style(underline=True, meta={"@click": action})
            if highlight:
                style += Style(color="green")
            text.append(label, style)
        return text

    def update_actions(self):
        self.action_bar.update(self.actions())

    def action_refresh(self):
        self.view.action_refresh()

    def action_probe(self):
        self.view.action_probe()

    def action_edit_request(self):
        self.view.action_edit_request()

    def action_submit(self):
        """Submit from the visible tab, as its Submit label does when clicked."""
        self.view.action_submit()

    def action_toggle_command(self):
        """Switch the visible tab's probes between sbatch and srun (a click on
        the command under Start estimates)."""
        self.view.action_toggle_command()

    def action_toggle_nodes(self):
        """Switch the visible tab between Partitions and Nodes by hardware (a
        click on either title)."""
        self.view.action_toggle_nodes()

    def action_next_scope(self):
        """Probe the visible tab with its next account/QOS (a click on the
        estimates' title)."""
        self.view.action_next_scope()

    def action_copy_command(self, command):
        """Copy a command from the details (a click on its Copy)."""
        self.copy_to_clipboard(command)
        self.notify(command, title="Copied")

    def action_release_job(self, job_id):
        """Release a job of the visible tab (a click on Release in its details)."""
        self.view.release_job(job_id)

    def action_set_mail(self, on):
        """Start the hourly email, reading `[smtp]` from the config, or stop it."""
        if not on:
            self.stop_mail()
            return
        try:
            self.smtp = profiles.load_smtp()
        except ValueError as error:
            self.notify(f"config: {error}", severity="error")
            return
        if self.smtp is None:
            self.notify(
                "Add an [smtp] table to your config (Config) to email hourly.",
                severity="warning",
            )
            return
        self.mail_timer = self.set_interval(mailer.INTERVAL, self.send_mail)
        self.update_actions()
        self.notify(f"Emailing {self.smtp.recipient} now and hourly until you quit.")
        self.send_mail()

    def send_mail(self):
        """Email the clusters shown so far: this one and any remote tab opened.

        Hidden tabs that were never opened are not contacted.
        """
        interval = datetime.timedelta(seconds=mailer.INTERVAL)
        self.mail_next = datetime.datetime.now() + interval
        targets = [
            (view.cluster, view.settings)
            for view in self.views
            if view.host.ssh is None or view.snap is not None
        ]
        self.mail_report(self.smtp, targets)
        self.update_status()

    @work(thread=True, exclusive=True, group="mail")
    def mail_report(self, smtp, targets):
        try:
            mailer.send_report(targets, smtp)
        except OSError as error:
            self.call_from_thread(self.mailed, smtp, str(error) or repr(error))
        else:
            self.call_from_thread(self.mailed, smtp, None)

    def mailed(self, smtp, error):
        if self.mail_timer is None:
            return
        self.mail_error = error
        if error is None:
            self.notify(f"Emailed {smtp.recipient}.")
        else:
            self.notify(f"Could not email {smtp.recipient}: {error}", severity="error")
        self.update_status()

    def stop_mail(self):
        self.mail_timer.stop()
        self.mail_timer = self.mail_next = self.mail_error = None
        self.update_actions()
        self.notify("Hourly email off.")
        self.update_status()

    def mail_status(self):
        """The status line's note on the hourly email; empty while it is off."""
        if self.mail_next is None:
            return Text()
        note = Text(f" · next email {self.mail_next:%H:%M}")
        if self.mail_error is not None:
            note.append(" (last failed)", style="red")
        return note

    def action_edit_config(self):
        """Open the config form on the settings in use; saving applies them."""
        shown = profiles.short_path(profiles.CONFIG_PATH)
        note = None
        try:
            config = profiles.load_config()
        except ValueError as error:  # the file broke since the start
            config = {}
            note = f"{shown}: {error}; saving replaces it"

        def done(saved):
            if saved:
                self.reload_config()

        self.push_screen(ConfigScreen(config, shown, note), done)

    def reload_config(self):
        """Apply the saved config without restarting.

        `[mail]` reaches every tab's request when it changed, the hourly
        email's server changes while it is on, profiles change in place and
        clusters added to or removed from the config gain or lose tabs.
        """
        shown = profiles.short_path(profiles.CONFIG_PATH)
        local = next((view for view in self.views if view.host.ssh is None), None)
        try:
            mail = profiles.load_mail()
            smtp = profiles.load_smtp()
            local_profile = local and profiles.load_profile(local.cluster.name)
            remotes = [
                profile
                for profile in profiles.remote_profiles()
                if local is None or profile.name != local.cluster.name
            ]
        except ValueError as error:  # the form checks first, so: changed since
            self.notify(f"{shown}: {error}", severity="error", timeout=15)
            return
        changes = []

        # Only a changed [mail] table reaches the tabs, so an address edited
        # for the session (Edit request) survives other edits.
        mail_fields = {
            "mail_user": mail and mail[0],
            "mail_type": mail[1] if mail else probes.Settings().mail_type,
        }
        settings = dataclasses.replace(self.settings, **mail_fields)
        if settings != self.settings:
            self.settings = settings
            for view in self.views:
                view.set_settings(dataclasses.replace(view.settings, **mail_fields))
            changes.append(f"job emails to {mail[0]}" if mail else "job emails off")

        if self.mail_timer is not None and smtp != self.smtp:
            if smtp is None:
                self.stop_mail()
                changes.append("hourly email off")
            else:
                self.smtp = smtp
                changes.append(f"hourly email through {smtp.server}")

        if local is not None and local.set_profile(local_profile):
            changes.append(f"{local.cluster.name} profile")
        wanted = {profile.name: profile for profile in remotes}
        gone = []
        for view in self.views:
            if view.host.ssh is None:
                continue
            profile = wanted.get(view.cluster.name)
            if profile is not None and profile.ssh == view.host.ssh:
                if view.set_profile(wanted.pop(view.cluster.name)):
                    changes.append(f"{view.cluster.name} profile")
            else:
                gone.append(view)
        for profile in wanted.values():
            self.add_cluster(profile)
            changes.append(f"{profile.name} tab added")
        for view in gone:
            if len(self.views) == 1:
                changes.append(f"{view.cluster.name} tab kept, being the only one")
                continue
            self.remove_view(view)
            changes.append(f"{view.cluster.name} tab removed")

        note = f"Saved {shown}. "
        note += f"Reloaded: {', '.join(changes)}." if changes else "Nothing changed."
        self.notify(note, title="Config", timeout=10)

    def add_cluster(self, profile):
        """Open a tab for a cluster added to the config; it connects when opened."""
        cluster = snapshot.Cluster(profile.name, profile, hosts.Remote(profile.ssh))
        pane_id = f"cluster-{self.pane_count}"
        self.pane_count += 1
        view = ClusterView(cluster, self.settings, pane_id)
        self.clusters.append(cluster)
        self.views.append(view)
        self.tabs.add_pane(TabPane(cluster.name, view, id=pane_id))

    def remove_view(self, view):
        """Close a cluster's tab, showing the first tab if it was the visible one.

        Removing the view cancels its workers.
        """
        self.clusters.remove(view.cluster)
        self.views.remove(view)
        if self.tabs.active == view.pane_id:
            # A tab added in the same reload may not be mounted yet.
            mounted = [v for v in self.views if v.parent is not None]
            self.tabs.active = (mounted or self.views)[0].pane_id
        self.tabs.remove_pane(view.pane_id)
