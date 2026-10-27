"""Plain-text report: the TUI's panels printed once, for pipes and scripts.

Everything prints to `out` (stdout by default), so the dashboard's hourly
email can collect the same text.
"""

import datetime

from . import hosts, probes, slurm, snapshot, views
from .snapshot import Scope

# Between the reports of two clusters.
SEPARATOR = "=" * 80


def print_table(columns, rows, out=None):
    widths = [
        max([len(columns[index]), *(len(str(row[index])) for row in rows)])
        for index in range(len(columns))
    ]
    separator = " | "
    header = separator.join(
        column.ljust(width) for column, width in zip(columns, widths)
    )
    print(header, file=out)
    print("-" * len(header), file=out)
    for row in rows:
        print(
            separator.join(
                str(value).ljust(width) for value, width in zip(row, widths)
            ),
            file=out,
        )
    print(file=out)


def print_partitions(snap, out=None):
    rows = [
        (
            summary["name"] + ("*" if summary["default"] else ""),
            "yes" if summary["usable"] else f"no ({summary['access_note']})",
            views.node_cpus(summary),
            views.node_mem(summary),
            views.node_gpus(summary, snap.gpu_names),
            f"{views.load_percent(summary)}%",
            views.free_resources(summary, snap.gpu_names),
            views.idle_nodes(summary),
            "?" if summary["pending"] is None else summary["pending"],
            views.max_time(summary),
        )
        for summary in snap.partitions
    ]
    print("Partitions (* default):", file=out)
    if not rows:
        print("none reported\n", file=out)
        return
    print_table(
        [
            "Partition",
            "Usable",
            "CPU/node",
            "Mem/node",
            "GPU/node",
            "Load",
            "Free",
            "Idle nodes",
            "Pending",
            "Max time",
        ],
        rows,
        out=out,
    )


def print_hardware(snap, out=None):
    rows = [
        (
            group["label"],
            group["node_count"],
            views.state_counts(group["states"]),
            f"{group['cpus']['free']}/{group['cpus']['total']}",
            f"{views.gb(group['mem']['free'])}/{views.gb(group['mem']['total'])}",
            ", ".join(
                f"{counts['free']}/{counts['total']} {snap.gpu_names.get(gpu, gpu)}"
                for gpu, counts in group["gpus"].items()
            )
            or "-",
        )
        for group in snap.hardware
    ]
    print("Nodes by hardware (free/total; CPUs exclude reserved cores):", file=out)
    print_table(
        ["Hardware", "Nodes", "States", "CPUs free", "Mem free (GB)", "GPUs free"],
        rows,
        out=out,
    )


def print_scopes(snap, user, out=None):
    print("Accounts & QOS (* used when omitted; ∞ no cap, ? unknown):", file=out)
    if not snap.scopes:
        print("none found\n", file=out)
        return
    rows = []
    for scope in snap.scopes:
        summary = views.scope_summary(snap, scope, user)
        rows.append(
            (
                summary["account"],
                summary["qos"],
                summary["jobs"],
                summary["pending"],
                summary["gpus"],
                summary["wall"],
                summary["fairshare"],
            )
        )
    print_table(
        ["Account", "QOS", "Running/cap", "Pending", "GPU cap", "Wall cap", "FS"],
        rows,
        out=out,
    )
    for scope in snap.scopes:
        rows = views.limit_rows(snap.limits(scope, user))
        print(f"Limits for {scope.label}:", file=out)
        if rows:
            print_table(["Scope", "Limit", "Max", "Used"], rows, out=out)
        else:
            print("none found\n", file=out)


def print_grid(grid, now, out=None):
    settings = grid.settings
    request = settings.describe()
    if settings.whole_node:
        request += ", whole node"
    if grid.gpus_by_node:
        request += ", GPU rows by whole node"
    if settings.mail_user:
        request += f", email {settings.mail_user} on {settings.mail_type}"
    print(
        f"Start estimates for {grid.scope.label} ({request}), "
        f"waits from {now:%Y-%m-%d %H:%M}:",
        file=out,
    )
    rows = [
        (
            request.label,
            *(
                views.cell(grid, row, column, now)[0]
                for column in range(len(grid.walltimes))
            ),
        )
        for row, request in enumerate(grid.requests)
    ]
    print_table(
        ["Request", *(slurm.format_duration(m) for m in grid.walltimes)], rows, out=out
    )
    if grid.max_wall is not None and any(
        grid.skipped(column) for column in range(len(grid.walltimes))
    ):
        print(
            f"'-' is over the {slurm.format_duration(grid.max_wall)} wall-time cap; "
            "not probed.",
            file=out,
        )

    print_reasons(grid, out=out)

    example = next(
        (result for _, result in sorted(grid.results.items()) if result.ok), None
    )
    if example is not None:
        print(f"Probe, e.g.: {example.probe}", file=out)
        print(f"Submit it with: {example.command}", file=out)
        differ = "Other cells differ only in --gres and -t"
        if grid.gpus_by_node:
            differ += "; GPU rows ask for --exclusive --mem=0 instead of -c/--mem"
        print(f"{differ}.", file=out)
    print(file=out)


def print_reasons(grid, out=None):
    """List each rejection reason once, with the requests it applies to.

    A request rejected at every probed walltime is named alone; otherwise
    the walltimes follow it.
    """
    probed = [c for c in range(len(grid.walltimes)) if not grid.skipped(c)]
    reasons = {}
    for (row, column), result in sorted(grid.results.items()):
        if not result.ok:
            reasons.setdefault(result.message, {}).setdefault(row, []).append(column)
    if not reasons:
        return
    print("Cannot run:", file=out)
    for message, rows in reasons.items():
        labels = []
        for row, columns in rows.items():
            label = grid.requests[row].label
            if columns != probed:
                walltimes = ", ".join(
                    slurm.format_duration(grid.walltimes[c]) for c in columns
                )
                label += f" at {walltimes}"
            labels.append(label)
        print(f"  {'; '.join(labels)}: {message}", file=out)


def print_jobs(snap, out=None):
    print("My jobs:", file=out)
    if snap.jobs is None:
        print("squeue failed\n", file=out)
        return
    if not snap.jobs:
        print("none\n", file=out)
        return
    print_table(
        [
            "Job",
            "Name",
            "State",
            "Partition",
            "Start",
            "Time",
            "Left",
            "Nodes",
            "Reason",
        ],
        [views.job_row(job) for job in snap.jobs],
        out=out,
    )


def print_report(info, snap, profile, settings, host, out=None):
    where = f" (via ssh {info.ssh})" if info.ssh else ""
    print(
        f"{info.user}@{info.hostname} · {info.name or 'unknown cluster'}{where} · "
        f"Slurm {info.version} · {snap.taken_at:%Y-%m-%d %H:%M:%S}",
        file=out,
    )
    for note in snap.notes:
        print(f"Note: {note}", file=out)
    print(file=out)
    print_partitions(snap, out=out)
    print_hardware(snap, out=out)
    print_scopes(snap, info.user, out=out)
    # Banner lines learned by one scope's probes also clean the next's reasons.
    banner = set()
    for scope in snap.scopes or [Scope()]:
        grid = probes.build_grid(snap, scope, settings, profile, info)
        grid.banner = banner
        probes.run_grid(grid, host)
        print_grid(grid, datetime.datetime.now(), out=out)
    print_jobs(snap, out=out)


def print_cluster(cluster, settings, out=None):
    """Print one cluster's report; return its snapshot, or None if unreachable."""
    try:
        info = snapshot.ClusterInfo.detect(cluster.host)
        snap = snapshot.Collector(info, cluster.host).collect()
        print_report(info, snap, cluster.profile, settings, cluster.host, out=out)
        return snap
    except hosts.ConnectionLost as error:
        print(
            f"{cluster.name}: cannot connect over ssh ({cluster.host.ssh}): {error}",
            file=out,
        )
        print(hosts.connect_hint(cluster.host.ssh), file=out)
        print(file=out)
        return None


def print_reports(clusters, settings, out=None):
    """One report per cluster, in order; an unreachable cluster says why."""
    for index, cluster in enumerate(clusters):
        if index:
            print(SEPARATOR, file=out)
            print(file=out)
        print_cluster(cluster, settings, out=out)
