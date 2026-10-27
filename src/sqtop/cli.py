"""Entry point for `sqtop`: the dashboard, or a one-shot text report."""

import argparse
import sys

from . import hosts, probes, profiles, report, slurm, snapshot


def find_clusters():
    """This cluster, if Slurm runs here, then each cluster with an ssh host.

    The local cluster is detected from `scontrol show config`; a configured
    remote with the same name is skipped, since it is already shown.
    """
    local = hosts.Local()
    config = local.run(slurm.CONFIG_COMMAND)
    clusters = []
    local_name = None
    if config is not None:
        local_name = slurm.parse_config(config).get("ClusterName") or None
        profile = profiles.load_profile(local_name or "")
        clusters.append(snapshot.Cluster(local_name or "local", profile, local))
    for profile in profiles.remote_profiles():
        if profile.name != local_name:
            remote = hosts.Remote(profile.ssh)
            clusters.append(snapshot.Cluster(profile.name, profile, remote))
    return clusters


def main(argv=None):
    # No options: clusters are detected and configured in the config, which
    # Config edits in the dashboard, where the probe request and hourly email
    # are set too. Parsing still rejects stray arguments and serves --help.
    config = profiles.short_path(profiles.CONFIG_PATH)
    argparse.ArgumentParser(
        prog="sqtop",
        description=(
            "Partitions, account/QOS limits and estimated start times for the "
            "Slurm cluster you are logged in to; clusters with an ssh host in "
            f"{config} get a tab that connects when opened, Config edits that "
            "config, and Email sends the dashboard hourly through its [smtp]. "
            "Prints a text report of this cluster when output is not a terminal."
        ),
    ).parse_args(argv)

    try:
        clusters = find_clusters()
        mail = profiles.load_mail()
    except ValueError as error:
        sys.exit(f"sqtop: {profiles.short_path(profiles.CONFIG_PATH)}: {error}")
    if not clusters:
        sys.exit(
            "sqtop: Slurm is not available here ('scontrol' failed), and no "
            f"cluster in {config} has an ssh host."
        )
    settings = probes.Settings()
    if mail is not None:
        settings = probes.Settings(mail_user=mail[0], mail_type=mail[1])

    if not sys.stdout.isatty():
        # Like the dashboard at startup, the report covers the cluster this
        # runs on; remote ones only when Slurm is not available here.
        reported = [c for c in clusters if c.host.ssh is None] or clusters
        report.print_reports(reported, settings)
        return

    # Textual is only imported for the dashboard, keeping the report fast.
    from .tui import SqtopApp

    SqtopApp(clusters, settings).run()


if __name__ == "__main__":
    main()
