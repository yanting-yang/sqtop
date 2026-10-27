"""The dashboard's hourly email: the text report of each cluster it shows.

Each email starts with your job counts and what happened to your jobs since
the previous email (submitted, started, ended), then has the report
`sqtop` prints when output is not a terminal. Every email has the
subject "sqtop" and refers to the first one, so mail clients keep them
in one conversation rather than as separate emails.

The server comes from `[smtp]` in the config (`profiles.load_smtp`). The
thread and the jobs last reported are kept in `STATE_PATH`, so the
conversation carries on from one run of the dashboard to the next.
"""

import datetime
import email.message
import email.utils
import html
import io
import json
import os
import smtplib
import ssl
from pathlib import Path

from . import hosts, report, slurm, views

SUBJECT = "sqtop"
INTERVAL = 3600
SSL_PORT = 465
SMTP_TIMEOUT = 60
STATE_PATH = (
    Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    / "sqtop"
    / "mail.json"
)


def compose(smtp, text, thread=None):
    """The email: `text` as plain text and as monospaced HTML.

    `thread` is the first email's Message-ID. Replying to it, under the same
    subject, keeps every email in one conversation.
    """
    message = email.message.EmailMessage()
    message["Subject"] = SUBJECT
    message["From"] = email.utils.formataddr((SUBJECT, smtp.sender))
    message["To"] = smtp.recipient
    message["Date"] = email.utils.formatdate(localtime=True)
    domain = smtp.sender.rpartition("@")[2] or None
    message["Message-ID"] = email.utils.make_msgid(SUBJECT, domain)
    if thread:
        message["In-Reply-To"] = thread
        message["References"] = thread
    message.set_content(text)
    # The tables only line up in a fixed-width font, which plain text
    # does not get everywhere (Gmail's web client, for one).
    message.add_alternative(
        f'<pre style="font: 12px/1.4 monospace">{html.escape(text)}</pre>',
        subtype="html",
    )
    return message


def send(smtp, message):
    """Deliver `message` over SSL (port 465) or STARTTLS.

    A password is never sent over a connection that is not encrypted.
    """
    context = ssl.create_default_context()
    if smtp.port == SSL_PORT:
        connection = smtplib.SMTP_SSL(
            smtp.server, smtp.port, timeout=SMTP_TIMEOUT, context=context
        )
    else:
        connection = smtplib.SMTP(smtp.server, smtp.port, timeout=SMTP_TIMEOUT)
    with connection:
        if smtp.port != SSL_PORT:
            connection.ehlo()
            if connection.has_extn("starttls"):
                connection.starttls(context=context)
                connection.ehlo()
            elif smtp.password:
                raise smtplib.SMTPException(
                    f"{smtp.server}:{smtp.port} does not offer STARTTLS; "
                    "not sending the password unencrypted"
                )
        if smtp.password:
            connection.login(smtp.sender, smtp.password)
        connection.send_message(message)


def job_states(jobs):
    """What the next email compares with: each job's state and name."""
    return {job["id"]: {"state": job["state"], "name": job["name"]} for job in jobs}


def job_changes(previous, jobs, ended=None):
    """One line per job whose state changed since `previous` (`job_states`).

    `ended` maps jobs that left the queue to their (state, exit code) from
    sacct; without an entry such a job only "left the queue".
    """
    ended = ended or {}
    changes = []
    for job in jobs:
        before = previous.get(job["id"], {}).get("state")
        if before == job["state"]:
            continue
        label = f"{job['id']} ({job['name']})"
        if job["state"] == "RUNNING":
            nodes, _ = views.job_nodes(job)
            changes.append(f"{label} started on {nodes}")
        elif before is None and job["state"] == "PENDING":
            text = f"{label} submitted"
            if (reason := views.job_reason(job)) != "-":
                text += f" ({reason})"
            if (start := views.job_start_time(job, compact=True)) != "-":
                text += f", expected to start {start}"
            changes.append(text)
        else:
            changes.append(f"{label} is now {job['state']}")
    current = {job["id"] for job in jobs}
    for job_id, job in previous.items():
        if job_id in current:
            continue
        label = f"{job_id} ({job['name']})"
        if job_id not in ended:
            changes.append(f"{label} left the queue")
            continue
        state, code = ended[job_id]
        exit_code = "" if code in ("", "0:0") else f", exit code {code}"
        changes.append(f"{label} ended: {state}{exit_code}")
    return changes


def final_states(host, ids):
    """How jobs that left the queue ended, from sacct; {} when it fails."""
    # A pending array (`12_[1-9]`) is not an id sacct takes in a list.
    ids = [job_id for job_id in ids if job_id.replace("_", "").isdigit()]
    if not ids:
        return {}
    try:
        output = host.run(slurm.sacct_command(ids))
    except hosts.ConnectionLost:
        return {}
    return {} if output is None else slurm.parse_sacct_states(output)


def digest(targets, state, now):
    """The email's text, and the jobs the next email compares with.

    `targets` pairs each cluster with the probe request to report. The text
    has a line of job counts per cluster, each followed by the changes since
    the email `state` records, then each cluster's report.
    """
    seen = state.get("jobs", {})
    since = views.parse_time(state.get("sent_at"))
    if since is not None:
        since = f"{since:%H:%M}" if since.date() == now.date() else f"{since:%F %H:%M}"
    latest = dict(seen)
    heads, reports = [], []
    for cluster, settings in targets:
        output = io.StringIO()
        snap = report.print_cluster(cluster, settings, out=output)
        reports.append(output.getvalue())
        if snap is None or snap.jobs is None:
            heads.append(f"{cluster.name}: cannot read your jobs")
            continue
        counts = slurm.job_counts(snap.jobs)
        head = (
            f"{cluster.name}: {counts['running']} running, {counts['pending']} pending"
        )
        previous = seen.get(cluster.name)
        latest[cluster.name] = job_states(snap.jobs)
        if previous is None or since is None:
            heads.append(head)
            continue
        current = {job["id"] for job in snap.jobs}
        gone = [job_id for job_id in previous if job_id not in current]
        changes = job_changes(previous, snap.jobs, final_states(cluster.host, gone))
        if not changes:
            heads.append(f"{head}; no changes since {since}")
            continue
        heads.append(f"{head}. Since {since}:")
        heads.extend(f"  {change}" for change in changes)
    separator = f"{report.SEPARATOR}\n\n"
    return "\n".join(heads) + "\n\n" + separator.join(reports), latest


def load_state(path):
    """What the last email recorded, or {} before the first one."""
    try:
        state = json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=1))
    temporary.replace(path)


def send_report(targets, smtp, path=None, now=None):
    """Email the report of `targets` once and return the message.

    Raises OSError (including `smtplib.SMTPException`) when it cannot be
    sent. The state is saved only after the email is sent, so the next email
    still lists every change since the last one that went out.
    """
    path = STATE_PATH if path is None else path
    now = now or datetime.datetime.now()
    state = load_state(path)
    text, jobs = digest(targets, state, now)
    threads = state.get("threads", {})
    message = compose(smtp, text, threads.get(smtp.recipient))
    send(smtp, message)
    threads.setdefault(smtp.recipient, message["Message-ID"])
    save_state(
        path,
        {
            "threads": threads,
            "sent_at": now.isoformat(timespec="seconds"),
            "jobs": jobs,
        },
    )
    return message
