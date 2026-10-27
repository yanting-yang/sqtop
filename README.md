# sqtop

A terminal dashboard for Slurm, in the spirit of
[hunoutl/slurmtop](https://github.com/hunoutl/slurmtop) and
[slmtop](https://github.com/dawnmy/slmtop), that answers "where can I run, under which
account/QOS, and when would it start?", and can email you that report every hour while
it runs.

```
 killarney  fir  tamia
╸━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 someone@klogin02  killarney  Slurm 25.05.9                                                              updated 13:49:06 (10s ago) · probed 13:49:06
╭─ Partitions ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Partition        Use  CPU/node  Mem/node  GPU/node  Load           Free         Idle   Pend  Max                                                   │
│ gpubase_h100_b1  yes  48/96     2011G     8x h100   ██████▒▒  75%  4/80 h100    0/10   117   0-03:00:00                                            │
│ gpubase_h100_b2  yes  48/96     2011G     8x h100   ███████▒  86%  1/64 h100    0/8    63    0-12:00:00                                            │
│ gpubase_interac  yes  64        502G      4x l40s   ███████░  90%  6/100 l40s   0/25   1     0-03:00:00                                            │
│ gpubase_l40s_b1  yes  64        502G      4x l40s   ███████▒  90%  18/672 l40s  1/168  2571  0-03:00:00                                            │
│ gpubase_l40s_b3  yes  64        502G      4x l40s   ███████▒  88%  8/336 l40s   1/84   90    1-00:00:00                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Start estimates · aip-abc123/normal ──────────────────────────────────────────────────╮╭─ Accounts & QOS ─────────────────────────────────────────╮
│ Request   0-03:00:00    0-12:00:00    1-00:00:00    3-00:00:00    7-00:00:00           ││ Account      QOS      Jobs  GPUs  Wall        FS         │
│ 1x h100     0-22:05:00    1-05:38:00    3-02:01:00    7-23:35:00   39-23:02:00         ││ aip-abc123*  interac  0/1   ∞     0-03:00:00  0.14       │
│ 8x h100     1-04:16:00    1-10:21:00    3-06:44:00    8-11:56:00   40-05:13:00         ││ aip-abc123*  normal*  0/∞   ∞     ∞           0.14       │
│ 1x l40s     0-02:12:00    0-01:27:00    0-01:48:00    0-19:17:00    7-15:47:00         │╰────────────────── * left out of commands · enter: probe ─╯
│ 4x l40s     0-02:12:00    0-02:17:00    0-01:48:00    0-21:38:00    7-18:03:00         │╭─ My jobs ────────────────────────────────────────────────╮
│ CPU only    0-02:12:00    0-01:27:00    0-01:48:00    0-19:17:00    7-15:47:00         ││ Job    Name   St  Start        Time  Nodes  Reason       │
│                                                                                        ││ No jobs in the queue                                     │
╰───────────────────────────────────────────────────────────────── sbatch -c4 --mem=32G ─╯╰──────────────────────────────────────────────────────────╯
╭─ Details ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ gpubase_h100_b1  UP  usable  · max time 0-03:00:00 · priority tier 1                                                                               │
│ Nodes 10 (5 mixed, 3 allocated, 2 unavailable) · pending jobs 117                                                                                  │
│ 9 nodes: 48 CPUs / 2011 GB / 8x h100 · 1 node: 96 CPUs / 2011 GB / 8x h100                                                                         │
│ CPUs 104 free / 328 allocated / 96 unavailable of 528 · Memory 12045 GB free of 20117 GB                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
 Refresh   Re-probe   Edit request   Email: off   Config   Quit
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (it fetches Python 3.13 if you do not have it)
- A Slurm login node (`scontrol` and `squeue` on `PATH`; estimates need `sbatch`/`srun`)

## Usage

```bash
uvx sqtop                  # run it from PyPI: one tab per cluster, this one first
uvx sqtop > report.txt     # text report of this cluster when not a terminal
uv tool install sqtop      # or install it once, then just type `sqtop`

# From a checkout
uv run sqtop
```

There are no options. The first tab is the cluster you are logged in to (from
`ClusterName` in `scontrol show config`), and it is the only one checked at startup.
Every other cluster with an `ssh` host in the [config file](#configuration) (Fir,
Killarney and Tamia) gets a tab too, but is only contacted, over ssh, once you open its
tab (click it). Every probe starts as 4 CPUs and 32 GB at each partition
time limit with `sbatch`; change a tab's request with `Edit request` (CPUs, memory,
walltimes, extra `sbatch` options), and click the command under Start estimates to
switch between `sbatch` and `srun`. The visible tab refreshes every minute and re-probes
every five. Click `Email: off` to email the report every hour until you quit
([Hourly email](#hourly-email)). Everything is a click: see [Mouse](#mouse).

Probes use `--test-only`, so nothing is submitted or cancelled until you confirm a
Submit or Release (see below), and it is safe to run on a shared login node. Run it on
the login node: on a compute node, such as inside an interactive job, `srun` probes fail
with "Requested operation is presently disabled".

### Remote clusters

Remote tabs run Slurm on the cluster's login node through `ssh`, with your
`~/.ssh/config` (aliases, users, keys). Nothing is sent to a remote cluster until you
open its tab, and only the visible tab refreshes.

sqtop never asks for a password or MFA. For a cluster that needs one, let ssh share
connections and log in with `ssh` in another terminal first; sqtop reuses that
connection:

```
# ~/.ssh/config
Host *.alliancecan.ca
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 4h
```

After `ssh fir.alliancecan.ca`, the connection stays open for 4 hours after your last
session there closes. With keys that log in without a prompt, nothing is needed:
sqtop opens its own shared connection, which closes 10 minutes after the last use.
A tab that cannot connect shows ssh's error and how to connect; then click `Refresh`.

The text report printed when output is not a terminal covers the cluster you are on;
it covers the remote ones only where Slurm is not available.

Each refresh of a remote tab is one ssh session, and the cache of limits is fetched
only for your accounts and QOS, because login nodes can take seconds to start a
session and the full cache can be tens of MB. Commands for a remote cluster are shown
with its host as the prompt (`fir.alliancecan.ca$ sbatch ...`): run them there.

### Mouse

sqtop has no keyboard shortcuts: everything is a click. The bar at the bottom holds
the dashboard's actions, and each dialog closes with its own buttons.

| Click | Action |
| --- | --- |
| a cluster tab | Show that cluster |
| a row or cell | Select it; the Details panel follows the selection |
| a highlighted Accounts & QOS row | Probe with that account/QOS |
| the Partitions title | Switch between Partitions and Nodes by hardware |
| the account/QOS above the estimates | Probe with the next account/QOS |
| the command under Start estimates | Switch between `sbatch` and `srun` probes |
| `Submit` in Details | Submit the selected `sbatch` request, after you confirm |
| `Copy` after a command in Details | Copy that command |
| `Release` in a job's Details | Release the job with `scancel`, after you confirm |
| `Refresh` / `Re-probe` | Refresh everything / re-probe now |
| `Edit request` | Edit the probe request: whole node, CPUs, memory, walltimes, extra options such as `--partition` or `--constraint`, and the email address (blank for none) |
| `Email: off` / `Email: on` | Email the report now and every hour until you quit / stop; the label says which is on |
| `Config` | Edit the [config](#configuration) in a form; saving writes the file and applies it |
| `Quit` | Quit |

## Reading the dashboard

- **Partitions**: `Use` says whether your accounts, QOS and Unix groups may submit
  there. `CPU/node`, `Mem/node` and `GPU/node` are each node's hardware (CPUs exclude
  reserved cores); a partition with two kinds of node shows both (`48/96`), with more
  a range. `Load` is the share of GPUs (CPUs on CPU-only partitions) in use: `█`
  allocated, `▒` unavailable (down, drained, reserved), `░` free. `Free` is free/total
  per GPU type, `Idle` counts wholly idle nodes, `Pend` counts pending jobs from all
  users, and `Max` is the time limit. Details list each kind of node with its count,
  then memory, `AllowAccounts`/`AllowQos`, the partition QOS and priority tier.
- **Accounts & QOS**: every account/QOS pair you can submit under; `*` marks what
  commands leave out because Slurm picks it anyway: the account when it is your only
  one (with several, commands always name it, since sites such as Fir require that
  for GPU jobs) and the account's default QOS. `Jobs` is your running jobs over the tightest per-user
  cap, `GPUs` and `Wall` are the tightest GPU and wall-time caps from any scope, and
  `FS` is your fair-share factor. Details list every cap with its usage: QOS per user,
  per account, per job and per node, plus association limits on you and your account.
  `∞` means no cap, `?` unknown.
- **Start estimates** (titled with the account/QOS probed): rows are request sizes
  (each GPU type up to the tightest GPU cap, then CPU only), columns are walltimes. Each
  cell is how long until the scheduler expects the job to start: `now`, a wait such as
  `0-10:54:00`, `no` when it cannot run (Details gives the reason), or `-` above the
  account/QOS wall-time cap. The earliest start in each row is underlined. Here a 1-day
  L40S job starts sooner than a 3-hour one, because short jobs queue in the busier `b1`
  partition.
- **Details** for an estimate shows the partition it would land in, the probe that
  produced the estimate with the scheduler's reply, and the command to submit the same
  request; click `Copy` after either to copy it. The probe has only what the estimate
  depends on; the submit command drops `--test-only` and adds your email options and
  `--output=/dev/null`, so no `slurm-<jobid>.out` file is left behind.
  `--account`/`--qos` are left out when Slurm would pick them anyway.
- **Submit**: click the `Submit` label in Details to run an `sbatch` command, after a
  dialog shows it and asks you to confirm; `Cancel` submits nothing. The job is submitted in the background (on the cluster's login node
  for a remote tab), its id pops up, and My jobs refreshes; it holds the allocation with
  `sleep infinity` until the time limit; its details in My jobs show the `scancel`
  command that releases it. An `srun` command opens an interactive shell, so `Submit`
  is not clickable for it: click `Copy` and run it in a terminal. A request the probe
  was refused for is not submitted.
- **My jobs** lists your queued and running jobs: `Partition` is where it runs or was
  submitted to (several for some pending jobs), `Start` is when a job started, or when
  the scheduler expects a pending one to (`-` until it has planned it), `Nodes` is where
  it runs, or the nodes planned for a pending job (dimmed, as they can still change),
  and `Reason` is why a pending job waits. A job's details show the command that
  attaches a shell to it (`srun --jobid=<id> --overlap --pty <shell>`, once it runs)
  and the one that releases it (`scancel <id>`), each with `Copy`. Clicking `Release`
  runs `scancel` after a dialog asks you to confirm (Cancel is selected first): a
  running job is killed and its allocation released, a pending one leaves the queue.

## Configuration

All settings live in one file, `~/.config/sqtop/config.toml`
(`$XDG_CONFIG_HOME/sqtop/config.toml` if you set that): email (`[mail]`, `[smtp]`)
and cluster profiles (`[clusters.<name>]`). Click `Config` in the dashboard to edit it
in a form: the job email address and states, the hourly email's server, port, sender,
password and recipient, and one row per cluster (name, GPU counts picked from `pow2`,
`all` and `whole_node`, and ssh host) with `Add cluster` and `Remove`. `Cancel` closes
it unchanged. `Save` checks the settings (a problem is shown under the form, which stays
open), writes the file privately (`chmod 600`, since it holds the SMTP password) and
applies what changed without a restart: clusters added get tabs and removed ones lose
theirs, changed profiles and `[mail]` settings are probed again, and the hourly email
switches to a changed `[smtp]`.

Without the file, `sqtop` shows only the cluster you are logged in to, with no email
settings. The file is plain TOML, so you can also edit it by hand; the form rewrites it
without comments.

### Other clusters

Nothing is hardcoded per cluster: partitions, GPU types, time tiers, accounts and
limits are read from Slurm. A profile records only what Slurm cannot report, keyed by
the cluster name. To add a cluster, or a tab for one you are not logged in to, add a
row in the config form (`Config`); in the file it is a table:

```toml
[clusters.mycluster]          # the ClusterName from `scontrol show config`
gpu_counts = "all"            # "pow2" (1, 2, 4, ... default), "all" or "whole_node"
                              # (GPU jobs then take whole nodes: --exclusive --mem=0)
ssh = "login.mycluster.org"   # a tab for it from anywhere; host or ~/.ssh/config alias
```

For example, the Alliance clusters this was written on need no rules, only an `ssh`
host for a tab, except `tamia`, which allocates GPUs by whole node:

```toml
[clusters.fir]
ssh = "fir.alliancecan.ca"

[clusters.killarney]
ssh = "killarney.alliancecan.ca"

[clusters.tamia]
ssh = "tamia.alliancecan.ca"
gpu_counts = "whole_node"
```

The walltimes probed are not a setting: they are the distinct time limits of the
partitions you can use, which is where routing changes, and `Edit request` overrides
them for a session.

### Whole nodes

Tick "Whole node" in the request editor (`Edit request`) to probe whole nodes: every command then
asks for `--exclusive --mem=0` (every CPU and all the memory) instead of `-c`/`--mem`,
and the GPU rows are each type's full node, such as `8x h100` or `4x l40s`. Untick it
to go back to the CPUs and memory in the editor.

On clusters that only allocate GPUs by node (`gpu_counts = "whole_node"`, such as
Tamia) the GPU rows always ask for the whole node this way, so a job that holds all of
a node's GPUs also gets its CPUs and memory; the CPU-only row keeps the editor's
values. The estimates panel then notes "GPUs by whole node".

### Email notifications

The `[mail]` table in your config (`Config`) adds `--mail-type` and `--mail-user` to every
command, so a job you submit from one emails you when it starts, ends or fails:

```toml
[mail]
user = "you@example.com"
type = "BEGIN,END,FAIL"       # any of sbatch's --mail-type values
```

Only the commands to submit carry them: probes leave them out, since email does not
change when a job would start. Remove the table to leave email out, or clear the
address for a session with `Edit request`.

### Hourly email

Click `Email: off` in the dashboard to email the report now and every hour until you
quit `sqtop` (or click `Email: on` to stop). Each email covers the cluster you are on and every
remote tab you have opened, with each tab's current request. Every email has the subject
`sqtop` and replies to the first one, so your mail client keeps them in one
conversation instead of separate emails. Each starts with your job counts and what
happened since the previous email, followed by the text report:

```
killarney: 1 running, 2 pending. Since 14:00:
  6284538 (wrap) started on kn021
  6233018 (wrap) ended: TIMEOUT
  6290112 (eval) submitted (Priority), expected to start 10-17 09:30
```

The server is read from `[smtp]` in your config (`Config`) when you turn it on:

```toml
[smtp]
server = "mail.example.org"
port = 587                       # STARTTLS; 465 for SSL
sender = "you@example.org"       # also the login
password = "..."
recipient = "you@example.org"    # default: user in [mail]
```

The bottom bar says whether the hourly email is on (`Email: on`) or off (`Email: off`).
While it is on, the top bar shows when the next email goes out (`next email 16:37`),
and marks it `(last failed)` if one could not be sent. The thread and the jobs last reported are kept
in `~/.local/state/sqtop/mail.json`, so the next run of the dashboard continues the
same conversation; delete the file to start a new one. Gmail starts a new conversation
after 100 emails, about every four days.

### When `sacctmgr` fails

On some login nodes `sacctmgr` fails with "Connection refused" because `slurmdbd` runs
on the controller. Accounting is still enforced: `sqtop` reads limits from
`scontrol show assoc_mgr` instead. Without `sacctmgr` it cannot list your QOS
assignments, so it test-submits each cached account/QOS pair and hides the ones Slurm
rejects.

## Development

```bash
uv sync
uv run python -m unittest discover -s tests -t tests
uvx ruff check src tests
```

See [AGENTS.md](AGENTS.md) for the code layout and site-specific Slurm quirks.
