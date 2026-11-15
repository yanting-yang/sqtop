"""Live usage of a running job, as Slurm's `sstat` reports it.

`take` reads one `Sample`: the CPU time, memory and, where the site gathers
them, GPU utilization and GPU memory of the job's running steps, next to
what the job holds. Nothing runs inside the job: sstat asks the node's step
daemons, so sampling starts no job step. CPU use is a rate, so it takes two
samples (`busy_cpus`); `rows` describes the latest ones as meters, the way
htop and nvitop show a machine's.
"""

import dataclasses
import time

from . import slurm

# CPU use is measured over the samples of this many seconds. CPU time comes
# in whole seconds, so a longer span reads steadier than one interval.
WINDOW = 30


@dataclasses.dataclass(frozen=True)
class Sample:
    """One reading of a running job."""

    # When, by the cluster's clock, in seconds.
    taken: float
    # CPU seconds each running step has used so far.
    steps: dict
    # MB the tasks hold now, and the largest task's peak.
    mem: float
    mem_peak: float
    # Percent of a GPU in use, summed over GPUs, and MB of GPU memory; None
    # where Slurm does not gather them.
    gpu_util: float | None
    gpu_mem: float | None
    # What the job holds: CPUs, MB of memory (None if unknown), GPUs by type.
    cpus: int
    mem_total: float | None
    gpus: dict


@dataclasses.dataclass(frozen=True)
class Row:
    """One meter: a resource, the share of it in use and that in words."""

    label: str
    # 0 to 1, or None when there is no bar to draw.
    fraction: float | None
    text: str
    # A full bar is a warning (memory near its limit) rather than good use.
    limit: bool = False


def take(host, job_id, earlier=None):
    """One sample of a running job, or None when sstat reports nothing.

    That is a job with no running step, such as one that just ended. What
    the job holds does not change while it runs, so it is copied from an
    `earlier` sample when there is one and asked of squeue only otherwise.
    Raises `hosts.ConnectionLost` if ssh fails.
    """
    commands = [slurm.usage_command(job_id), slurm.CLOCK_COMMAND]
    if earlier is None:
        commands.append(slurm.allocation_command(job_id))
    stats, clock, *allocation = host.run_batch(commands)
    reading = slurm.parse_usage(stats or "")
    if reading is None:
        return None
    if earlier is None:
        held = slurm.parse_allocation(allocation[0] or "")
        if held is None:
            return None
        held = {"cpus": held["cpus"], "mem_total": held["mem"], "gpus": held["gpus"]}
    else:
        held = {
            "cpus": earlier.cpus,
            "mem_total": earlier.mem_total,
            "gpus": earlier.gpus,
        }
    try:
        taken = float(clock)
    except (TypeError, ValueError):  # no clock, or a `date` without %N
        taken = time.time()
    return Sample(taken=taken, **held, **reading)


def add(samples, sample):
    """`samples` followed by `sample`, without those older than `WINDOW`."""
    recent = [
        earlier
        for earlier in samples
        if sample.taken - WINDOW <= earlier.taken < sample.taken
    ]
    return [*recent, sample]


def busy_cpus(samples):
    """CPUs kept busy over the span of `samples`, or None with under two.

    A step that started within the span used all of its CPU time in it; one
    that ended is left out, as what it used in the span is not known.
    """
    if len(samples) < 2:
        return None
    first, last = samples[0], samples[-1]
    span = last.taken - first.taken
    if span <= 0:
        return None
    used = sum(cpu - first.steps.get(step, 0.0) for step, cpu in last.steps.items())
    return max(0.0, used / span)


def _share(used, total):
    return min(1.0, max(0.0, used / total)) if total else None


def _gb(megabytes):
    return f"{megabytes / 1024:.1f}"


def _whole_gb(megabytes):
    """A total in GB, without a needless `.0`."""
    return f"{round(megabytes / 1024, 1):g}"


def gpu_rows(sample):
    """GPU utilization and memory, for a job that holds GPUs.

    Sites that do not gather GPU use leave it out of sstat. A MIG slice has
    no utilization to report (NVML has none per process), but its type names
    its memory, which makes its memory row a meter.
    """
    if not sample.gpus:
        return []
    count = sum(sample.gpus.values())
    kinds = list(sample.gpus)
    held = f"{count} {kinds[0]}" if len(kinds) == 1 else f"{count} GPUs"
    if sample.gpu_util is None and sample.gpu_mem is None:
        return [Row("GPU", None, f"use of the {held} is not reported by Slurm here")]
    slices = [slurm.mig_memory(gpu) for gpu in sample.gpus]
    rows = []
    if all(slices):
        note = f"utilization is not reported for MIG slices ({held})"
        rows.append(Row("GPU", None, note))
    elif sample.gpu_util is not None:
        rows.append(Row("GPU", _share(sample.gpu_util, 100 * count), f"of {held}"))
    if sample.gpu_mem is not None and all(slices):
        sizes = zip(slices, sample.gpus.values())
        total = 1024 * sum(size * number for size, number in sizes)
        text = f"{_gb(sample.gpu_mem)} of {_whole_gb(total)} GB"
        rows.append(Row("GPU mem", _share(sample.gpu_mem, total), text, limit=True))
    elif sample.gpu_mem is not None:
        rows.append(Row("GPU mem", None, f"{_gb(sample.gpu_mem)} GB in use"))
    return rows


def rows(samples):
    """The latest of `samples` as meters: CPU, memory, then any GPUs.

    CPU use needs two samples; until then its row says so.
    """
    sample = samples[-1]
    busy = busy_cpus(samples)
    if busy is None:
        cpu = Row("CPU", None, f"measuring, of {sample.cpus} cores")
    else:
        text = f"{busy:.1f} of {sample.cpus} cores"
        cpu = Row("CPU", _share(busy, sample.cpus), text)
    peak = f" · peak {_gb(sample.mem_peak)} GB"
    if sample.mem_total:
        text = f"{_gb(sample.mem)} of {_whole_gb(sample.mem_total)} GB{peak}"
        memory = Row("Memory", _share(sample.mem, sample.mem_total), text, limit=True)
    else:
        memory = Row("Memory", None, f"{_gb(sample.mem)} GB{peak}")
    return [cpu, memory, *gpu_rows(sample)]
