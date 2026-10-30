"""A stand-in for `hosts.Local`/`hosts.Remote` that answers from canned text."""

from sqtop import hosts


def scheduler_reply(start=None, partition="gpu", error="error: rejected"):
    """What sbatch/srun --test-only print: a start time, or an error."""
    if start is None:
        return 1, f"sbatch: {error}"
    return (
        0,
        f"sbatch: Job 1 to start at {start} a using 4 processors in partition {partition}",
    )


class FakeHost:
    """Answers `run`/`run_batch` by command prefix and probes with `probe`.

    `responses` maps the start of a command line (e.g. "scontrol show node")
    to its stdout, or None for a failure; unmatched commands fail.
    `probe(args)` returns (returncode, output) for each streamed command.
    `lost` makes every call raise `hosts.ConnectionLost`. `act` answers with
    `act_reply`, recording the commands in `acted`.
    """

    def __init__(self, responses=None, probe=None, ssh=None, lost=None):
        self.responses = responses or {}
        self.probe = probe or (lambda args: scheduler_reply())
        self.ssh = ssh
        self.lost = lost
        self.env = {}
        self.workdir = None
        self.batches = []
        self.streamed = []
        self.act_reply = (0, "Submitted batch job 77")
        self.acted = []

    def act(self, args, timeout=60):
        if self.lost:
            raise hosts.ConnectionLost(self.lost)
        self.acted.append(args)
        return self.act_reply

    def respond(self, args):
        line = " ".join(args)
        for prefix, output in self.responses.items():
            if line.startswith(prefix):
                return output(args) if callable(output) else output
        return None

    def run(self, args, timeout=60):
        return self.run_batch([args])[0]

    def run_batch(self, commands, timeout=120, login_path=False):
        if self.lost:
            raise hosts.ConnectionLost(self.lost)
        self.batches.append(commands)
        return [self.respond(args) for args in commands]

    def run_stream(self, commands, on_result, should_stop=lambda: False):
        if self.lost:
            raise hosts.ConnectionLost(self.lost)
        self.streamed.append(commands)
        for index, args in enumerate(commands):
            if should_stop():
                return
            returncode, output = self.probe(args)
            on_result(index, returncode, output)
