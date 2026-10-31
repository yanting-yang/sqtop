"""sqtop: a Slurm dashboard of partitions, limits and estimated start times."""

from importlib.metadata import PackageNotFoundError, version

try:
    # From the installed package, so it always matches pyproject.toml.
    __version__ = version("sqtop")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "unknown"
