"""Filesystem roots for the experiments.

Everything is resolved relative to the repository, so a checkout works without editing
paths. Set JEVSEC_ROOT to write runs and figures somewhere else, for example a scratch
directory on a cluster.
"""
import os

ROOT = os.environ.get(
    "JEVSEC_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "figs")
