"""Benchmark layer: score engine quality on known cases and track it per version."""

from researchforge.benchmark.cases import BenchmarkCase, default_cases
from researchforge.benchmark.run import BenchmarkReport, run_benchmark, save_report

__all__ = [
    "BenchmarkCase",
    "default_cases",
    "BenchmarkReport",
    "run_benchmark",
    "save_report",
]
