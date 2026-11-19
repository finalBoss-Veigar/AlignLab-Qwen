"""Deterministic data governance; model dependencies are optional."""

from .pipeline import run_fixture_pipeline, run_pipeline

__all__ = ["run_pipeline", "run_fixture_pipeline"]
