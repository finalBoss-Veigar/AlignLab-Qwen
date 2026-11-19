"""Offline evaluation plumbing; official metrics require actual model outputs."""

from .report import write_report

__all__ = ["write_report"]
