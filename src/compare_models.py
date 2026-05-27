"""Backward-compatible CLI/import path for model comparison summaries."""

from .eval.compare_models import *
from .eval.compare_models import main as _main


if __name__ == "__main__":
    _main()
