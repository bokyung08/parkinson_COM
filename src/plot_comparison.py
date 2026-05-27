"""Backward-compatible CLI/import path for comparison plots."""

from .eval.plot_comparison import *
from .eval.plot_comparison import main as _main


if __name__ == "__main__":
    _main()
