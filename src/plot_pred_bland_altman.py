"""Backward-compatible CLI/import path for prediction diagnostic plots."""

from .eval.plot_pred_bland_altman import *
from .eval.plot_pred_bland_altman import main as _main


if __name__ == "__main__":
    _main()
