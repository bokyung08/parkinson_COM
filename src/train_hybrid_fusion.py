"""Backward-compatible CLI/import path for TensorFlow hybrid fusion training."""

from .trainers.train_hybrid_fusion import *
from .trainers.train_hybrid_fusion import main as _main


if __name__ == "__main__":
    _main()
