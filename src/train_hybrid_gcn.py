"""Backward-compatible CLI/import path for PyTorch hybrid GCN training."""

from .trainers.train_hybrid_gcn import *
from .trainers.train_hybrid_gcn import main as _main


if __name__ == "__main__":
    _main()
