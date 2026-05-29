from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CARE-PD with huggingface-cli after access is approved.")
    parser.add_argument("--local_dir", default="data/raw/CARE-PD")
    parser.add_argument("--repo_id", default="vida-adl/CARE-PD")
    args = parser.parse_args()
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "huggingface_hub.commands.huggingface_cli",
        "download",
        args.repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        args.local_dir,
    ]
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
