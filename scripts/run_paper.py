"""Run the paper pipeline from the project directory on one NVIDIA GPU.

Install: python -m pip install -e '.[cuda]'
Offline check: python -m paraattn smoke --output runs/smoke
Tests: python -m unittest discover -s tests -v
One seed: python scripts/run_paper.py
Five seeds: python scripts/run_paper.py --seeds 42 43 44 45 46
Ablations: python -m paraattn suite --suite heads --output runs/head_ablation
Other suites: representations, layers, weights.

Generation attempts, item predictions, checkpoints, actual parameter counts,
confidence intervals and paired statistics are created under data/ and runs/.
The ZIP contains source only; downloading model/data files happens at runtime.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--candidates", default="data/arc_candidates")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output", default="runs/paper_baselines")
    parser.add_argument("--skip-generation", action="store_true", help="Use existing, audited JSONL data")
    args = parser.parse_args()
    from paraattn.config import load_config
    config = load_config(args.config)

    def run(*arguments):
        subprocess.run([sys.executable, "-m", "paraattn", *arguments], check=True)

    if not args.skip_generation:
        if not (Path(args.candidates) / "manifest.json").exists():
            run("prepare", "--config", args.config, "--output", args.candidates)
        run("paraphrase", "--config", args.config, "--candidates", args.candidates,
            "--output", config.data.root)
    run("audit-data", "--config", args.config)
    run("suite", "--config", args.config, "--suite", "baselines", "--output", args.output,
        "--seeds", *(str(seed) for seed in args.seeds))


if __name__ == "__main__":
    main()
