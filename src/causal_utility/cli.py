"""Command-line interface for reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import load_config, make_figures, run_experiment, save_results, save_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate causal utility of synthetic health data")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a YAML-configured Monte Carlo experiment")
    run.add_argument("--config", required=True, help="Path to YAML config")
    run.add_argument("--output", default="results", help="Output directory for aggregate CSVs")
    run.add_argument("--figures", default="figures", help="Output directory for figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        results = run_experiment(cfg)
        save_results(results, args.output)
        save_run_metadata(cfg, args.output)
        make_figures(results, args.figures)
        print(f"Completed experiment. Aggregate results: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
