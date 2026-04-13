"""Experiment CLI (dev workflow): run modular pipeline on a config and persist artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataton_tri_losya_49.constants import DEFAULT_BATCH_SIZE
from dataton_tri_losya_49.pipeline.config import load_experiment_config
from dataton_tri_losya_49.pipeline.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the speakerid-experiment CLI.

    Returns:
        Configured :class:argparse.ArgumentParser.
    """
    p = argparse.ArgumentParser(prog="speakerid-experiment")
    p.add_argument("--config", type=Path, required=True, help="Path to experiment TOML")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return p


def main(argv: list[str] | None = None) -> int:
    """
    Run experiment CLI: config.toml -> artifacts directory.

    Args:
        argv: Optional argv override (useful for tests). If None, uses sys.argv.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    cfg = load_experiment_config(args.config)
    art = run_experiment(cfg, config_path=args.config, batch_size=int(args.batch_size))
    print(f"run_dir: {art.run_dir}")
    print(f"embeddings: {art.embeddings_path}")
    print(f"submission: {art.submission_path}")
    if art.metrics_path is not None:
        print(f"metrics: {art.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
