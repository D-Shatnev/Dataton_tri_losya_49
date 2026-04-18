"""
Experiment CLI (dev workflow): run modular pipeline from a TOML config and persist artifacts.

Usage:

    speakerid-experiment --config configs/experiments/my_exp.toml

All components (encoder, indexer, loader, evaluator) are resolved from the TOML
config via :mod:dataton_tri_losya_49.pipeline.registry.  To swap an encoder,
edit only the [encoder] section and re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataton_tri_losya_49.constants import DEFAULT_BATCH_SIZE
from dataton_tri_losya_49.pipeline.config import load_experiment_config
from dataton_tri_losya_49.pipeline.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the speakerid-experiment CLI.

    Returns:
        Configured :class:argparse.ArgumentParser with --config and
        --batch-size arguments.
    """
    p = argparse.ArgumentParser(
        prog="speakerid-experiment",
        description="Run a speaker recognition experiment from a TOML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True, help="Path to experiment TOML config")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Encoder batch size")
    p.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="Number of waveforms to prefetch from disk in background (0 = disable)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run experiment CLI: config.toml -> artifacts directory."""
    args = build_parser().parse_args(argv)

    cfg = load_experiment_config(args.config)

    print(f"[experiment] {cfg.experiment.name}")
    print(f"  encoder  : {cfg.encoder.type}  ({cfg.encoder.model_path})")
    print(f"  indexer  : {cfg.index.backend}  topk={cfg.index.topk}")
    print(f"  loader   : {cfg.loader.type}  sr={cfg.loader.target_sr}")
    print(f"  evaluator: {cfg.evaluation.type}  ks={cfg.evaluation.ks}")
    print(f"  providers: {'auto' if cfg.encoder.providers is None else cfg.encoder.providers}")

    art = run_experiment(cfg, config_path=args.config, batch_size=int(args.batch_size), prefetch=int(args.prefetch))

    print(f"\nrun_dir   : {art.run_dir}")
    print(f"embeddings: {art.embeddings_path}")
    print(f"submission: {art.submission_path}")
    print(f"timing    : {art.timing_path}")
    if art.metrics_path is not None:
        print(f"metrics   : {art.metrics_path}")

    t = json.loads(art.timing_path.read_text(encoding="utf-8"))
    print(
        f"\ntiming    : inference={t['inference_time_s']:.3f}s"
        f"  search={t['search_time_s']:.3f}s"
        f"  total={t['total_time_s']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
