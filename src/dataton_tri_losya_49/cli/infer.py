"""
Inference CLI: input CSV -> output submission CSV.

This is the primary user-facing workflow.

Design:
  - All *component* settings (which encoder/indexer/loader to use, non-path
    defaults) come from an inference TOML config.  The default config is
    configs/inference/baseline.toml; override with --config.
  - All *path* arguments (csv, output, model, root) come from CLI flags so
    the same config can be reused with different datasets / submissions.
  - ONNX providers are auto-detected at runtime (CUDA → CPU fallback);
    to override, add providers = [...] to [encoder] in your TOML.

Minimal invocation (using all TOML defaults):

    speakerid-infer --csv data/test_public.csv --out submission.csv

With explicit model override:

    speakerid-infer --csv data/test_public.csv --out submission.csv \\
                    --model models/my_encoder.onnx

With custom config:

    speakerid-infer --csv data/test_public.csv --out submission.csv \\
                    --config configs/inference/my_config.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataton_tri_losya_49.io import write_submission_csv
from dataton_tri_losya_49.pipeline.components.loaders import CsvAudioDatasetLoader
from dataton_tri_losya_49.pipeline.config import load_inference_config
from dataton_tri_losya_49.pipeline.registry import auto_providers, build_encoder, build_indexer, build_waveform_loader
from dataton_tri_losya_49.pipeline.runner import extract_embeddings

# Default config path — stable, can always be overridden via --config
_DEFAULT_CONFIG = Path("configs/inference/baseline.toml")


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the speakerid-infer CLI.

    Returns:
        Configured :class:argparse.ArgumentParser.
    """
    p = argparse.ArgumentParser(
        prog="speakerid-infer",
        description="Speaker recognition inference: CSV → submission.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required paths ---
    p.add_argument("--csv", type=Path, required=True, help="Path to input CSV with audio filepaths")
    p.add_argument("--out", type=Path, required=True, help="Path to output submission CSV")

    # --- optional path overrides ---
    p.add_argument("--root", type=Path, default=Path("."), help="Root directory for resolving relative audio paths")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override model path from TOML config (e.g. models/my_encoder.onnx)",
    )

    # --- config ---
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to inference TOML config",
    )

    # --- optional param overrides (override TOML defaults) ---
    p.add_argument("--k", type=int, default=None, help="Override topk from TOML config")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size from TOML config")
    p.add_argument("--chunk-seconds", type=float, default=None, help="Override chunk_seconds from TOML config")

    return p


def main(argv: list[str] | None = None) -> int:
    """
    Run inference CLI: CSV → embeddings → kNN → submission CSV.

    Steps:
      1. Load inference TOML config (components + defaults).
      2. Apply CLI overrides (model path, topk, batch_size, chunk_seconds).
      3. Detect ONNX providers (auto unless specified in config).
      4. Build encoder, indexer, dataset loader via registry.
      5. Extract embeddings, run kNN, write submission.

    Args:
        argv: Optional argv override (useful for tests). If None, uses sys.argv.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)

    # --- load TOML config ---
    config_path: Path = args.config.resolve() if not args.config.is_absolute() else args.config
    cfg = load_inference_config(config_path)

    # --- resolve effective parameters (CLI wins over TOML defaults) ---
    topk: int = int(args.k) if args.k is not None else cfg.index.topk
    batch_size: int = int(args.batch_size) if args.batch_size is not None else cfg.defaults.batch_size
    chunk_seconds: float = float(args.chunk_seconds) if args.chunk_seconds is not None else cfg.defaults.chunk_seconds

    # --- providers: use config if explicit, else auto-detect ---
    providers = cfg.encoder.providers if cfg.encoder.providers is not None else auto_providers()

    # --- build components via registry ---
    encoder = build_encoder(cfg.encoder, providers=providers, model_path_override=args.model)
    indexer = build_indexer(cfg.index)
    waveform_loader = build_waveform_loader(cfg.loader)

    # --- dataset ---
    dataset = CsvAudioDatasetLoader(
        csv_path=args.csv.resolve(),
        root=args.root.resolve(),
        filepath_col=cfg.defaults.filepath_col,
        chunk_seconds=chunk_seconds,
        loader=waveform_loader,
    )
    filepaths = list(dataset.filepaths)

    # --- inference ---
    embeddings = extract_embeddings(
        waveforms=dataset.iter_waveforms(),
        encoder=encoder,
        batch_size=batch_size,
    )

    neighbors = indexer.neighbors(embeddings, topk=topk)
    write_submission_csv(args.out, filepaths=filepaths, neighbors=neighbors)
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
