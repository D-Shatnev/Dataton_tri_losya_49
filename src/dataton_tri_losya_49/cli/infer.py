"""
Inference CLI: input CSV -> output submission CSV.

This is the primary user workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataton_tri_losya_49.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FILEPATH_COL,
    DEFAULT_ONNX_EMBEDDINGS_OUTPUT_NAME,
    DEFAULT_ONNX_PROVIDERS,
    DEFAULT_TARGET_SR,
    DEFAULT_TOPK,
)
from dataton_tri_losya_49.io import write_submission_csv
from dataton_tri_losya_49.pipeline.components.encoders import OnnxEncoder
from dataton_tri_losya_49.pipeline.components.indexers import FaissInnerProductIndexer
from dataton_tri_losya_49.pipeline.components.loaders import CsvAudioDatasetLoader, SoundFileWaveformLoader
from dataton_tri_losya_49.pipeline.runner import extract_embeddings


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the speakerid-infer CLI.

    Returns:
        Configured :class:argparse.ArgumentParser.
    """
    p = argparse.ArgumentParser(prog="speakerid-infer")
    p.add_argument("--model", type=Path, required=True, help="Path to ONNX model")
    p.add_argument("--csv", type=Path, required=True, help="CSV with at least filepath column")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=DEFAULT_TOPK, help="Number of neighbors per item")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    p.add_argument(
        "--providers",
        nargs="+",
        default=list(DEFAULT_ONNX_PROVIDERS),
        help="ONNX Runtime providers priority list",
    )
    p.add_argument(
        "--filepath-col",
        type=str,
        default=DEFAULT_FILEPATH_COL,
        help="CSV column with relative paths to audio files",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_ONNX_EMBEDDINGS_OUTPUT_NAME,
        help="ONNX output name",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run inference CLI: CSV -> embeddings -> kNN -> submission CSV.

    Args:
        argv: Optional argv override (useful for tests). If None, uses sys.argv.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)

    dataset = CsvAudioDatasetLoader(
        csv_path=args.csv.resolve(),
        root=args.root.resolve(),
        filepath_col=str(args.filepath_col),
        chunk_seconds=float(args.chunk_seconds),
        loader=SoundFileWaveformLoader(target_sr=DEFAULT_TARGET_SR),
    )
    filepaths = list(dataset.filepaths)
    encoder = OnnxEncoder(
        model_path=args.model.resolve(),
        providers=list(args.providers),
        output_name=str(args.output_name),
    )

    embeddings = extract_embeddings(
        waveforms=dataset.iter_waveforms(),
        encoder=encoder,
        batch_size=int(args.batch_size),
    )

    indexer = FaissInnerProductIndexer()
    neighbors = indexer.neighbors(embeddings, topk=int(args.k))

    write_submission_csv(args.out, filepaths=filepaths, neighbors=neighbors)
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
