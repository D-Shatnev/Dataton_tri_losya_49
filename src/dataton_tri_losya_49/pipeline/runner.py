"""
Experiment runner.

This module contains a small, modular pipeline to:

1) Load a dataset (CSV -> audio).
2) Extract speaker embeddings with a swappable encoder.
3) Build a kNN index and generate a challenge submission.
4) Optionally evaluate the submission if labels are available.

Component instantiation is fully delegated to
:mod:dataton_tri_losya_49.pipeline.registry so that adding a new encoder /
indexer / loader / evaluator requires **only** a TOML config change and (once)
a new branch in the registry — no changes to this runner needed.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from dataton_tri_losya_49.io import save_embeddings_npz, write_submission_csv
from dataton_tri_losya_49.pipeline.config import ExperimentConfig
from dataton_tri_losya_49.pipeline.interfaces import DatasetLoader, Encoder, Evaluator, Indexer
from dataton_tri_losya_49.pipeline.registry import build_dataset_loader, build_encoder, build_evaluator, build_indexer
from dataton_tri_losya_49.pipeline.utils import prefetch_chunk_batches, prefetch_iter, resolve_path


@dataclass(frozen=True)
class RunArtifacts:
    """
    Paths to artifacts produced by :func:run_experiment.

    Attributes:
        run_dir: Directory containing all artifacts for a single run.
        embeddings_path: .npz file with filepaths and embeddings.
        submission_path: submission.csv compatible with challenge format.
        metrics_path: Optional metrics.json with evaluator outputs.
        timing_path: timing.json with inference/search/total wall-clock times.
    """

    run_dir: Path
    embeddings_path: Path
    submission_path: Path
    metrics_path: Path | None
    timing_path: Path


@dataclass(frozen=True)
class Components:
    """All pipeline components required to run an experiment."""

    encoder: Encoder
    indexer: Indexer
    evaluator: Evaluator
    dataset: DatasetLoader


def build_components(cfg: ExperimentConfig) -> Components:
    """
    Construct all experiment components from config via registry.

    Args:
        cfg: Parsed :class:~dataton_tri_losya_49.pipeline.config.ExperimentConfig.

    Returns:
        :class:Components with all pipeline components instantiated and ready.
    """
    return Components(
        encoder=build_encoder(cfg.encoder),
        indexer=build_indexer(cfg.index),
        evaluator=build_evaluator(cfg.evaluation),
        dataset=build_dataset_loader(cfg.data, cfg.loader),
    )


def prepare_run_dir(cfg: ExperimentConfig, config_path: Path) -> Path:
    """Create run directory and copy config for reproducibility."""
    run_dir = resolve_path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.toml")
    return run_dir


def load_labels(dataset: DatasetLoader, cfg: ExperimentConfig) -> np.ndarray | None:
    """Resolve labels from dataset or from cfg.evaluation.labels_npy."""
    labels = dataset.labels
    if labels is None and cfg.evaluation.labels_npy is not None:
        labels = np.load(resolve_path(cfg.evaluation.labels_npy), allow_pickle=True)
    return labels


def write_metrics_json(path: Path, metrics: dict) -> None:
    """Write metrics as pretty JSON."""
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_waveform_iter(dataset: DatasetLoader, prefetch: int) -> Iterable[np.ndarray]:
    """Return a waveform iterator, optionally with background prefetching.

    When ``prefetch >= 1``, waveforms are loaded from disk in a background
    thread so that I/O overlaps with GPU inference.  The dataset loader's
    ``iter_waveforms`` is split into two stages:

    1. Iterating over file paths (cheap, done in the background thread).
    2. Calling ``loader.load`` for each path (I/O-bound, done in background).

    When the dataset loader does not expose a ``loader`` attribute (i.e. it
    does not follow the ``root / filepath → loader.load`` pattern), we fall
    back to wrapping ``iter_waveforms()`` directly with an identity prefetch,
    which still provides a read-ahead buffer of ``prefetch`` waveforms.

    Args:
        dataset: Dataset loader instance.
        prefetch: Number of waveforms to buffer ahead.  0 disables prefetching.

    Returns:
        Iterator of float32 waveform arrays.
    """
    if prefetch <= 0:
        return dataset.iter_waveforms()

    inner_loader = getattr(dataset, "loader", None)
    root = getattr(dataset, "root", None)

    if inner_loader is not None and root is not None and hasattr(inner_loader, "load"):
        # Preferred path: prefetch at the path→load boundary so I/O runs in
        # background while the encoder processes the previous waveform.
        filepaths: Iterable[str] = getattr(dataset, "filepaths", [])
        return prefetch_iter(
            source=iter(filepaths),
            load_fn=lambda fp: inner_loader.load(root / fp),
            prefetch=prefetch,
        )

    # Fallback: wrap the existing iterator with a lookahead buffer.
    return prefetch_iter(
        source=dataset.iter_waveforms(),
        load_fn=lambda wav: wav,
        prefetch=prefetch,
    )


def build_timing(dataset: DatasetLoader, inference_time_s: float, search_time_s: float) -> dict:
    """
    Build timing dict, optionally including VAD breakdown.

    Reads ``vad_time_s`` from the inner waveform loader if it exposes that
    attribute (i.e. when :class:`~dataton_tri_losya_49.pipeline.components.loaders.VadWaveformLoader`
    is used). Falls back to 0.0 so the function is safe for any loader type.

    Args:
        dataset: Dataset loader instance (may or may not have a ``loader`` attribute).
        inference_time_s: Total wall-clock time spent in :func:`extract_embeddings`.
        search_time_s: Wall-clock time spent in the kNN search step.

    Returns:
        JSON-serialisable timing dict with keys ``inference_time_s``,
        ``encoder_time_s``, ``search_time_s``, ``total_time_s`` and,
        when VAD was used, ``vad_time_s``.
    """
    _inner_loader = getattr(dataset, "loader", None)
    vad_time_s: float = getattr(_inner_loader, "vad_time_s", 0.0)
    encoder_time_s = inference_time_s - vad_time_s

    timing: dict = {
        "inference_time_s": round(inference_time_s, 6),
        "encoder_time_s": round(encoder_time_s, 6),
        "search_time_s": round(search_time_s, 6),
        "total_time_s": round(inference_time_s + search_time_s, 6),
    }
    if vad_time_s > 0.0:
        timing["vad_time_s"] = round(vad_time_s, 6)
    return timing


def run_experiment(
    cfg: ExperimentConfig,
    config_path: Path,
    batch_size: int = 1,
    prefetch: int = 8,
) -> RunArtifacts:
    """
    Run inference -> neighbors -> (optional) metrics.

    The experiment run is fully defined by a TOML config. This function:

    1) Creates a run directory and copies the config there.
    2) Builds pipeline components via registry (encoder/indexer/evaluator/dataset loader).
    3) Extracts embeddings for all dataset items.
    4) Runs kNN retrieval to produce a submission.
    5) Optionally computes metrics if labels are available.

    When the encoder is a :class:`~dataton_tri_losya_49.pipeline.components.encoders.chunking_encoder.ChunkingEncoder`,
    uses :func:`extract_embeddings_chunked` which pre-chunks waveforms in a
    background thread and feeds the GPU a continuous stream of chunk batches,
    keeping GPU utilisation high.  Falls back to :func:`extract_embeddings`
    for non-chunking encoders.

    Args:
        cfg: Parsed experiment configuration.
        config_path: Path to the original TOML file (will be copied into run_dir).
        batch_size: Number of waveforms per GPU batch.  For chunking encoders
            this controls how many files are chunked together into one GPU call.
        prefetch: Number of pre-chunked batches to buffer in the background
            thread.  Set to 0 to disable prefetching (useful for debugging).
            Default is 8.

    Returns:
        :class:RunArtifacts with paths to all produced files.

    Notes:
      - Metrics are calculated if labels are available either from CSV
        speaker_id column or from evaluation.labels_npy.
      - Component types are resolved via registry:
        see :mod:dataton_tri_losya_49.pipeline.registry.
    """
    run_dir = prepare_run_dir(cfg, config_path=config_path)

    components = build_components(cfg)
    filepaths = list(components.dataset.filepaths)

    if len(filepaths) < 2:
        raise ValueError(f"Dataset must contain at least 2 items to build a kNN submission (got {len(filepaths)}).")

    _t0_inference = time.perf_counter()

    from dataton_tri_losya_49.pipeline.components.encoders.chunking_encoder import ChunkingEncoder

    if isinstance(components.encoder, ChunkingEncoder):
        # Fast path: pre-chunk in background thread, GPU never waits for I/O.
        waveforms = _make_waveform_iter(components.dataset, prefetch=prefetch)
        embeddings = extract_embeddings_chunked(
            waveforms=waveforms,
            encoder=components.encoder,
            batch_size=batch_size,
            prefetch=max(prefetch, 1),
            total=len(filepaths),
        )
    else:
        waveforms = _make_waveform_iter(components.dataset, prefetch=prefetch)
        embeddings = extract_embeddings(
            waveforms=waveforms,
            encoder=components.encoder,
            batch_size=batch_size,
            total=len(filepaths),
        )

    inference_time_s = time.perf_counter() - _t0_inference

    embeddings_path = run_dir / "embeddings.npz"
    save_embeddings_npz(embeddings_path, filepaths=filepaths, embeddings=embeddings)

    _t0_search = time.perf_counter()
    neighbors = components.indexer.neighbors(embeddings, topk=cfg.index.topk)
    search_time_s = time.perf_counter() - _t0_search

    submission_path = run_dir / "submission.csv"
    write_submission_csv(submission_path, filepaths=filepaths, neighbors=neighbors)

    timing = build_timing(components.dataset, inference_time_s, search_time_s)
    timing_path = run_dir / "timing.json"
    write_metrics_json(timing_path, timing)

    metrics_path: Path | None = None
    labels = load_labels(components.dataset, cfg)
    if labels is not None:
        metrics = components.evaluator.evaluate(neighbors=neighbors, labels=labels, ks=cfg.evaluation.ks)
        metrics["timing"] = timing
        metrics_path = run_dir / "metrics.json"
        write_metrics_json(metrics_path, metrics)

    return RunArtifacts(
        run_dir=run_dir,
        embeddings_path=embeddings_path,
        submission_path=submission_path,
        metrics_path=metrics_path,
        timing_path=timing_path,
    )


def extract_embeddings_chunked(
    waveforms: Iterable[np.ndarray],
    encoder: Any,
    batch_size: int,
    prefetch: int = 4,
    total: int | None = None,
) -> np.ndarray:
    """
    Extract embeddings using a ChunkingEncoder with background pre-chunking.

    Waveforms are chunked in a background thread via :func:`prefetch_chunk_batches`
    so that CPU work (I/O + chunking) overlaps with GPU inference.  The GPU
    receives a continuous stream of pre-chunked batches and is never blocked
    waiting for the next file to load.

    When ``encoder.encoder`` exposes a ``max_audio_duration_s > 0`` attribute
    (set via :class:`~dataton_tri_losya_49.pipeline.config.EncoderSection`),
    the padded tensor height is computed upfront so ``torch.compile`` always
    sees the same shape and never recompiles mid-run.

    Args:
        waveforms: Iterable of 1-D float32 waveform arrays (one per file).
        encoder: A :class:`~dataton_tri_losya_49.pipeline.components.encoders.chunking_encoder.ChunkingEncoder`
            instance.  Its ``chunk_duration_s``, ``chunk_overlap_s`` and
            ``sample_rate`` attributes are used to compute chunking parameters.
        batch_size: Number of waveforms to accumulate per GPU batch.
        prefetch: Number of pre-chunked batches to buffer in the background
            thread.  Must be >= 1.
        total: Total number of waveforms (used for tqdm progress bar).

    Returns:
        float32 embeddings array shaped [N, D].
    """
    chunk_samples = int(encoder.chunk_duration_s * encoder.sample_rate)
    hop_samples = int((encoder.chunk_duration_s - encoder.chunk_overlap_s) * encoder.sample_rate)

    # Shape stabilisation is handled inside ChunkingEncoder._embed_chunks:
    # each sub-batch is zero-padded to exactly max_chunk_batch rows so that
    # torch.compile always sees the same tensor shape.  We therefore pass
    # padded_total=0 here (no extra padding at the batch level).
    batches = prefetch_chunk_batches(
        waveforms=iter(waveforms),
        chunk_samples=chunk_samples,
        hop_samples=hop_samples,
        batch_size=batch_size,
        prefetch=prefetch,
        padded_total=0,
    )

    outs: list[np.ndarray] = []
    pbar = tqdm(total=total, desc="inference", unit="wav", dynamic_ncols=True)

    for combined_chunks, chunk_counts in batches:
        # combined_chunks: [total_chunks, chunk_samples].
        # _embed_chunks splits into sub-batches of max_chunk_batch rows, padding
        # each to a fixed size so torch.compile always sees the same shape.
        all_embs = encoder._embed_chunks(combined_chunks)  # [total_chunks, D]

        # Split back per waveform and average.
        offset = 0
        for n in chunk_counts:
            outs.append(all_embs[offset : offset + n].mean(axis=0))
            offset += n

        pbar.update(len(chunk_counts))

    pbar.close()

    if not outs:
        return np.empty((0, 0), dtype=np.float32)

    return np.stack(outs, axis=0)


def extract_embeddings(
    waveforms: Iterable[np.ndarray],
    encoder: Encoder,
    batch_size: int,
    total: int | None = None,
) -> np.ndarray:
    """
    Extract embeddings for an iterable of waveforms.

    This is a small batching utility shared by CLI and experiment pipeline.
    It pads waveforms inside each batch to the maximum length in that batch.

    Args:
        waveforms: Iterable that yields 1-D float32 waveforms.
        encoder: Encoder component.
        batch_size: Number of waveforms per encoder call.
        total: Total number of waveforms (used for tqdm progress bar).
            Pass None to show a spinner without percentage.

    Returns:
        float32 embeddings array shaped [N, D].
    """
    waveforms = tqdm(waveforms, total=total, desc="inference", unit="wav", dynamic_ncols=True)
    batch_waves: list[np.ndarray] = []
    outs: list[np.ndarray] = []

    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be > 0")

    for wav in waveforms:
        batch_waves.append(wav)
        if len(batch_waves) == bs:
            outs.append(_embed_batch(batch_waves, encoder=encoder))
            batch_waves = []

    if batch_waves:
        outs.append(_embed_batch(batch_waves, encoder=encoder))

    # Handle empty input gracefully.
    if not outs:
        # We don't know the embedding dimensionality without calling the encoder,
        # so return an empty [0, 0] matrix.
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(outs, axis=0)


def _embed_batch(waves: list[np.ndarray], encoder: Encoder) -> np.ndarray:
    """
    Pad a list of waveforms and run encoder once.

    Args:
        waves: List of 1-D waveforms.
        encoder: Encoder component.

    Returns:
        float32 embeddings shaped [B, D].
    """
    if not waves:
        raise ValueError("waves must be non-empty")
    max_len = max(int(w.shape[0]) for w in waves)
    batch = np.zeros((len(waves), max_len), dtype=np.float32)
    for i, w in enumerate(waves):
        w = np.asarray(w, dtype=np.float32)
        batch[i, : w.shape[0]] = w
    return encoder.embed(batch)
