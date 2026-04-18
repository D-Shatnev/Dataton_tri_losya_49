"""
Component registry for the modular speaker recognition pipeline.

This module is the **single source of truth** for wiring component type-strings
(as written in TOML configs) to concrete Python classes.

Both inference and experiment flows use this registry to build components,
so swapping an encoder / indexer / loader / evaluator across experiments
requires only a TOML change (and, once, registering the new class here).

Usage example (experiment runner)::

    from dataton_tri_losya_49.pipeline.registry import build_encoder, build_indexer
    encoder = build_encoder(cfg.encoder)
    indexer = build_indexer(cfg.index)

Usage example (inference CLI)::

    from dataton_tri_losya_49.pipeline.registry import auto_providers, build_encoder
    providers = auto_providers()
    encoder = build_encoder(cfg.encoder, providers=providers)

Auto-provider detection
-----------------------
auto_providers() inspects onnxruntime.get_available_providers() at
runtime and returns ['CUDAExecutionProvider', 'CPUExecutionProvider'] if
CUDA is available, or ['CPUExecutionProvider'] otherwise.

Adding a new encoder
--------------------
1. Create a class in pipeline/components/encoders/ implementing
   :class:~dataton_tri_losya_49.pipeline.interfaces.Encoder.
2. Add an elif section.type == "your_type": branch in :func:build_encoder.
3. Write a TOML config with [encoder] type = "your_type".
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

from dataton_tri_losya_49.pipeline.components.encoders import OnnxEncoder, ReDimNetEncoder
from dataton_tri_losya_49.pipeline.components.evaluators import PrecisionAtKEvaluator
from dataton_tri_losya_49.pipeline.components.indexers import FaissASNormIndexer, FaissInnerProductIndexer
from dataton_tri_losya_49.pipeline.components.loaders import (
    CsvAudioDatasetLoader,
    PrefetchDatasetLoader,
    SoundFileWaveformLoader,
    VadWaveformLoader,
)
from dataton_tri_losya_49.pipeline.config import (
    DataSection,
    EncoderSection,
    EvaluationSection,
    IndexSection,
    LoaderSection,
)
from dataton_tri_losya_49.pipeline.interfaces import DatasetLoader, Encoder, Evaluator, Indexer, WaveformLoader
from dataton_tri_losya_49.pipeline.utils import resolve_path

# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------


def auto_providers() -> list[str]:
    """
    Return an ONNX Runtime providers list based on runtime availability.

    Checks onnxruntime.get_available_providers() and returns:

    - ['CUDAExecutionProvider', 'CPUExecutionProvider'] — if CUDA is available.
    - ['CPUExecutionProvider'] — otherwise.

    Returns:
        List of ONNX Runtime provider strings.
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def resolve_providers(providers: list[str] | None) -> list[str]:
    """
    Resolve providers list: if None → auto-detect, otherwise use as-is.

    Args:
        providers: Explicit list or None.

    Returns:
        Non-empty providers list.
    """
    if providers is None:
        return auto_providers()
    return list(providers)


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------


def build_encoder(
    section: EncoderSection, providers: list[str] | None = None, model_path_override: Path | None = None
) -> Encoder:
    """
    Instantiate an :class:~dataton_tri_losya_49.pipeline.interfaces.Encoder from config section.

    Args:
        section: EncoderSection dataclass instance.
        providers: Explicit providers list. If None, section.providers is used;
            if that is also None, :func:auto_providers is called.
        model_path_override: If given, overrides section.model_path.
            Used by inference CLI so that --model flag wins over TOML default.

    Returns:
        Encoder instance matching section.type.

    Raises:
        ValueError: If section.type is unknown.
        FileNotFoundError: If the resolved model path does not exist (raised by OnnxEncoder).

    Supported types:
        - "onnx" → :class:~dataton_tri_losya_49.pipeline.components.encoders.OnnxEncoder
    """
    if section.type == "onnx":
        effective_providers = providers if providers is not None else resolve_providers(section.providers)
        effective_path = resolve_path(model_path_override if model_path_override is not None else section.model_path)
        return OnnxEncoder(
            model_path=effective_path,
            providers=effective_providers,
            output_name=section.output_name,
        )

    if section.type == "redimnet":
        return ReDimNetEncoder(
            hub_repo=section.hub_repo,
            model_name=section.model_name,
            train_type=section.train_type,
            dataset=section.dataset,
            device=section.device,
            embedding_dim=section.embedding_dim,
            force_reload=section.force_reload,
        )

    raise ValueError(
        f"Unknown encoder type: {section.type!r}. " "Register a new encoder in pipeline/registry.py :: build_encoder()."
    )


# ---------------------------------------------------------------------------
# Indexer factory
# ---------------------------------------------------------------------------


def build_indexer(section: IndexSection) -> Indexer:
    """
    Instantiate an :class:~dataton_tri_losya_49.pipeline.interfaces.Indexer from config section.

    Args:
        section: IndexSection dataclass instance.

    Returns:
        Indexer instance matching section.backend.

    Raises:
        ValueError: If section.backend is unknown.

    Supported backends:
        - "faiss_ip" → :class:~dataton_tri_losya_49.pipeline.components.indexers.FaissInnerProductIndexer
    """
    if section.backend == "faiss_ip":
        return FaissInnerProductIndexer()

    if section.backend == "faiss_as_norm":
        return FaissASNormIndexer(
            cohort_size=section.cohort_size,
            top_n=section.top_n,
            cohort_seed=section.cohort_seed,
            faiss_candidates=section.faiss_candidates,
        )

    raise ValueError(
        f"Unknown indexer backend: {section.backend!r}. "
        "Register a new indexer in pipeline/registry.py :: build_indexer()."
    )


# ---------------------------------------------------------------------------
# Waveform loader factory
# ---------------------------------------------------------------------------


def build_waveform_loader(section: LoaderSection) -> WaveformLoader:
    """
    Instantiate a :class:~dataton_tri_losya_49.pipeline.interfaces.WaveformLoader from config section.

    Args:
        section: LoaderSection dataclass instance.

    Returns:
        WaveformLoader instance matching section.type.

    Raises:
        ValueError: If section.type is unknown.

    Supported types:
        - "soundfile" → :class:~dataton_tri_losya_49.pipeline.components.loaders.SoundFileWaveformLoader
    """
    if section.type == "soundfile":
        return SoundFileWaveformLoader(target_sr=section.target_sr, clip=section.clip)

    if section.type == "soundfile_vad":
        if section.vad_model_dir is None:
            raise ValueError(
                "loader.vad_model_dir is required when loader.type = 'soundfile_vad'"
            )
        return VadWaveformLoader(
            target_sr=section.target_sr,
            clip=section.clip,
            vad_model_dir=resolve_path(section.vad_model_dir),
            vad_use_gpu=section.vad_use_gpu,
            vad_speech_threshold=section.vad_speech_threshold,
            vad_min_speech_frame=section.vad_min_speech_frame,
        )

    raise ValueError(
        f"Unknown loader type: {section.type!r}. "
        "Register a new loader in pipeline/registry.py :: build_waveform_loader()."
    )


# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------


def build_evaluator(section: EvaluationSection) -> Evaluator:
    """
    Instantiate an :class:~dataton_tri_losya_49.pipeline.interfaces.Evaluator from config section.

    Args:
        section: EvaluationSection dataclass instance.

    Returns:
        Evaluator instance matching section.type.

    Raises:
        ValueError: If section.type is unknown.

    Supported types:
        - "precision_at_k" → :class:~dataton_tri_losya_49.pipeline.components.evaluators.PrecisionAtKEvaluator
    """
    if section.type == "precision_at_k":
        return PrecisionAtKEvaluator()

    raise ValueError(
        f"Unknown evaluator type: {section.type!r}. "
        "Register a new evaluator in pipeline/registry.py :: build_evaluator()."
    )


# ---------------------------------------------------------------------------
# Dataset loader factory (dataset is currently always CSV-backed)
# ---------------------------------------------------------------------------


def build_dataset_loader(
    data_section: DataSection,
    loader_section: LoaderSection,
) -> DatasetLoader:
    """
    Build a :class:~dataton_tri_losya_49.pipeline.interfaces.DatasetLoader.

    Dataset loading is currently always CSV-backed via
    :class:~dataton_tri_losya_49.pipeline.components.loaders.CsvAudioDatasetLoader.
    The waveform loader is selected via loader_section.type.

    Args:
        data_section: DataSection dataclass (csv / root / filepath_col / etc.).
        loader_section: LoaderSection dataclass (waveform loader type/params).

    Returns:
        Configured :class:~dataton_tri_losya_49.pipeline.interfaces.DatasetLoader.
    """
    waveform_loader = build_waveform_loader(loader_section)

    dataset = CsvAudioDatasetLoader(
        csv_path=resolve_path(data_section.csv),
        root=resolve_path(data_section.root),
        filepath_col=data_section.filepath_col,
        speaker_id_col=data_section.speaker_id_col,
        chunk_seconds=float(data_section.chunk_seconds),
        loader=waveform_loader,
    )

    if loader_section.prefetch_factor > 0:
        return PrefetchDatasetLoader(inner=dataset, prefetch_factor=loader_section.prefetch_factor)

    return dataset
