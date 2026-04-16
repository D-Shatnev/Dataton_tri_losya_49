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
import torch

from dataton_tri_losya_49.pipeline.components.encoders import OnnxEncoder
from dataton_tri_losya_49.pipeline.components.encoders import SpeechBrainEncoder
from dataton_tri_losya_49.pipeline.components.encoders import WavLMEncoder
from dataton_tri_losya_49.pipeline.components.evaluators import PrecisionAtKEvaluator
from dataton_tri_losya_49.pipeline.components.indexers import FaissInnerProductIndexer
from dataton_tri_losya_49.pipeline.components.loaders import CsvAudioDatasetLoader, SoundFileWaveformLoader
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


def onnx_auto_providers() -> list[str]:
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


def torch_auto_providers() -> list[str]:
    """
    Return a devices list for speechbrain model inference based on machine hardware.
    possible providers:
        - 'cuda' if torch.cuda.is_available()
        - 'mps' if torch.backends.mps.is_available()
        - 'cpu' - always

    Returns:
        List of devices strings to run speechbrain model on.
    """
    available = []
    if torch.cuda.is_available():
        available.append("cuda")
    if torch.backends.mps.is_available():
        available.append("mps")
    available.append("cpu")

    return available


def resolve_providers(section_type: str, providers: list[str] | None) -> list[str]:
    """
    Resolve providers list: if None → auto-detect, otherwise use as-is.

    Args:
        section_type: name of some section variant.
        providers: Explicit list or None.

    Returns:
        Non-empty providers list.
    """
    if section_type == "onnx":
        if providers is None:
            return onnx_auto_providers()
    elif section_type == "speechbrain":
        if providers is None:
            return torch_auto_providers()
    elif section_type == "wavlm":
        if providers is None:
            return torch_auto_providers()
    else:
        raise ValueError(f"Not implemented providers resolve case for {section_type}")

    return list(providers)


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------


def build_encoder(
    section: EncoderSection,
    providers: list[str] | None = None,
    model_name_override: str | None = None,
    model_path_override: Path | None = None,
    save_dir_override: Path | None = None,
) -> Encoder:
    """
    Instantiate an :class:~dataton_tri_losya_49.pipeline.interfaces.Encoder from config section.

    Args:
        section: EncoderSection dataclass instance.
        providers: Explicit providers list. If None, section.providers is used;
            if that is also None, :func:auto_providers is called.
        model_path_override: If given, overrides section.model_path.
            Used by inference CLI so that --model flag wins over TOML default.
        save_dir_override: If given, overrides section.save_dir.
            Used by inference CLI so that --save-dir flag wins over TOML default.

    Returns:
        Encoder instance matching section.type.

    Raises:
        ValueError: If section.type is unknown.
        FileNotFoundError: If the resolved model path does not exist (raised by OnnxEncoder).

    Supported types:
        - "onnx" → :class:~dataton_tri_losya_49.pipeline.components.encoders.OnnxEncoder
    """
    if section.type == "onnx":
        effective_providers = providers if providers is not None else resolve_providers(section.type, section.providers)
        effective_path = resolve_path(model_path_override if model_path_override is not None else section.model_path)
        return OnnxEncoder(
            model_path=effective_path,
            providers=effective_providers,
            output_name=section.output_name,
        )
    elif section.type == "speechbrain":
        effective_providers = providers if providers is not None else resolve_providers(section.type, section.providers)
        effective_dir = resolve_path(save_dir_override if save_dir_override is not None else section.save_dir)
        return SpeechBrainEncoder(
            save_dir=effective_dir, providers=effective_providers, output_name=section.output_name
        )
    elif section.type == "wavlm":
        effective_providers = providers if providers is not None else resolve_providers(section.type, section.providers)
        effective_dir = resolve_path(save_dir_override if save_dir_override is not None else section.save_dir)
        effective_model_name = model_name_override if model_name_override is not None else section.model_name
        if effective_model_name is None:
            raise ValueError("For Wavlm encoder it's nesessary to choose model_name")
        return WavLMEncoder(
            save_dir=effective_dir, 
            model_name=effective_model_name,
            providers=effective_providers, 
            output_name=section.output_name
        )

    raise ValueError(
        f"Unknown encoder type: {section.type!r}. Register a new encoder in pipeline/registry.py :: build_encoder()."
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

    return CsvAudioDatasetLoader(
        csv_path=resolve_path(data_section.csv),
        root=resolve_path(data_section.root),
        filepath_col=data_section.filepath_col,
        speaker_id_col=data_section.speaker_id_col,
        chunk_seconds=float(data_section.chunk_seconds),
        loader=waveform_loader,
    )
