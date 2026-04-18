"""
Configuration schemas for the modular speaker recognition pipeline.

Two config flavors are supported:

* :class:ExperimentConfig - full dev-experiment config (all sections).
  Loaded by :func:load_experiment_config from a TOML file.
  Consumed by speakerid-experiment CLI and :func:~dataton_tri_losya_49.pipeline.runner.run_experiment.

* :class:InferenceConfig - lightweight inference config (no data paths, no evaluation).
  Loaded by :func:load_inference_config from a TOML file.
  Consumed by speakerid-infer CLI; file paths (csv / out / root) come from CLI args.

TOML structure for experiments (high-level):

* [experiment]: experiment name and output directory
* [data]: input CSV with audio paths/labels and preprocessing parameters
* [encoder]: embedding model settings (type, path, providers)
* [loader]: waveform loader type and parameters
* [index]: nearest-neighbor backend and top-k
* [evaluation]: metric type, k values, optional external labels

TOML structure for inference (high-level):

* [encoder]: embedding model settings (type, model_path default, output_name)
* [loader]: waveform loader type and parameters
* [index]: nearest-neighbor backend and top-k
* [defaults]: chunk_seconds, batch_size, filepath_col

Notes:
    - Relative paths are interpreted relative to the current working directory.
    - encoder.providers is optional; None (absent from TOML) means auto-detect
      at runtime via :func:~dataton_tri_losya_49.pipeline.registry.auto_providers.
    - [loader] and [evaluation] sections are optional in experiment configs
      and fall back to sensible defaults (soundfile / precision_at_k).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataton_tri_losya_49.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FILEPATH_COL,
    DEFAULT_SPEAKER_ID_COL,
    DEFAULT_TARGET_SR,
)

# ---------------------------------------------------------------------------
# Shared sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSection:
    """
    Encoder (embedder) configuration.

    Attributes:
        type: Encoder type identifier (e.g. "onnx", "redimnet").
            See :func:~dataton_tri_losya_49.pipeline.registry.build_encoder for supported values.
        model_path: Path to the model artifact. Required for type="onnx".
            Leave empty string for hub-based models (e.g. type="redimnet").
        output_name: Output node name used when extracting embeddings (onnx only).
        providers: ONNX Runtime providers priority list.
            None (default) means auto-detect at runtime:
            CUDA if available, CPU otherwise.
        hub_repo: torch.hub repository slug (redimnet only).
        model_name: Model size identifier, e.g. "b6" (redimnet only).
        train_type: Training regime, e.g. "ft_lm" (redimnet only).
        dataset: Pretraining dataset, e.g. "vox2" (redimnet only).
        device: Compute device for torch-based encoders. "auto" selects
            CUDA when available, else CPU (redimnet only).
        embedding_dim: Expected embedding dimensionality (redimnet only).
        force_reload: If True, bypass torch.hub cache (redimnet only).
    """

    type: str
    model_path: Path = Path("")
    output_name: str = "embeddings"
    providers: list[str] | None = None
    # ReDimNet-specific fields (ignored for type="onnx")
    hub_repo: str = "IDRnD/ReDimNet"
    model_name: str = "b6"
    train_type: str = "ft_lm"
    dataset: str = "vox2"
    device: str = "auto"
    embedding_dim: int = 192
    force_reload: bool = False


@dataclass(frozen=True)
class LoaderSection:
    """
    Waveform loader configuration.

    Attributes:
        type: Loader type identifier (e.g. "soundfile", "soundfile_vad").
            See :func:~dataton_tri_losya_49.pipeline.registry.build_waveform_loader for supported values.
        target_sr: Target sample rate in Hz. Audio will be resampled if necessary.
        clip: If True, clip waveform values to [-1, 1].
        vad_model_dir: Path to FireRedVAD model directory. Required when type="soundfile_vad".
        vad_use_gpu: Whether to run VAD inference on GPU. Only used when type="soundfile_vad".
        vad_speech_threshold: VAD probability threshold for speech detection.
            Only used when type="soundfile_vad".
        vad_min_speech_frame: Minimum consecutive frames to be labelled as speech.
            Only used when type="soundfile_vad".
    """

    type: str = "soundfile"
    target_sr: int = DEFAULT_TARGET_SR
    clip: bool = False
    vad_model_dir: Path | None = None
    vad_use_gpu: bool = False
    vad_speech_threshold: float = 0.4
    vad_min_speech_frame: int = 20


@dataclass(frozen=True)
class IndexSection:
    """Nearest-neighbor index configuration.

    Attributes:
        topk: Number of nearest neighbors to return per query.
        backend: Indexer backend identifier. See
            :func:~dataton_tri_losya_49.pipeline.registry.build_indexer for supported values.
        cohort_size: Number of randomly sampled embeddings used as the AS-Norm cohort.
            Only used when backend is "faiss_as_norm".
        top_n: Number of top cohort scores used to estimate normalization statistics
            (mean and std) per item. Must be <= cohort_size.
            Only used when backend is "faiss_as_norm".
        cohort_seed: Random seed for cohort sampling. Ensures reproducibility.
            Only used when backend is "faiss_as_norm".
        faiss_candidates: Number of candidates retrieved by FAISS before AS-Norm
            re-ranking. Must be >= topk. Recommended: 5-10x topk.
            Only used when backend is "faiss_as_norm".
    """

    topk: int = 10
    backend: str = "faiss_ip"
    cohort_size: int = 1000
    top_n: int = 200
    cohort_seed: int = 42
    faiss_candidates: int = 100


# ---------------------------------------------------------------------------
# Experiment-only sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSection:
    """Experiment identity and output location."""

    name: str
    out_dir: Path = Path("experiments")


@dataclass(frozen=True)
class DataSection:
    """Input data description (CSV + root directory) and deterministic chunking."""

    csv: Path
    root: Path = Path(".")
    filepath_col: str = DEFAULT_FILEPATH_COL
    speaker_id_col: str = DEFAULT_SPEAKER_ID_COL
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS


@dataclass(frozen=True)
class EvaluationSection:
    """
    Evaluation settings.

    Attributes:
        type: Evaluator type identifier (e.g. "precision_at_k").
            See :func:~dataton_tri_losya_49.pipeline.registry.build_evaluator for supported values.
        ks: A list of k values for metrics like Precision@k.
        labels_npy: Optional path to external labels if CSV does not contain
            a speaker-id column.
    """

    ks: list[int] = field(default_factory=lambda: [10])
    type: str = "precision_at_k"
    labels_npy: Path | None = None


# ---------------------------------------------------------------------------
# Inference-only section
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceDefaultsSection:
    """
    Default non-path parameters for the inference CLI.

    These values are baked into the inference TOML and can be overridden
    by CLI flags (--chunk-seconds, --batch-size, --filepath-col).

    Attributes:
        chunk_seconds: Fixed chunk length in seconds for audio normalization.
        batch_size: Number of waveforms per encoder call.
        filepath_col: CSV column that contains relative audio file paths.
    """

    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    filepath_col: str = DEFAULT_FILEPATH_COL


# ---------------------------------------------------------------------------
# Root config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Root config object used by :func:~dataton_tri_losya_49.pipeline.runner.run_experiment."""

    experiment: ExperimentSection
    data: DataSection
    encoder: EncoderSection
    index: IndexSection
    evaluation: EvaluationSection
    loader: LoaderSection = field(default_factory=LoaderSection)

    @property
    def run_dir(self) -> Path:
        """Directory where artifacts of this experiment run should be stored.

        It is computed as experiment.out_dir / experiment.name.
        """
        return self.experiment.out_dir / self.experiment.name


@dataclass(frozen=True)
class InferenceConfig:
    """
    Lightweight config for the inference CLI.

    Unlike :class:ExperimentConfig, this config does not contain data paths
    (CSV, output) or evaluation settings - those are provided via CLI flags.

    Attributes:
        encoder: Encoder type / default model path / output name.
        index: Indexer backend and top-k.
        loader: Waveform loader type and parameters.
        defaults: Non-path inference defaults (chunk_seconds, batch_size, filepath_col).
    """

    encoder: EncoderSection
    index: IndexSection
    loader: LoaderSection
    defaults: InferenceDefaultsSection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(d: dict[str, Any], key: str, section: str) -> Any:
    """
    Get a required key from a dict or raise a helpful error.

    Args:
        d: Raw section mapping (parsed from TOML).
        key: Required key.
        section: TOML section name (used only for error message).

    Returns:
        The value stored under key.

    Raises:
        ValueError: If key is absent in d.
    """
    if key not in d:
        raise ValueError(f"Missing key '{key}' in section [{section}]")
    return d[key]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_experiment_config(path: Path) -> ExperimentConfig:
    """
    Load and validate an experiment TOML config.

    The config is parsed into a typed dataclass tree and validated by
    :func:_validate_experiment_config.

    Path handling:
        This function **does not** resolve paths to absolute.
        Resolution is performed later in :func:~dataton_tri_losya_49.pipeline.runner.run_experiment.

    Args:
        path: Path to a TOML file.

    Returns:
        Parsed and validated :class:ExperimentConfig.

    Raises:
        ValueError: If required keys are missing or configuration values are invalid.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    exp_raw = raw.get("experiment", {})
    data_raw = raw.get("data", {})
    enc_raw = raw.get("encoder", {})
    ldr_raw = raw.get("loader", {})
    idx_raw = raw.get("index", {})
    eval_raw = raw.get("evaluation", {})

    # providers: absent or null in TOML -> None (auto-detect)
    raw_providers = enc_raw.get("providers", None)
    providers: list[str] | None = [str(x) for x in raw_providers] if raw_providers is not None else None

    cfg = ExperimentConfig(
        experiment=ExperimentSection(
            name=str(_require(exp_raw, "name", "experiment")),
            out_dir=Path(str(exp_raw.get("out_dir", "experiments"))),
        ),
        data=DataSection(
            csv=Path(str(_require(data_raw, "csv", "data"))),
            root=Path(str(data_raw.get("root", "."))),
            filepath_col=str(data_raw.get("filepath_col", DEFAULT_FILEPATH_COL)),
            speaker_id_col=str(data_raw.get("speaker_id_col", DEFAULT_SPEAKER_ID_COL)),
            chunk_seconds=float(data_raw.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)),
        ),
        encoder=EncoderSection(
            type=str(_require(enc_raw, "type", "encoder")),
            model_path=Path(str(enc_raw.get("model_path", ""))),
            output_name=str(enc_raw.get("output_name", "embeddings")),
            providers=providers,
            hub_repo=str(enc_raw.get("hub_repo", "IDRnD/ReDimNet")),
            model_name=str(enc_raw.get("model_name", "b6")),
            train_type=str(enc_raw.get("train_type", "ft_lm")),
            dataset=str(enc_raw.get("dataset", "vox2")),
            device=str(enc_raw.get("device", "auto")),
            embedding_dim=int(enc_raw.get("embedding_dim", 192)),
            force_reload=bool(enc_raw.get("force_reload", False)),
        ),
        loader=LoaderSection(
            type=str(ldr_raw.get("type", "soundfile")),
            target_sr=int(ldr_raw.get("target_sr", DEFAULT_TARGET_SR)),
            clip=bool(ldr_raw.get("clip", False)),
            vad_model_dir=(Path(str(ldr_raw["vad_model_dir"])) if "vad_model_dir" in ldr_raw else None),
            vad_use_gpu=bool(ldr_raw.get("vad_use_gpu", False)),
            vad_speech_threshold=float(ldr_raw.get("vad_speech_threshold", 0.4)),
            vad_min_speech_frame=int(ldr_raw.get("vad_min_speech_frame", 20)),
        ),
        index=IndexSection(
            topk=int(idx_raw.get("topk", 10)),
            backend=str(idx_raw.get("backend", "faiss_ip")),
            cohort_size=int(idx_raw.get("cohort_size", 1000)),
            top_n=int(idx_raw.get("top_n", 200)),
            cohort_seed=int(idx_raw.get("cohort_seed", 42)),
            faiss_candidates=int(idx_raw.get("faiss_candidates", 100)),
        ),
        evaluation=EvaluationSection(
            type=str(eval_raw.get("type", "precision_at_k")),
            ks=[int(x) for x in eval_raw.get("ks", [10])],
            labels_npy=(Path(str(eval_raw["labels_npy"])) if "labels_npy" in eval_raw else None),
        ),
    )

    _validate_experiment_config(cfg)
    return cfg


def load_inference_config(path: Path) -> InferenceConfig:
    """
    Load an inference TOML config.

    Inference config is simpler than experiment config: it describes
    which components to use and their non-path defaults. Data paths
    (csv, output) are provided separately via CLI.

    Args:
        path: Path to a TOML file (e.g. configs/inference/baseline.toml).

    Returns:
        Parsed :class:InferenceConfig.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    enc_raw = raw.get("encoder", {})
    ldr_raw = raw.get("loader", {})
    idx_raw = raw.get("index", {})
    def_raw = raw.get("defaults", {})

    raw_providers = enc_raw.get("providers", None)
    providers: list[str] | None = [str(x) for x in raw_providers] if raw_providers is not None else None

    cfg = InferenceConfig(
        encoder=EncoderSection(
            type=str(_require(enc_raw, "type", "encoder")),
            model_path=Path(str(enc_raw.get("model_path", ""))),
            output_name=str(enc_raw.get("output_name", "embeddings")),
            providers=providers,
            hub_repo=str(enc_raw.get("hub_repo", "IDRnD/ReDimNet")),
            model_name=str(enc_raw.get("model_name", "b6")),
            train_type=str(enc_raw.get("train_type", "ft_lm")),
            dataset=str(enc_raw.get("dataset", "vox2")),
            device=str(enc_raw.get("device", "auto")),
            embedding_dim=int(enc_raw.get("embedding_dim", 192)),
            force_reload=bool(enc_raw.get("force_reload", False)),
        ),
        loader=LoaderSection(
            type=str(ldr_raw.get("type", "soundfile")),
            target_sr=int(ldr_raw.get("target_sr", DEFAULT_TARGET_SR)),
            clip=bool(ldr_raw.get("clip", False)),
            vad_model_dir=(Path(str(ldr_raw["vad_model_dir"])) if "vad_model_dir" in ldr_raw else None),
            vad_use_gpu=bool(ldr_raw.get("vad_use_gpu", False)),
            vad_speech_threshold=float(ldr_raw.get("vad_speech_threshold", 0.4)),
            vad_min_speech_frame=int(ldr_raw.get("vad_min_speech_frame", 20)),
        ),
        index=IndexSection(
            topk=int(idx_raw.get("topk", 10)),
            backend=str(idx_raw.get("backend", "faiss_ip")),
            cohort_size=int(idx_raw.get("cohort_size", 1000)),
            top_n=int(idx_raw.get("top_n", 200)),
            cohort_seed=int(idx_raw.get("cohort_seed", 42)),
            faiss_candidates=int(idx_raw.get("faiss_candidates", 100)),
        ),
        defaults=InferenceDefaultsSection(
            chunk_seconds=float(def_raw.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)),
            batch_size=int(def_raw.get("batch_size", DEFAULT_BATCH_SIZE)),
            filepath_col=str(def_raw.get("filepath_col", DEFAULT_FILEPATH_COL)),
        ),
    )

    _validate_inference_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _validate_experiment_config(cfg: ExperimentConfig) -> None:
    """
    Validate experiment config invariants.

    Checks generic constraints (non-empty strings, positive integers, etc.).
    Encoder type / indexer backend validity is enforced by the registry factories,
    not here - this separation keeps config schema stable.
    """
    if not str(cfg.encoder.type).strip():
        raise ValueError("encoder.type must be a non-empty string")

    if cfg.encoder.type == "onnx" and not str(cfg.encoder.model_path).strip():
        raise ValueError("encoder.model_path must be set for type='onnx'")

    if not str(cfg.index.backend).strip():
        raise ValueError("index.backend must be a non-empty string")

    if cfg.index.topk <= 0:
        raise ValueError("index.topk must be > 0")

    # providers: None is valid (auto-detect); if given, must be non-empty
    if cfg.encoder.providers is not None and len(cfg.encoder.providers) == 0:
        raise ValueError("encoder.providers must be non-empty when explicitly specified")

    if any(k <= 0 for k in cfg.evaluation.ks):
        raise ValueError("evaluation.ks must contain only positive integers")

    if cfg.data.chunk_seconds <= 0:
        raise ValueError("data.chunk_seconds must be > 0")

    if not str(cfg.loader.type).strip():
        raise ValueError("loader.type must be a non-empty string")

    if cfg.loader.target_sr <= 0:
        raise ValueError("loader.target_sr must be > 0")

    _validate_loader_vad(cfg.loader)


def _validate_inference_config(cfg: InferenceConfig) -> None:
    """Validate inference config invariants."""

    if not str(cfg.encoder.type).strip():
        raise ValueError("encoder.type must be a non-empty string")

    if not str(cfg.index.backend).strip():
        raise ValueError("index.backend must be a non-empty string")

    if cfg.index.topk <= 0:
        raise ValueError("index.topk must be > 0")

    if cfg.encoder.providers is not None and len(cfg.encoder.providers) == 0:
        raise ValueError("encoder.providers must be non-empty when explicitly specified")

    if not str(cfg.loader.type).strip():
        raise ValueError("loader.type must be a non-empty string")

    if cfg.loader.target_sr <= 0:
        raise ValueError("loader.target_sr must be > 0")

    if cfg.defaults.chunk_seconds <= 0:
        raise ValueError("defaults.chunk_seconds must be > 0")

    if cfg.defaults.batch_size <= 0:
        raise ValueError("defaults.batch_size must be > 0")

    _validate_loader_vad(cfg.loader)


def _validate_loader_vad(loader: LoaderSection) -> None:
    """Validate VAD-specific loader constraints.

    Args:
        loader: LoaderSection to validate.

    Raises:
        ValueError: If loader.type is "soundfile_vad" but vad_model_dir is not set,
            or if vad_speech_threshold / vad_min_speech_frame are out of range.
    """
    if loader.type != "soundfile_vad":
        return

    if loader.vad_model_dir is None:
        raise ValueError("loader.vad_model_dir is required when loader.type = 'soundfile_vad'")

    if not 0.0 < loader.vad_speech_threshold < 1.0:
        raise ValueError("loader.vad_speech_threshold must be in (0, 1)")

    if loader.vad_min_speech_frame <= 0:
        raise ValueError("loader.vad_min_speech_frame must be > 0")
