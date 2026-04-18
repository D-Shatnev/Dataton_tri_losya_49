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
        type: Encoder type identifier (e.g. "onnx", "espnet").
            See :func:~dataton_tri_losya_49.pipeline.registry.build_encoder for supported values.
        model_path: Path to the model artifact. Required for type="onnx".
            For type="espnet" either model_path or model_tag must be set.
        model_tag: HuggingFace / ESPnet model tag (e.g. "espnet/voxcelebs12_ecapa_wavlm_joint").
            Used by type="espnet" when loading from a pretrained hub model.
        output_name: Output node name used when extracting embeddings (ONNX only).
        providers: ONNX Runtime providers priority list.
            None (default) means auto-detect at runtime:
            CUDA if available, CPU otherwise.
    """

    type: str
    model_path: Path | None = None
    model_tag: str | None = None
    output_name: str = "embeddings"
    providers: list[str] | None = None


@dataclass(frozen=True)
class LoaderSection:
    """
    Waveform loader configuration.

    Attributes:
        type: Loader type identifier (e.g. "soundfile").
            See :func:~dataton_tri_losya_49.pipeline.registry.build_waveform_loader for supported values.
        target_sr: Target sample rate in Hz. Audio will be resampled if necessary.
        clip: If True, clip waveform values to [-1, 1].
    """

    type: str = "soundfile"
    target_sr: int = DEFAULT_TARGET_SR
    clip: bool = False


@dataclass(frozen=True)
class IndexSection:
    """Nearest-neighbor index configuration."""

    topk: int = 10
    backend: str = "faiss_ip"


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
            model_path=Path(str(enc_raw["model_path"])) if "model_path" in enc_raw else None,
            model_tag=str(enc_raw["model_tag"]) if "model_tag" in enc_raw else None,
            output_name=str(enc_raw.get("output_name", "embeddings")),
            providers=providers,
        ),
        loader=LoaderSection(
            type=str(ldr_raw.get("type", "soundfile")),
            target_sr=int(ldr_raw.get("target_sr", DEFAULT_TARGET_SR)),
            clip=bool(ldr_raw.get("clip", False)),
        ),
        index=IndexSection(
            topk=int(idx_raw.get("topk", 10)),
            backend=str(idx_raw.get("backend", "faiss_ip")),
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
            model_path=Path(str(enc_raw["model_path"])) if "model_path" in enc_raw else None,
            model_tag=str(enc_raw["model_tag"]) if "model_tag" in enc_raw else None,
            output_name=str(enc_raw.get("output_name", "embeddings")),
            providers=providers,
        ),
        loader=LoaderSection(
            type=str(ldr_raw.get("type", "soundfile")),
            target_sr=int(ldr_raw.get("target_sr", DEFAULT_TARGET_SR)),
            clip=bool(ldr_raw.get("clip", False)),
        ),
        index=IndexSection(
            topk=int(idx_raw.get("topk", 10)),
            backend=str(idx_raw.get("backend", "faiss_ip")),
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


def _validate_encoder_section(enc: "EncoderSection") -> None:
    """
    Validate encoder section invariants shared across config types.

    Args:
        enc: EncoderSection to validate.

    Raises:
        ValueError: If encoder configuration is invalid.
    """
    if not str(enc.type).strip():
        raise ValueError("encoder.type must be a non-empty string")

    if enc.providers is not None and len(enc.providers) == 0:
        raise ValueError("encoder.providers must be non-empty when explicitly specified")

    if enc.type == "onnx":
        if enc.model_path is None:
            raise ValueError("encoder.model_path is required for type='onnx'")

    if enc.type == "espnet":
        if enc.model_tag is None and enc.model_path is None:
            raise ValueError("encoder: model_tag or model_path is required for type='espnet'")


def _validate_experiment_config(cfg: ExperimentConfig) -> None:
    """
    Validate experiment config invariants.

    Checks generic constraints (non-empty strings, positive integers, etc.).
    Encoder type / indexer backend validity is enforced by the registry factories,
    not here - this separation keeps config schema stable.
    """
    _validate_encoder_section(cfg.encoder)

    if not str(cfg.index.backend).strip():
        raise ValueError("index.backend must be a non-empty string")

    if cfg.index.topk <= 0:
        raise ValueError("index.topk must be > 0")

    if any(k <= 0 for k in cfg.evaluation.ks):
        raise ValueError("evaluation.ks must contain only positive integers")

    if cfg.data.chunk_seconds <= 0:
        raise ValueError("data.chunk_seconds must be > 0")

    if not str(cfg.loader.type).strip():
        raise ValueError("loader.type must be a non-empty string")

    if cfg.loader.target_sr <= 0:
        raise ValueError("loader.target_sr must be > 0")


def _validate_inference_config(cfg: InferenceConfig) -> None:
    """Validate inference config invariants."""
    _validate_encoder_section(cfg.encoder)

    if not str(cfg.index.backend).strip():
        raise ValueError("index.backend must be a non-empty string")

    if cfg.index.topk <= 0:
        raise ValueError("index.topk must be > 0")

    if not str(cfg.loader.type).strip():
        raise ValueError("loader.type must be a non-empty string")

    if cfg.loader.target_sr <= 0:
        raise ValueError("loader.target_sr must be > 0")

    if cfg.defaults.chunk_seconds <= 0:
        raise ValueError("defaults.chunk_seconds must be > 0")

    if cfg.defaults.batch_size <= 0:
        raise ValueError("defaults.batch_size must be > 0")
