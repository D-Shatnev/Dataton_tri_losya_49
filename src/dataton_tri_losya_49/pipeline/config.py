"""
Experiment configuration for the modular speaker recognition pipeline.

This module defines a typed, validated configuration schema used to run reproducible
experiments via :mod:dataton_tri_losya_49.pipeline.runner.

The primary entrypoint is :func:load_experiment_config, which reads a TOML file
and returns an :class:ExperimentConfig dataclass.

TOML structure (high-level):

* [experiment]: experiment name and output directory
* [data]: input CSV with audio paths/labels and preprocessing parameters
* [encoder]: embedding model settings (type, path, providers)
* [index]: nearest-neighbor backend and top-k
* [evaluation]: metric parameters and optional external labels

Notes:
    - Relative paths are interpreted relative to the current working directory
      (repo root in our typical workflow).
    - Only a subset of backends may be supported depending on the current
      iteration (see :func:_validate_config).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataton_tri_losya_49.constants import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FILEPATH_COL,
    DEFAULT_SPEAKER_ID_COL,
)


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
class EncoderSection:
    """Encoder (embedder) configuration.

    Attributes:
        type: Encoder type identifier. For now only "onnx" is supported.
        model_path: Path to the model artifact.
        providers: ONNX Runtime providers priority list.
        output_name: Output node name used when extracting embeddings.
    """

    type: str
    model_path: Path
    providers: list[str]
    output_name: str = "embeddings"


@dataclass(frozen=True)
class IndexSection:
    """Nearest-neighbor index configuration."""

    topk: int = 10
    backend: str = "faiss_ip"


@dataclass(frozen=True)
class EvaluationSection:
    """Evaluation settings.

    Attributes:
        ks: A list of k values for metrics like Precision@k.
        labels_npy: Optional path to external labels if CSV does not contain
            a speaker-id column.
    """

    ks: list[int]
    labels_npy: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Root config object used by :func:dataton_tri_losya_49.pipeline.runner.run_experiment."""

    experiment: ExperimentSection
    data: DataSection
    encoder: EncoderSection
    index: IndexSection
    evaluation: EvaluationSection

    @property
    def run_dir(self) -> Path:
        """Directory where artifacts of this experiment run should be stored.

        It is computed as experiment.out_dir / experiment.name.
        """
        return self.experiment.out_dir / self.experiment.name


def _require(d: dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing key '{key}' in section [{section}]")
    return d[key]


def load_experiment_config(path: Path) -> ExperimentConfig:
    """
    Load and validate an experiment TOML config.

    All relative paths in config are resolved against the repo root (current working directory).

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
    idx_raw = raw.get("index", {})
    eval_raw = raw.get("evaluation", {})

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
            model_path=Path(str(_require(enc_raw, "model_path", "encoder"))),
            providers=[str(x) for x in _require(enc_raw, "providers", "encoder")],
            output_name=str(enc_raw.get("output_name", "embeddings")),
        ),
        index=IndexSection(
            topk=int(idx_raw.get("topk", 10)),
            backend=str(idx_raw.get("backend", "faiss_ip")),
        ),
        evaluation=EvaluationSection(
            ks=[int(x) for x in eval_raw.get("ks", [10])],
            labels_npy=(Path(str(eval_raw["labels_npy"])) if "labels_npy" in eval_raw else None),
        ),
    )

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ExperimentConfig) -> None:
    """
    Validate basic config invariants.

    This function intentionally does **not** validate that a particular
    encoder.type or index.backend is supported.

    Supported implementations are enforced in factories
    (:func:dataton_tri_losya_49.pipeline.runner.make_encoder,
    :func:dataton_tri_losya_49.pipeline.runner.make_indexer).

    Keeping this validation backend-agnostic makes it easier to add new
    encoders/indexers without touching multiple modules.
    """

    if not str(cfg.encoder.type).strip():
        raise ValueError("encoder.type must be a non-empty string")

    if not str(cfg.index.backend).strip():
        raise ValueError("index.backend must be a non-empty string")

    if cfg.index.topk <= 0:
        raise ValueError("index.topk must be > 0")

    if len(cfg.encoder.providers) == 0:
        raise ValueError("encoder.providers must be non-empty")

    if any(k <= 0 for k in cfg.evaluation.ks):
        raise ValueError("evaluation.ks must contain only positive integers")

    if cfg.data.chunk_seconds <= 0:
        raise ValueError("data.chunk_seconds must be > 0")
