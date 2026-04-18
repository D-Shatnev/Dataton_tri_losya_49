"""
WeSpeaker ONNX encoder.

This module provides WeSpeakerOnnxEncoder - a wrapper around an ONNX
WeSpeaker model that handles the full PCM -> FBANK -> embedding pipeline.

WeSpeaker models expect Kaldi-compatible log-mel filterbank (FBANK) features
as input, not raw waveforms. This encoder uses kaldi-native-fbank to compute
features that exactly match the WeSpeaker training pipeline.

Input contract (same as OnnxEncoder):
    batch_waveforms: float32 array shaped [B, T], 16 kHz mono PCM.

Output contract:
    embeddings: float32 array shaped [B, D].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

from dataton_tri_losya_49.pipeline.config import WeSpeakerEncoderSection


@dataclass
class WeSpeakerOnnxEncoder:
    """
    Computes speaker embeddings using a WeSpeaker ONNX model.

    Internally performs:
      1. Per-waveform Kaldi-compatible FBANK feature extraction
         (via kaldi-native-fbank).
      2. Optional per-utterance CMVN (mean subtraction over time axis).
      3. Batch ONNX inference -> L2-normalised embeddings.

    Args:
        model_path: Path to the WeSpeaker ONNX model file.
        providers: ONNX Runtime execution providers list.
        output_name: Name of the ONNX output tensor containing embeddings.
        num_mel_bins: Number of mel filterbank bins.
        frame_length_ms: Analysis frame length in milliseconds.
        frame_shift_ms: Frame shift (hop) in milliseconds.
        low_freq: Lower frequency cutoff for mel filterbank in Hz.
        high_freq: Upper frequency cutoff in Hz. 0.0 -> Nyquist.
        apply_cmvn: If True, subtract per-utterance mean over time axis.
        sample_rate: Expected sample rate of input waveforms in Hz.

    Attributes:
        dim: Embedding dimensionality (D).
    """

    model_path: Path
    providers: list[str]
    output_name: str = "embs"
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    low_freq: float = 20.0
    high_freq: float = 0.0
    apply_cmvn: bool = True
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        """Initialise ONNX session and infer embedding dimensionality."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"WeSpeaker ONNX model not found: {self.model_path}")

        self._sess = ort.InferenceSession(str(self.model_path), providers=self.providers)

        inputs = self._sess.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected 1 ONNX input, got {len(inputs)}")
        self._input_name = inputs[0].name

        if self.output_name not in {o.name for o in self._sess.get_outputs()}:
            available = [o.name for o in self._sess.get_outputs()]
            raise ValueError(
                f"Unknown output_name={self.output_name!r}. Available outputs: {available}"
            )

        # Probe embedding dimensionality with a 3-second dummy waveform.
        probe_samples = int(3.0 * self.sample_rate)
        dummy_wav = np.zeros(probe_samples, dtype=np.float32)
        dummy_feats = self._extract_fbank(dummy_wav)          # (T_frames, num_mel_bins)
        dummy_batch = dummy_feats[np.newaxis, ...]             # (1, T_frames, num_mel_bins)
        out = self._sess.run([self.output_name], {self._input_name: dummy_batch})
        emb = np.asarray(out[0], dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {emb.shape}")
        self._dim = int(emb.shape[1])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimensionality D."""
        return self._dim

    def embed(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Compute embeddings for a batch of waveforms.

        Each waveform is processed independently through FBANK extraction
        (and optional CMVN), then the resulting feature tensors are stacked
        into a batch and passed through the ONNX model in a single call.

        Args:
            batch_waveforms: float32 array shaped [B, T].

        Returns:
            float32 embeddings shaped [B, D].
        """
        x = np.asarray(batch_waveforms, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected waveforms [B, T], got shape {x.shape}")

        # Extract features per waveform; pad to the longest sequence.
        feats_list = [self._extract_fbank(x[i]) for i in range(x.shape[0])]
        max_frames = max(f.shape[0] for f in feats_list)
        batch_feats = np.zeros(
            (len(feats_list), max_frames, self.num_mel_bins), dtype=np.float32
        )
        for i, f in enumerate(feats_list):
            batch_feats[i, : f.shape[0], :] = f

        out = self._sess.run([self.output_name], {self._input_name: batch_feats})
        emb = np.asarray(out[0], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != x.shape[0]:
            raise ValueError(
                f"Unexpected output shape {emb.shape} for input batch {x.shape}"
            )
        return emb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_fbank(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute Kaldi-compatible log-mel filterbank features for one waveform.

        Args:
            waveform: 1-D float32 PCM array at self.sample_rate Hz.

        Returns:
            float32 array shaped [T_frames, num_mel_bins].
        """
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = float(self.sample_rate)
        opts.frame_opts.frame_length_ms = self.frame_length_ms
        opts.frame_opts.frame_shift_ms = self.frame_shift_ms
        opts.frame_opts.dither = 0.0          # deterministic (no random dither)
        opts.mel_opts.num_bins = self.num_mel_bins
        opts.mel_opts.low_freq = self.low_freq
        opts.mel_opts.high_freq = (
            self.high_freq if self.high_freq > 0.0 else float(self.sample_rate) / 2.0
        )

        fbank = knf.OnlineFbank(opts)
        # kaldi-native-fbank expects int16-scaled float samples
        fbank.accept_waveform(float(self.sample_rate), (waveform * 32768.0).tolist())
        fbank.input_finished()

        num_frames = fbank.num_frames_ready
        if num_frames == 0:
            # Return a single zero-frame to avoid empty tensors.
            return np.zeros((1, self.num_mel_bins), dtype=np.float32)

        frames = np.stack(
            [np.array(fbank.get_frame(i), dtype=np.float32) for i in range(num_frames)]
        )  # (T_frames, num_mel_bins)

        if self.apply_cmvn:
            frames = frames - frames.mean(axis=0, keepdims=True)

        return frames

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        section: WeSpeakerEncoderSection,
        providers: list[str],
    ) -> WeSpeakerOnnxEncoder:
        """
        Construct a WeSpeakerOnnxEncoder from a config section.

        Args:
            section: Parsed :class:~dataton_tri_losya_49.pipeline.config.WeSpeakerEncoderSection.
            providers: Resolved ONNX Runtime providers list.

        Returns:
            Configured :class:WeSpeakerOnnxEncoder instance.
        """
        return cls(
            model_path=section.model_path,
            providers=providers,
            output_name=section.output_name,
            num_mel_bins=section.num_mel_bins,
            frame_length_ms=section.frame_length_ms,
            frame_shift_ms=section.frame_shift_ms,
            low_freq=section.low_freq,
            high_freq=section.high_freq,
            apply_cmvn=section.apply_cmvn,
            sample_rate=section.sample_rate,
        )
