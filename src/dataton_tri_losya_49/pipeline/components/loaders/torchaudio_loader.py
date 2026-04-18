"""
Waveform loader based on torchaudio with FFmpeg backend.

This module contains TorchAudioWaveformLoader — a drop-in replacement for
:class:`~dataton_tri_losya_49.pipeline.components.loaders.soundfile.SoundFileWaveformLoader`
that uses ``torchaudio.io.StreamReader`` for decoding.

``torchaudio.io.StreamReader`` is the **only** torchaudio component that is
guaranteed to use FFmpeg: it links directly against libavcodec / libavformat
at runtime and has no alternative backend. This is in contrast to
``torchaudio.load()``, which may fall back to sox or soundfile depending on
the installed extras.

Advantages over the soundfile-based loader:

- Supports a much wider range of audio formats (MP3, AAC, Opus, OGG, FLAC,
  WAV, M4A, …) via the system FFmpeg installation.
- Resampling is performed by ``torchaudio.functional.resample`` (native torch,
  no scipy dependency for this path).

Requirements:

- ``torchaudio >= 2.0`` (already a project dependency).
- System FFmpeg libraries (libavcodec, libavformat, libavutil) must be present.
  In Docker these are provided by the ``ffmpeg`` apt package.
  ``torchaudio.io.StreamReader`` will raise ``RuntimeError`` at construction
  time if FFmpeg is not available, giving an early and explicit failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional
import torchaudio.io
import torchaudio.utils.ffmpeg_utils

from dataton_tri_losya_49.constants import DEFAULT_TARGET_SR


def _check_ffmpeg_available() -> None:
    """Verify that torchaudio can find FFmpeg libraries.

    ``torchaudio.io.StreamReader`` is backed exclusively by FFmpeg.
    This helper probes availability at loader construction time so that
    missing FFmpeg produces a clear error message rather than a cryptic
    failure during the first ``load()`` call.

    Raises:
        RuntimeError: If FFmpeg libraries are not found by torchaudio.
    """
    try:
        # get_audio_decoders() queries libavcodec; raises if FFmpeg absent.
        torchaudio.utils.ffmpeg_utils.get_audio_decoders()
    except Exception as exc:
        raise RuntimeError(
            "TorchAudioWaveformLoader requires FFmpeg libraries (libavcodec / "
            "libavformat). Install the system 'ffmpeg' package and rebuild the "
            "Docker image. Original error: " + str(exc)
        ) from exc


@dataclass(frozen=True)
class TorchAudioWaveformLoader:
    """
    Load audio with torchaudio (FFmpeg via StreamReader) and resample to target SR (mono float32).

    Uses ``torchaudio.io.StreamReader`` which is the only torchaudio API that
    is **guaranteed** to use FFmpeg — it links directly against libavcodec /
    libavformat with no alternative backend. This enables support for formats
    not handled by soundfile (e.g. MP3, AAC, Opus).

    FFmpeg availability is verified at construction time via
    ``torchaudio.utils.ffmpeg_utils.get_audio_decoders()``, so a missing
    FFmpeg installation produces a clear error immediately rather than during
    the first audio load.

    Args:
        target_sr: Sampling rate to resample to. Defaults to 16 000 Hz.
        clip: If True, clip waveform values to [-1, 1]. Default is False to
            match the baseline behaviour (no explicit clipping).

    Note:
        The dataclass is frozen to prevent accidental mutation of configuration
        during a run (for reproducibility).
    """

    target_sr: int = DEFAULT_TARGET_SR
    clip: bool = False

    # Private sentinel — set in __post_init__ to trigger FFmpeg check once.
    _ffmpeg_checked: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Verify FFmpeg availability at construction time."""
        _check_ffmpeg_available()
        object.__setattr__(self, "_ffmpeg_checked", True)

    def load(self, path: Path) -> np.ndarray:
        """Load an audio file and return a mono waveform at target_sr.

        Uses ``torchaudio.io.StreamReader`` which is exclusively FFmpeg-backed.

        Args:
            path: Path to an audio file decodable by FFmpeg.

        Returns:
            1-D mono waveform of shape (T,) with dtype float32.

        Notes:
            - Multi-channel audio is downmixed to mono by mean across channels.
            - If the file sample rate differs from target_sr, audio is resampled
              using ``torchaudio.functional.resample``.
            - NaNs/Infs are replaced with zeros.
            - If clip=True, values are clipped to [-1, 1].

        Raises:
            RuntimeError: If the file cannot be decoded by FFmpeg.
        """
        # StreamReader is exclusively FFmpeg-backed (libavcodec/libavformat).
        streamer = torchaudio.io.StreamReader(str(path))

        # Inspect the first audio stream to get the native sample rate.
        stream_info = streamer.get_src_stream_info(0)
        native_sr: int = int(stream_info.sample_rate)
        num_channels: int = int(stream_info.num_channels)

        # Add output stream: decode to float32 planar PCM at native SR.
        # format="fltp" → float32 planar, output shape [T, C].
        streamer.add_basic_audio_stream(
            frames_per_chunk=-1,  # read entire file in one chunk
            stream_index=0,
            sample_rate=native_sr,
            format="fltp",
        )

        chunks = [chunk for (chunk,) in streamer.stream()]
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        # StreamReader yields [T, C] tensors; concatenate along time axis -> [T_total, C]
        waveform: torch.Tensor = torch.cat(chunks, dim=0)

        # Transpose to [C, T_total] for consistent downstream processing
        waveform = waveform.T

        # Downmix to mono: [C, T] -> [1, T]
        if num_channels > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if native_sr != int(self.target_sr):
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=native_sr, new_freq=int(self.target_sr)
            )

        # [1, T] -> [T], convert to numpy float32
        audio: np.ndarray = waveform.squeeze(0).numpy().astype(np.float32, copy=False)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip:
            audio = np.clip(audio, -1.0, 1.0)
        return audio
