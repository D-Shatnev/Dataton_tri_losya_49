"""
Dataset loaders.

This package contains loaders that combine reading metadata (CSV) and loading audio.
"""

from dataton_tri_losya_49.pipeline.components.loaders.csv_audio_dataset import CsvAudioDatasetLoader
from dataton_tri_losya_49.pipeline.components.loaders.soundfile import SoundFileWaveformLoader

__all__ = ["CsvAudioDatasetLoader", "SoundFileWaveformLoader"]
