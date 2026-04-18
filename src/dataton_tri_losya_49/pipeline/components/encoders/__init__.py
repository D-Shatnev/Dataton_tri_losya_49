"""
Encoder components.

This package provides encoder implementations compatible with the
dataton_tri_losya_49.pipeline.interfaces.Encoder protocol.

Encoders map a batch of waveforms with shape (B, T) to embeddings with shape (B, D).
"""

from dataton_tri_losya_49.pipeline.components.encoders.onnx_encoder import OnnxEncoder
from dataton_tri_losya_49.pipeline.components.encoders.wespeaker_onnx_encoder import WeSpeakerOnnxEncoder

__all__ = ["OnnxEncoder", "WeSpeakerOnnxEncoder"]
