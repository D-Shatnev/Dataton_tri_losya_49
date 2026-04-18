"""
Encoder components.

This package provides encoder implementations compatible with the
dataton_tri_losya_49.pipeline.interfaces.Encoder protocol.

Encoders map a batch of waveforms with shape (B, T) to embeddings with shape (B, D).
"""

from dataton_tri_losya_49.pipeline.components.encoders.espnet_encoder import EspnetEncoder
from dataton_tri_losya_49.pipeline.components.encoders.onnx_encoder import OnnxEncoder

__all__ = ["EspnetEncoder", "OnnxEncoder"]
