import torch
import soundfile as sf
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from .base import SpeakerEmbedder

class SpeechBrainEmbedder(SpeakerEmbedder):
    def __init__(self, device="cpu"):
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
    
    def extract_embedding(self, file_path: str) -> np.ndarray:
        signal, fs = sf.read(file_path, dtype='float32')
        signal = torch.from_numpy(signal).unsqueeze(0)
        
        embedding = self.model.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()