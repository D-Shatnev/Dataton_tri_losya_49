
import onnxruntime as ort
import numpy as np
import soundfile as sf

from .base import SpeakerEmbedder


class ONNXEmbedder(SpeakerEmbedder):
    def __init__(self, model_path: str, use_mono_mean=True):
        """
        model_path: путь к .onnx файлу
        use_mono_mean: если True и аудио стерео, усредняет каналы
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.use_mono_mean = use_mono_mean
        self.num_outputs = len(self.session.get_outputs())
        
    def _load_audio(self, file_path: str) -> np.ndarray:
        """Загружает и приводит аудио к формату [batch, time]"""
        data, samplerate = sf.read(file_path)
        
        if len(data.shape) == 2:
            if self.use_mono_mean:
                waveform = np.mean(data, axis=1)
            else:
                waveform = data[:, 0]
        else:
            waveform = data
        
        waveform = waveform.reshape(1, -1).astype(np.float32)
        return waveform
    
    def extract_embedding(self, file_path: str) -> np.ndarray:
        waveform = self._load_audio(file_path)
        outputs = self.session.run(None, {self.input_name: waveform})
        return outputs[0].flatten()