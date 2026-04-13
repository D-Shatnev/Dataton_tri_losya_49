from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine
import numpy as np

class SpeakerEmbedder(ABC):    
    @abstractmethod
    def extract_embedding(self, file_path: str) -> np.ndarray:
        """Принимает путь к файлу, возвращает вектор эмбеддинга"""
        pass
    
    def compare(self, file1: str, file2: str) -> float:
        """Сравнивает два файла, возвращает косинусное сходство (-1..1)"""
        emb1 = self.extract_embedding(file1)
        emb2 = self.extract_embedding(file2)
        distance = cosine(emb1, emb2)
        return 1 - distance