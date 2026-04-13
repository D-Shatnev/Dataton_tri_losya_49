"""
GMM-UBM адаптированный под интерфейс SpeakerEmbedder.
Возвращает нормированный LLR.
"""

import warnings
import pickle
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

from .base import SpeakerEmbedder


class GMMUBMEmbedder(SpeakerEmbedder):
    """
    GMM-UBM реализация интерфейса SpeakerEmbedder.
    Возвращает нормированный LLR.
    """

    def __init__(self, ubm_path: str, relevance_factor: float = 16.0, embedding_dim: int = 256):
        """
        Args:
            ubm_path: путь к предобученной UBM модели (.pkl)
            relevance_factor: параметр MAP-адаптации
        """
        self.relevance_factor = relevance_factor
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.n_fft = 512
        self.hop_length = 160

        with open(ubm_path, "rb") as f:
            self.ubm = pickle.load(f)

        np.random.seed(42)
        supervector_dim = self.ubm.n_components * 39  # 512 * 39 ≈ 20K
        self.projection = np.random.randn(supervector_dim, embedding_dim) / np.sqrt(
            embedding_dim
        )

        print(f"✓ GMM-UBM инициализирован: {self.ubm.n_components} компонент")

    def _extract_mfcc(self, file_path: str) -> np.ndarray:
        """Извлекает MFCC + Δ + ΔΔ из аудиофайла."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
        return features

    def _create_blueprint(self, features: np.ndarray):
        """
        Создаёт GMM-слепок голоса методом MAP-адаптации UBM.
        """
        n_components = self.ubm.n_components
        n_features = features.shape[1]

        posteriors = self.ubm.predict_proba(features)
        n_k = np.sum(posteriors, axis=0)

        means_adapted = np.zeros((n_components, n_features))

        for k in range(n_components):
            if n_k[k] > 0:
                weighted_sum = np.sum(posteriors[:, k : k + 1] * features, axis=0)
                means_adapted[k] = weighted_sum / n_k[k]
            else:
                means_adapted[k] = self.ubm.means_[k]

        alpha = n_k / (n_k + self.relevance_factor)
        alpha = alpha.reshape(-1, 1)
        adapted_means = alpha * means_adapted + (1 - alpha) * self.ubm.means_

        gmm_speaker = GaussianMixture(
            n_components=n_components, covariance_type="diag", random_state=42
        )
        gmm_speaker.means_ = adapted_means
        gmm_speaker.covariances_ = self.ubm.covariances_.copy()
        gmm_speaker.weights_ = self.ubm.weights_.copy()
        gmm_speaker.precisions_cholesky_ = self.ubm.precisions_cholesky_.copy()

        return gmm_speaker
    
    def _gmm_to_supervector(self, gmm) -> np.ndarray:
        """
        Преобразует GMM-слепок в супервектор:
        [адаптированные средние - средние UBM] / sqrt(ковариации UBM)
        """
        means_diff = gmm.means_ - self.ubm.means_
        std_inv = 1.0 / np.sqrt(self.ubm.covariances_ + 1e-10)
        normalized_diff = means_diff * std_inv

        weighted_diff = normalized_diff * np.sqrt(self.ubm.weights_[:, np.newaxis])

        return weighted_diff.flatten()

    def _compute_llr_norm(self, gmm_speaker, features: np.ndarray) -> float:
        """
        Вычисляет нормированный LLR: (log P(X|speaker) - log P(X|UBM)) / n_frames
        """
        ll_speaker = gmm_speaker.score(features)
        ll_ubm = self.ubm.score(features)
        llr_norm = (ll_speaker - ll_ubm) / features.shape[0]
        return llr_norm

    def extract_embedding(self, file_path: str) -> np.ndarray:
        """
        Создаёт векторный эмбеддинг для быстрого поиска.

        Процесс:
        1. Создаёт GMM-слепок
        2. Преобразует в супервектор (≈20K dim)
        3. Сжимает через случайную проекцию в embedding_dim
        4. Нормализует для косинусного сходства
        """
        features = self._extract_mfcc(file_path)
        blueprint = self._create_blueprint(features)

        supervector = self._gmm_to_supervector(blueprint)

        embedding = supervector @ self.projection
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        return embedding.astype(np.float32)

    def compare(self, file1: str, file2: str) -> float:
        """
            Сравнивает два аудиофайла через GMM-UBM.

            Алгоритм:
          1. Создаёт слепок из file1
          2. Вычисляет нормированный LLR для file2 относительно слепка file1

          Returns:
            float: нормированный LLR (>0 = тот же диктор, <0 = разные)
        """
        features1 = self._extract_mfcc(file1)
        features2 = self._extract_mfcc(file2)

        blueprint = self._create_blueprint(features1)
        llr_norm = self._compute_llr_norm(blueprint, features2)

        return llr_norm