from __future__ import annotations

import numpy as np

from src.preprocessing import clean_text


class BERFFeaturePipeline:
    def __init__(self, embedding_model, rf_model):
        self.embedding_model = embedding_model
        self.rf_model = rf_model

    def transform(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(text) for text in texts]
        embeddings = self.embedding_model.encode(cleaned)
        rf_proba = self.rf_model.predict_proba(embeddings)
        if embeddings.ndim != 2 or rf_proba.ndim != 2:
            raise ValueError("Embedding and RF probabilities must be 2D arrays")
        return np.concatenate([embeddings, rf_proba], axis=1)
