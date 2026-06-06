from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.demo_models import DemoEmbeddingModel, extract_reason_codes
from src.feature_pipeline import BERFFeaturePipeline


def get_confidence_level(score: float) -> str:
    distance = abs(score - 0.5)
    if distance >= 0.35:
        return "High"
    if distance >= 0.2:
        return "Medium"
    return "Low"


class ReviewHelpfulnessPredictor:
    def __init__(self, model_dir: str = "models"):
        model_path = Path(model_dir)
        config_path = model_path / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "Missing model artifacts. Run: python scripts/train_demo_models.py"
            )

        self.config: dict[str, Any] = json.loads(config_path.read_text())
        self.rf_model = joblib.load(model_path / "rf_model.pkl")
        self.final_model = joblib.load(model_path / "lgbm_model.pkl")
        self.embedding_model = DemoEmbeddingModel(self.config["embedding_dim"])
        self.pipeline = BERFFeaturePipeline(self.embedding_model, self.rf_model)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.helpful_class = int(self.config.get("helpful_class", 1))

    def _helpful_index(self, model) -> int:
        classes = list(model.classes_)
        if self.helpful_class not in classes:
            raise ValueError(f"Helpful class {self.helpful_class} not in model classes")
        return classes.index(self.helpful_class)

    def predict(self, review_text: str) -> dict[str, Any]:
        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty")
        features = self.pipeline.transform([review_text])
        expected = int(self.config["final_feature_dim"])
        if features.shape[1] != expected:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected}, got {features.shape[1]}"
            )
        probabilities = self.final_model.predict_proba(features)
        score = float(probabilities[0, self._helpful_index(self.final_model)])
        score = float(np.clip(score, 0.0, 1.0))
        return {
            "label": "Helpful" if score >= self.threshold else "Not Helpful",
            "helpfulness_score": round(score, 4),
            "confidence_level": get_confidence_level(score),
            "model_reason_codes": extract_reason_codes(review_text),
            "model_variant": self.config.get("model_variant", "unknown"),
        }

    def predict_many(self, reviews: list[str]) -> list[dict[str, Any]]:
        return [self.predict(review) for review in reviews]
