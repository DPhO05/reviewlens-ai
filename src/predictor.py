from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.demo_models import DemoEmbeddingModel, extract_reason_codes
from src.feature_pipeline import BERFFeaturePipeline

DEFAULT_HF_MODEL_ID = "DPhO05/tiki-review-helpfulness-phobert-berf-gold-v2"
HF_ARTIFACTS = {
    "config.json": "config.json",
    "rf_model.joblib": "final_model/rf_embedding_probability_model.joblib",
    "metadata_scaler.joblib": "final_model/metadata_scaler.joblib",
    "annotation_feature_predictor.joblib": (
        "final_model/annotation_feature_predictor.joblib"
    ),
    "final_model.joblib": "final_model/final_ml_model.joblib",
}


def get_confidence_level(score: float) -> str:
    distance = abs(score - 0.5)
    if distance >= 0.35:
        return "High"
    if distance >= 0.2:
        return "Medium"
    return "Low"


def build_metadata(
    review_text: str,
    rating: int,
    metadata_cols: list[str],
    title: str = "",
) -> np.ndarray:
    content = str(review_text or "")
    title = str(title or "")
    text = f"{title} {content}".strip()
    words = text.split()
    values = {
        "rating": float(rating),
        "word_count": float(len(words)),
        "content_len": float(len(content)),
        "title_len": float(len(title)),
        "text_len": float(len(text)),
        "has_title": float(bool(title.strip())),
        "has_content": float(bool(content.strip())),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "digit_count": float(sum(ch.isdigit() for ch in text)),
        "uppercase_ratio": (
            float(sum(ch.isupper() for ch in text)) / max(len(text), 1)
        ),
        "is_low_rating": float(rating <= 2),
        "is_high_rating": float(rating >= 4),
        "is_extreme_rating": float(rating in (1, 5)),
    }
    missing = [column for column in metadata_cols if column not in values]
    if missing:
        raise ValueError(f"Unsupported metadata columns in model config: {missing}")
    return np.asarray([[values[column] for column in metadata_cols]], dtype=np.float32)


class PhoBERTEmbeddingModel:
    def __init__(self, model_name: str, max_length: int = 192):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PhoBERT backend requires torch and transformers"
            ) from exc

        self.torch = torch
        torch_threads = int(os.getenv("TORCH_NUM_THREADS", "1"))
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HF_TOKEN") or None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                token=token,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                token=token,
                local_files_only=True,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                token=token,
            )
            self.model = AutoModel.from_pretrained(model_name, token=token)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            summed = self.torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = self.torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        return pooled.cpu().numpy().astype(np.float32)


class HFGoldV2Predictor:
    def __init__(
        self,
        cache_dir: str,
        repo_id: str = DEFAULT_HF_MODEL_ID,
    ):
        self.cache_dir = Path(cache_dir)
        self.repo_id = repo_id
        self._loaded = False
        self.config: dict[str, Any] = {
            "model_variant": "hf-phobert-berf-gold-v2",
            "model_id": repo_id,
        }

    def _ensure_artifacts(self) -> dict[str, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        resolved: dict[str, Path] = {}
        missing: list[tuple[str, str]] = []
        for local_name, remote_name in HF_ARTIFACTS.items():
            local_path = self.cache_dir / local_name
            resolved[local_name] = local_path
            if not local_path.exists():
                missing.append((local_name, remote_name))

        if missing:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "Install huggingface-hub to download gold-v2 artifacts"
                ) from exc
            token = os.getenv("HF_TOKEN") or None
            for local_name, remote_name in missing:
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_name,
                    token=token,
                )
                resolved[local_name].write_bytes(Path(downloaded).read_bytes())
        return resolved

    def _load(self) -> None:
        if self._loaded:
            return
        artifacts = self._ensure_artifacts()
        self.config = json.loads(
            artifacts["config.json"].read_text(encoding="utf-8")
        )
        self.config["model_variant"] = "hf-phobert-berf-gold-v2"
        self.config["model_id"] = self.repo_id
        self.rf_model = joblib.load(artifacts["rf_model.joblib"])
        if hasattr(self.rf_model, "n_jobs"):
            self.rf_model.n_jobs = int(os.getenv("MODEL_INFERENCE_JOBS", "1"))
        self.metadata_scaler = joblib.load(artifacts["metadata_scaler.joblib"])
        self.annotation_model = joblib.load(
            artifacts["annotation_feature_predictor.joblib"]
        )
        self.final_model = joblib.load(artifacts["final_model.joblib"])
        if hasattr(self.final_model, "set_params"):
            try:
                self.final_model.set_params(
                    n_jobs=int(os.getenv("MODEL_INFERENCE_JOBS", "1"))
                )
            except ValueError:
                pass
        self.embedding_model = PhoBERTEmbeddingModel(
            self.config["embedding_model"],
            int(self.config.get("max_length", 192)),
        )
        self._loaded = True

    @staticmethod
    def _helpful_index(model) -> int:
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Final model does not contain helpful class 1")
        return classes.index(1)

    def predict(
        self,
        review_text: str,
        rating: int = 5,
        title: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty")
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        self._load()

        text = f"{title} {review_text}".strip()
        embedding = self.embedding_model.encode([text])
        metadata = build_metadata(
            review_text=review_text,
            rating=rating,
            metadata_cols=self.config["metadata_cols"],
            title=title,
        )
        metadata_scaled = self.metadata_scaler.transform(metadata)
        rf_proba = self.rf_model.predict_proba(embedding)

        annotation_input = np.hstack([embedding, metadata_scaled])
        annotation_features = self.annotation_model.predict(annotation_input)
        annotation_features = np.asarray(annotation_features, dtype=np.float32)
        annotation_features[:, 0] = np.clip(annotation_features[:, 0], -2, 7)
        annotation_features[:, 1] = np.clip(annotation_features[:, 1], 0, 1)

        final_features = np.hstack(
            [rf_proba, metadata_scaled, annotation_features]
        )
        expected = len(self.config["final_features"])
        if final_features.shape[1] != expected:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected}, "
                f"got {final_features.shape[1]}"
            )

        final_frame = pd.DataFrame(
            final_features,
            columns=self.config["final_features"],
        )
        probabilities = self.final_model.predict_proba(final_frame)
        score = float(
            probabilities[0, self._helpful_index(self.final_model)]
        )
        score = float(np.clip(score, 0.0, 1.0))
        return {
            "label": "Helpful" if score >= 0.5 else "Not Helpful",
            "helpfulness_score": round(score, 4),
            "confidence_level": get_confidence_level(score),
            "model_reason_codes": extract_reason_codes(review_text),
            "predicted_rubric_score": round(float(annotation_features[0, 0]), 3),
            "predicted_annotation_confidence": round(
                float(annotation_features[0, 1]), 3
            ),
            "model_variant": self.config["model_variant"],
            "model_id": self.repo_id,
        }

    def predict_many(
        self,
        reviews: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self.predict(**review) for review in reviews]


class DemoPredictor:
    def __init__(self, model_dir: str):
        model_path = Path(model_dir)
        self.config = json.loads(
            (model_path / "model_config.json").read_text(encoding="utf-8")
        )
        self.rf_model = joblib.load(model_path / "rf_model.pkl")
        self.final_model = joblib.load(model_path / "lgbm_model.pkl")
        self.embedding_model = DemoEmbeddingModel(self.config["embedding_dim"])
        self.pipeline = BERFFeaturePipeline(self.embedding_model, self.rf_model)
        self.threshold = float(self.config.get("threshold", 0.5))

    def predict(
        self,
        review_text: str,
        rating: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        del rating
        if not review_text or not review_text.strip():
            raise ValueError("Review text cannot be empty")
        features = self.pipeline.transform([review_text])
        probabilities = self.final_model.predict_proba(features)
        helpful_index = list(self.final_model.classes_).index(1)
        score = float(probabilities[0, helpful_index])
        return {
            "label": "Helpful" if score >= self.threshold else "Not Helpful",
            "helpfulness_score": round(score, 4),
            "confidence_level": get_confidence_level(score),
            "model_reason_codes": extract_reason_codes(review_text),
            "model_variant": self.config.get("model_variant", "demo"),
        }


class ReviewHelpfulnessPredictor:
    def __init__(self, model_dir: str = "models"):
        backend = os.getenv("MODEL_BACKEND", "hf_gold_v2").strip().lower()
        if backend == "demo":
            self.backend = DemoPredictor(model_dir)
        elif backend == "hf_gold_v2":
            repo_id = os.getenv("HF_MODEL_ID", DEFAULT_HF_MODEL_ID)
            cache_dir = os.getenv(
                "HF_MODEL_CACHE_DIR",
                str(Path(model_dir) / "hf_gold_v2"),
            )
            self.backend = HFGoldV2Predictor(cache_dir, repo_id)
        else:
            raise ValueError(f"Unsupported MODEL_BACKEND: {backend}")

    @property
    def config(self) -> dict[str, Any]:
        return self.backend.config

    def predict(self, review_text: str, **context: Any) -> dict[str, Any]:
        return self.backend.predict(review_text=review_text, **context)

    def predict_many(
        self,
        reviews: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self.predict(**review) for review in reviews]
