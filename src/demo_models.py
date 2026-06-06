from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class DemoEmbeddingModel:
    """Small offline embedding adapter used only for the product demo."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectorizer = HashingVectorizer(
            n_features=dimension,
            alternate_sign=False,
            norm="l2",
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.vectorizer.transform(list(texts)).toarray().astype(np.float32)


def extract_reason_codes(review_text: str) -> list[str]:
    text = review_text.lower()
    codes: list[str] = []
    if re.search(r"\b\d+\s*(ngày|tuần|tháng|giờ|phút|lần|km|mah|gb|cm|%)?\b", text):
        codes.append("specific_experience")
    feature_words = (
        "pin", "mic", "âm thanh", "màn hình", "camera", "mùi", "vị", "chất liệu",
        "kích thước", "độ bền", "đóng gói", "giao hàng", "hiệu năng", "giá",
    )
    if sum(word in text for word in feature_words) >= 2:
        codes.append("mentions_product_features")
    pros = ("tốt", "ổn", "rõ", "nhanh", "đẹp", "êm", "tiện", "ưu điểm")
    cons = ("nhưng", "tuy nhiên", "hơi", "kém", "rè", "chậm", "nhược điểm")
    if any(word in text for word in pros) and any(word in text for word in cons):
        codes.append("mentions_pros_and_cons")
    if len(text.split()) >= 25:
        codes.append("sufficient_detail")
    if not codes and len(text.split()) <= 8:
        codes.append("too_generic")
    return codes
