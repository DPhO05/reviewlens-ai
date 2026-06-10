import numpy as np

from src.predictor import HFGoldV2Predictor, build_metadata


METADATA_COLS = [
    "rating",
    "word_count",
    "content_len",
    "title_len",
    "text_len",
    "has_content",
    "exclamation_count",
    "question_count",
    "digit_count",
    "uppercase_ratio",
    "is_low_rating",
    "is_high_rating",
    "is_extreme_rating",
]


class FakeEmbedding:
    def encode(self, texts):
        assert texts == ["Pin dùng 6 giờ nhưng mic hơi rè."]
        return np.ones((1, 4), dtype=np.float32)


class FakeScaler:
    def transform(self, values):
        assert values.shape == (1, len(METADATA_COLS))
        return values


class FakeRF:
    classes_ = np.array([0, 1])

    def predict_proba(self, values):
        assert values.shape == (1, 4)
        return np.array([[0.3, 0.7]])


class FakeAnnotationModel:
    def predict(self, values):
        assert values.shape == (1, 4 + len(METADATA_COLS))
        return np.array([[9.0, 1.5]])


class FakeFinalModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, values):
        values = np.asarray(values)
        assert values.shape == (1, 2 + len(METADATA_COLS) + 2)
        assert values[0, -2] == 7.0
        assert values[0, -1] == 1.0
        return np.array([[0.4, 0.6]])


def test_build_metadata_matches_gold_v2_config_order():
    values = build_metadata(
        "Pin dùng 6 giờ nhưng mic hơi rè!",
        rating=4,
        metadata_cols=METADATA_COLS,
    )
    assert values.shape == (1, 13)
    assert values[0, 0] == 4
    assert values[0, 1] == 8
    assert values[0, 5] == 1
    assert values[0, 6] == 1
    assert values[0, 8] == 1
    assert values[0, 11] == 1


def test_gold_v2_predictor_builds_final_feature_contract(tmp_path):
    predictor = HFGoldV2Predictor(str(tmp_path), "test/repo")
    predictor.config = {
        "metadata_cols": METADATA_COLS,
        "final_features": [
            "rf_0",
            "rf_1",
            *METADATA_COLS,
            "predicted_helpfulness_score",
            "predicted_confidence",
        ],
        "model_variant": "hf-phobert-berf-gold-v2",
    }
    predictor.embedding_model = FakeEmbedding()
    predictor.metadata_scaler = FakeScaler()
    predictor.rf_model = FakeRF()
    predictor.annotation_model = FakeAnnotationModel()
    predictor.final_model = FakeFinalModel()
    predictor._loaded = True

    result = predictor.predict(
        "Pin dùng 6 giờ nhưng mic hơi rè.",
        rating=4,
    )

    assert result["label"] == "Helpful"
    assert result["helpfulness_score"] == 0.6
    assert result["predicted_rubric_score"] == 7.0
    assert result["predicted_annotation_confidence"] == 1.0
    assert result["model_id"] == "test/repo"
