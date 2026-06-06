from src.gemini_service import GeminiReviewExplainer


def test_local_fallback_returns_detailed_reason_text():
    explainer = GeminiReviewExplainer(api_key=None)
    result = explainer._fallback(
        "Dùng 2 tuần, pin khoảng 6 giờ và âm thanh rõ nhưng mic hơi rè.",
        {
            "label": "Helpful",
            "model_reason_codes": [
                "specific_experience",
                "mentions_product_features",
                "mentions_pros_and_cons",
            ],
        },
    )

    assert len(result["why"]) == 3
    assert "kiểm chứng" in result["why"][0]
    assert "từng thuộc tính" in result["why"][1]
    assert "đánh đổi" in result["why"][2]
    assert result["score_interpretation"]["label"] == "Helpful"
    assert result["evidence_analysis"]
    assert result["suggested_rewrite"]
    assert 0 <= result["quality_assessment"]["information_quality_score"] <= 100


def test_prompt_requires_evidence_and_preserves_model_decision():
    explainer = GeminiReviewExplainer(api_key="test-key")
    prompt = explainer._build_prompt(
        {"product_name": "Tai nghe A", "rating": 4},
        "Pin dùng được 6 giờ nhưng mic hơi rè.",
        {
            "label": "Helpful",
            "helpfulness_score": 0.87,
            "confidence_level": "High",
            "model_reason_codes": ["specific_experience"],
        },
    )

    assert "Mỗi mục trong evidence_analysis phải trích" in prompt
    assert "tuyệt đối không tự đổi label hoặc điểm số helpfulness_score" in prompt
    assert '"suggested_rewrite"' in prompt
    assert '"helpfulness_score": 0.87' in prompt
    assert '"rubric_scores"' in prompt
    assert "41-60" in prompt


def test_normalize_result_restores_required_fields_and_model_values():
    explainer = GeminiReviewExplainer(api_key="test-key")
    model_result = {
        "label": "Not Helpful",
        "helpfulness_score": 0.21,
        "confidence_level": "Medium",
        "model_reason_codes": ["too_generic"],
    }
    result = explainer._normalize_result(
        {
            "short_assessment": "Phân tích từ Gemini.",
            "score_interpretation": {
                "label": "Helpful",
                "helpfulness_score": 0.99,
                "confidence_level": "High",
            },
            "why": "invalid",
        },
        "Ok shop.",
        model_result,
    )

    assert result["score_interpretation"]["label"] == "Not Helpful"
    assert result["score_interpretation"]["helpfulness_score"] == 0.21
    assert result["score_interpretation"]["confidence_level"] == "Medium"
    assert isinstance(result["why"], list)
    assert result["source"] == "gemini"
    assert result["quality_assessment"]["information_quality_score"] == 0


def test_generation_config_reserves_tokens_for_json_response():
    explainer = GeminiReviewExplainer(api_key="test-key", model="gemini-2.5-flash")
    config = explainer._generation_config()

    assert config["maxOutputTokens"] == 8192
    assert config["thinkingConfig"]["thinkingBudget"] == 512


def test_quality_score_uses_weighted_rubric_and_clamps_values():
    explainer = GeminiReviewExplainer(api_key="test-key")
    assessment = explainer._quality_assessment(
        {
            "specificity": 60,
            "feature_coverage": 50,
            "balance": 70,
            "usage_context": 40,
            "verifiability": 55,
            "buyer_relevance": 120,
        }
    )

    assert assessment["information_quality_score"] == 61.8
    assert assessment["dimensions"][-1]["score"] == 100
    assert assessment["score_type"] == "supplemental_rubric_score"
