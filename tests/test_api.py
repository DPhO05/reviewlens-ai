from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detailed_review_scores_above_generic_review():
    base = {
        "product_name": "Tai nghe A",
        "category": "electronics",
        "rating": 4,
        "use_gemini": False,
    }
    detailed = client.post(
        "/analyze-review",
        json={
            **base,
            "review_text": (
                "Dùng 2 tuần, pin khoảng 6 giờ, âm thanh rõ nhưng mic hơi rè ngoài đường."
            ),
        },
    )
    generic = client.post(
        "/analyze-review",
        json={**base, "review_text": "Ok shop."},
    )
    assert detailed.status_code == 200
    assert generic.status_code == 200
    assert (
        detailed.json()["helpfulness_score"]
        > generic.json()["helpfulness_score"]
    )


def test_rank_reviews_descending():
    response = client.post(
        "/rank-reviews",
        json={
            "reviews": [
                {
                    "product_name": "A",
                    "category": "electronics",
                    "rating": 5,
                    "review_text": "Ok shop.",
                    "use_gemini": False,
                },
                {
                    "product_name": "A",
                    "category": "electronics",
                    "rating": 4,
                    "review_text": (
                        "Dùng 1 tháng, pin 7 giờ, màn hình sáng nhưng loa hơi nhỏ."
                    ),
                    "use_gemini": False,
                },
            ]
        },
    )
    assert response.status_code == 200
    scores = [item["helpfulness_score"] for item in response.json()["reviews"]]
    assert scores == sorted(scores, reverse=True)
