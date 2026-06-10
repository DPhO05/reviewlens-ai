from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.gemini_service import GeminiReviewExplainer
from src.predictor import ReviewHelpfulnessPredictor

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="ReviewLens AI API",
    version="1.0.0",
    description="BERF-style review helpfulness product demo",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ReviewHelpfulnessPredictor(str(ROOT / "models"))
explainer = GeminiReviewExplainer()


class ReviewRequest(BaseModel):
    product_name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=1, le=5)
    review_text: str = Field(min_length=1, max_length=10000)
    price: float | None = Field(default=None, ge=0)
    brand: str | None = Field(default=None, max_length=100)
    verified_purchase: bool | None = None
    use_gemini: bool = True


class BatchRequest(BaseModel):
    reviews: list[ReviewRequest] = Field(min_length=1, max_length=500)
    use_gemini_summary: bool = False


def analyze_item(request: ReviewRequest) -> dict[str, Any]:
    try:
        model_result = predictor.predict(
            review_text=request.review_text,
            rating=request.rating,
            verified_purchase=request.verified_purchase,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    explanation = None
    if request.use_gemini:
        explanation = explainer.explain(
            product_info={
                "product_name": request.product_name,
                "category": request.category,
                "rating": request.rating,
                "price": request.price,
                "brand": request.brand,
                "verified_purchase": request.verified_purchase,
            },
            review_text=request.review_text,
            model_result=model_result,
        )
    return {**model_result, "gemini_explanation": explanation}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_variant": predictor.config.get("model_variant"),
        "model_id": predictor.config.get("model_id"),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


@app.post("/analyze-review")
def analyze_review(request: ReviewRequest) -> dict[str, Any]:
    return analyze_item(request)


@app.post("/rank-reviews")
def rank_reviews(request: BatchRequest) -> dict[str, Any]:
    ranked = []
    for index, review in enumerate(request.reviews):
        item = analyze_item(review.model_copy(update={"use_gemini": False}))
        ranked.append(
            {
                "original_index": index,
                **review.model_dump(),
                **item,
            }
        )
    ranked.sort(key=lambda item: item["helpfulness_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {"count": len(ranked), "reviews": ranked}
