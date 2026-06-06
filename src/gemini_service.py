from __future__ import annotations

import json
import os
from typing import Any

import requests


REASON_TEXT = {
    "specific_experience": (
        "Review có dẫn chứng từ trải nghiệm sử dụng thực tế và đưa ra thông tin "
        "định lượng như thời gian, số lần sử dụng hoặc mức hiệu năng. Những chi "
        "tiết này giúp người mua dễ kiểm chứng và hình dung sản phẩm hoạt động "
        "như thế nào trong điều kiện sử dụng cụ thể."
    ),
    "mentions_product_features": (
        "Review đề cập từ hai đặc điểm cụ thể của sản phẩm, chẳng hạn như pin, "
        "âm thanh, micro, màn hình, chất liệu hoặc độ bền. Việc đánh giá theo "
        "từng thuộc tính cung cấp nhiều thông tin thực tế hơn một nhận xét chung "
        "và giúp người mua đối chiếu với nhu cầu của mình."
    ),
    "mentions_pros_and_cons": (
        "Review trình bày cả điểm đáp ứng tốt và điểm còn hạn chế của sản phẩm, "
        "thay vì chỉ khen hoặc chê một chiều. Góc nhìn cân bằng này giúp người "
        "mua hiểu rõ các đánh đổi và đưa ra quyết định phù hợp hơn."
    ),
    "sufficient_detail": (
        "Review có độ dài và lượng thông tin đủ để mô tả bối cảnh sử dụng, đặc "
        "điểm đã trải nghiệm và nhận xét của người dùng. Nội dung chi tiết làm "
        "tăng giá trị tham khảo so với các bình luận chỉ gồm vài từ."
    ),
    "too_generic": (
        "Review quá ngắn hoặc chỉ đưa ra nhận xét chung như “tốt”, “đẹp” hay "
        "“giao nhanh” mà chưa mô tả tính năng, chất lượng hoặc trải nghiệm sử "
        "dụng sản phẩm. Vì thiếu dẫn chứng cụ thể, người mua khó dùng bình luận "
        "này để so sánh sản phẩm hoặc hỗ trợ quyết định mua."
    ),
}


class GeminiReviewExplainer:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def _fallback(self, review_text: str, model_result: dict[str, Any]) -> dict[str, Any]:
        codes = model_result.get("model_reason_codes", [])
        why = [REASON_TEXT[code] for code in codes if code in REASON_TEXT]
        if not why:
            why = ["Mức độ chi tiết và thông tin thực tế trong review còn hạn chế."]
        helpful = model_result["label"] == "Helpful"
        return {
            "short_assessment": (
                "Review này cung cấp thông tin hữu ích cho người mua."
                if helpful
                else "Review này chưa cung cấp đủ thông tin để hỗ trợ quyết định mua."
            ),
            "score_interpretation": {
                "label": model_result["label"],
                "helpfulness_score": model_result.get("helpfulness_score"),
                "confidence_level": model_result.get("confidence_level"),
                "explanation": (
                    "Điểm số cho thấy model nhận diện được nhiều tín hiệu thường "
                    "xuất hiện trong review hữu ích."
                    if helpful
                    else "Điểm số cho thấy review còn thiếu các tín hiệu thông tin "
                    "mà model thường liên hệ với một review hữu ích."
                ),
            },
            "why": why[:3],
            "evidence_analysis": [
                {
                    "criterion": code,
                    "status": "detected",
                    "evidence": "Tín hiệu được phát hiện bằng rule-based reason code.",
                    "analysis": REASON_TEXT[code],
                    "buyer_impact": (
                        "Tín hiệu này làm tăng lượng thông tin mà người mua có thể "
                        "dùng để đánh giá mức độ phù hợp của sản phẩm."
                    ),
                }
                for code in codes[:3]
                if code in REASON_TEXT
            ],
            "strengths": why[:3] if helpful else [],
            "limitations": (
                []
                if helpful
                else [
                    "Review chưa mô tả đủ bối cảnh sử dụng hoặc thuộc tính sản phẩm.",
                    "Nhận xét chưa có dẫn chứng định lượng để người mua kiểm chứng.",
                ]
            ),
            "missing_information": (
                ["Trải nghiệm dài hạn hoặc độ bền sau một thời gian sử dụng."]
                if helpful
                else [
                    "Thời gian và bối cảnh sử dụng sản phẩm.",
                    "Các tính năng hoặc thuộc tính đã trực tiếp trải nghiệm.",
                    "Ví dụ, số liệu hoặc so sánh làm căn cứ cho nhận xét.",
                ]
            ),
            "improvement_suggestion": (
                "Có thể bổ sung thời gian sử dụng, tính năng đã thử, ưu nhược điểm và số liệu cụ thể."
                if not helpful
                else "Có thể bổ sung ảnh thực tế và trải nghiệm sử dụng dài hạn."
            ),
            "suggested_rewrite": (
                "Sau [thời gian sử dụng], tôi đã thử [tính năng chính]. Sản phẩm "
                "hoạt động tốt ở [điểm cụ thể], nhưng còn hạn chế ở [điểm cụ thể]. "
                "Sản phẩm phù hợp với [nhóm nhu cầu] vì [lý do]."
            ),
            "buyer_value": (
                "Giúp người mua hình dung rõ hơn trải nghiệm sử dụng thực tế."
                if helpful
                else "Giá trị tham khảo còn thấp do thiếu thông tin kiểm chứng được."
            ),
            "model_alignment": (
                "Phần giải thích diễn giải các tín hiệu hỗ trợ kết quả của model; "
                "Gemini/local fallback không thay đổi nhãn hoặc điểm dự đoán."
            ),
            "source": "local_fallback",
        }

    def _build_prompt(
        self,
        product_info: dict[str, Any],
        review_text: str,
        model_result: dict[str, Any],
    ) -> str:
        return f"""Bạn là chuyên gia phân tích mức độ hữu ích của review thương mại điện tử.

MỤC TIÊU
Diễn giải thật chi tiết kết quả của model machine learning để người dùng hiểu:
1. Review đang cung cấp thông tin gì.
2. Những chi tiết nào làm review hữu ích hoặc chưa hữu ích.
3. Thông tin đó hỗ trợ quyết định mua như thế nào.
4. Review còn thiếu gì và nên cải thiện ra sao.

RANH GIỚI VAI TRÒ
- Model ML là thành phần duy nhất quyết định label và helpfulness_score.
- Bạn chỉ giải thích kết quả, tuyệt đối không tự đổi label hoặc điểm số.
- Không đánh giá sản phẩm là tốt hay xấu; chỉ đánh giá chất lượng thông tin của review.
- Không suy đoán thông số, trải nghiệm, ý định hoặc đặc điểm không xuất hiện trong input.
- Phân biệt rõ “bằng chứng có trong review” và “thông tin còn thiếu”.
- Nếu confidence thấp, phải nói rõ kết quả gần ranh giới phân loại và cần thận trọng.
- Nếu rating mâu thuẫn với nội dung, chỉ nêu đây là điểm chưa nhất quán, không tự kết luận.

INPUT
Thông tin sản phẩm:
{json.dumps(product_info, ensure_ascii=False, indent=2)}

Review nguyên văn:
{json.dumps(review_text, ensure_ascii=False)}

Kết quả từ model ML:
{json.dumps(model_result, ensure_ascii=False, indent=2)}

CÁC TIÊU CHÍ CẦN PHÂN TÍCH
- Tính cụ thể: có thời gian, số liệu, tình huống hoặc điều kiện sử dụng hay không.
- Độ bao phủ thuộc tính: review nói đến những tính năng/khía cạnh nào của sản phẩm.
- Tính cân bằng: có nêu cả ưu điểm, hạn chế hoặc điều kiện đánh đổi hay không.
- Bối cảnh trải nghiệm: người viết đã dùng trong hoàn cảnh nào và trong bao lâu.
- Khả năng kiểm chứng: nhận xét có dẫn chứng cụ thể hay chỉ là cảm nhận chung.
- Mức liên quan: nội dung có tập trung vào trải nghiệm sản phẩm hay chỉ nói về shop/giao hàng.
- Giá trị ra quyết định: người mua có thể dùng thông tin nào để so sánh hoặc xác định độ phù hợp.
- Khoảng trống thông tin: còn thiếu điều gì để review đáng tin và hữu ích hơn.

YÊU CẦU VỀ BẰNG CHỨNG
- Mỗi mục trong evidence_analysis phải trích một đoạn rất ngắn từ review ở field evidence.
- Nếu review không có bằng chứng cho tiêu chí, dùng evidence là chuỗi rỗng và status là "missing".
- Không được tạo câu trích dẫn không có trong review.
- analysis phải giải thích mối liên hệ giữa bằng chứng và mức độ hữu ích, không chỉ lặp lại evidence.
- buyer_impact phải nêu tác động cụ thể đến quyết định của người mua.

OUTPUT
Trả về duy nhất một JSON object hợp lệ, không markdown, không code fence, theo đúng cấu trúc:
{{
  "short_assessment": "Kết luận 2-3 câu, nhắc đúng label, score và confidence của model.",
  "score_interpretation": {{
    "label": "Giữ nguyên label từ model",
    "helpfulness_score": 0.0,
    "confidence_level": "Giữ nguyên confidence từ model",
    "explanation": "Giải thích ý nghĩa điểm số và độ tự tin, không mô tả đây là xác suất tuyệt đối."
  }},
  "why": [
    "3-5 lý do quan trọng nhất, mỗi lý do 2-3 câu và gắn với review."
  ],
  "evidence_analysis": [
    {{
      "criterion": "Tên tiêu chí bằng tiếng Việt",
      "status": "strong | partial | missing",
      "evidence": "Trích đoạn ngắn có thật trong review hoặc chuỗi rỗng",
      "analysis": "Phân tích chi tiết bằng chứng hoặc phần còn thiếu",
      "buyer_impact": "Thông tin này giúp hoặc hạn chế người mua ra quyết định thế nào"
    }}
  ],
  "strengths": ["Các điểm mạnh cụ thể của cách viết review"],
  "limitations": ["Các hạn chế cụ thể, không lặp lại máy móc"],
  "missing_information": ["Thông tin còn thiếu và lý do thông tin đó quan trọng"],
  "improvement_suggestion": "Một đoạn hướng dẫn cải thiện theo thứ tự ưu tiên.",
  "suggested_rewrite": "Phiên bản review được viết lại từ đúng thông tin đã có; dùng [cần bổ sung] cho dữ liệu còn thiếu.",
  "buyer_value": "Một đoạn mô tả review giúp người mua nào, trong quyết định nào và còn giới hạn gì.",
  "model_alignment": "Giải thích các nhận định trên liên hệ với reason codes và kết quả model ra sao."
}}

QUY TẮC CHẤT LƯỢNG
- Viết hoàn toàn bằng tiếng Việt tự nhiên, rõ ràng, không dùng thuật ngữ khó hiểu nếu không giải thích.
- evidence_analysis nên có 4-7 tiêu chí, why có 3-5 ý.
- Không lặp lại cùng một nhận xét ở nhiều field.
- suggested_rewrite không được thêm thông số hoặc trải nghiệm mới.
- helpfulness_score phải giữ nguyên kiểu số và giá trị từ model.
- Chỉ trả JSON hợp lệ."""

    def _normalize_result(
        self,
        result: dict[str, Any],
        review_text: str,
        model_result: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = self._fallback(review_text, model_result)
        normalized = {**fallback, **result}
        normalized["score_interpretation"] = {
            **fallback["score_interpretation"],
            **(
                result.get("score_interpretation", {})
                if isinstance(result.get("score_interpretation"), dict)
                else {}
            ),
            "label": model_result["label"],
            "helpfulness_score": model_result.get("helpfulness_score"),
            "confidence_level": model_result.get("confidence_level"),
        }
        for field in (
            "why",
            "evidence_analysis",
            "strengths",
            "limitations",
            "missing_information",
        ):
            if not isinstance(normalized.get(field), list):
                normalized[field] = fallback[field]
        normalized["source"] = "gemini"
        return normalized

    def _generation_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 8192,
        }
        if self.model.startswith("gemini-2.5"):
            config["thinkingConfig"] = {"thinkingBudget": 512}
        return config

    @staticmethod
    def _fallback_reason(exc: Exception) -> str:
        if isinstance(exc, requests.HTTPError):
            status = exc.response.status_code if exc.response is not None else None
            return f"gemini_http_{status}" if status else "gemini_http_error"
        if isinstance(exc, requests.Timeout):
            return "gemini_timeout"
        if isinstance(exc, requests.RequestException):
            return "gemini_connection_error"
        if isinstance(exc, json.JSONDecodeError):
            return "gemini_invalid_json"
        message = str(exc)
        if message.startswith("gemini_finish_reason:"):
            return message.replace("gemini_finish_reason:", "gemini_", 1).lower()
        return "gemini_response_error"

    def explain(
        self,
        product_info: dict[str, Any],
        review_text: str,
        model_result: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.api_key:
            return self._fallback(review_text, model_result)

        prompt = self._build_prompt(product_info, review_text, model_result)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":generateContent?key={self.api_key}"
        )
        try:
            response = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": self._generation_config(),
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            candidate = payload["candidates"][0]
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason != "STOP":
                raise ValueError(f"gemini_finish_reason:{finish_reason}")
            text = candidate["content"]["parts"][0]["text"]
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("Gemini response must be a JSON object")
            return self._normalize_result(result, review_text, model_result)
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            fallback = self._fallback(review_text, model_result)
            fallback["source"] = "local_fallback_after_gemini_error"
            fallback["fallback_reason"] = self._fallback_reason(exc)
            return fallback
