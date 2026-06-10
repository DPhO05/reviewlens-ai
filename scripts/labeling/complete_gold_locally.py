from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from relabel_gold_data import (
    CHECKPOINT_PATH,
    GOLD_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    REPORT_PATH,
    REVIEW_QUEUE_PATH,
    clean_text,
    create_outputs,
    load_checkpoint,
    load_source,
    refine_flags,
)

GENERIC_PATTERNS = {
    "tốt",
    "tot",
    "ok",
    "okay",
    "hay",
    "đẹp",
    "ngon",
    "thơm",
    "ổn",
    "tạm được",
    "bình thường",
    "rất tốt",
    "quá tốt",
    "tuyệt vời",
    "ưng ý",
    "hài lòng",
    "rất hài lòng",
    "cực kì hài lòng",
    "cực kỳ hài lòng",
    "sản phẩm tốt",
    "sp tốt",
    "chất lượng tốt",
    "đáng tiền",
    "năm sao",
    "5 sao",
}

PRODUCT_ATTRIBUTES = (
    "pin",
    "sạc",
    "âm thanh",
    "loa",
    "mic",
    "màn hình",
    "camera",
    "chất liệu",
    "vải",
    "size",
    "kích thước",
    "mùi",
    "vị",
    "ngọt",
    "cay",
    "đắng",
    "mỏng",
    "dày",
    "nặng",
    "nhẹ",
    "rộng",
    "chật",
    "nóng",
    "mát",
    "êm",
    "ồn",
    "rè",
    "bền",
    "độ bền",
    "hiệu năng",
    "tốc độ",
    "nguồn",
    "nắp",
    "dây",
    "khóa",
    "đường may",
    "màu",
    "hình ảnh",
    "nội dung",
    "cốt truyện",
    "bản dịch",
    "giấy",
    "da",
    "tóc",
    "dạ dày",
    "bao bì",
    "hộp",
    "chai",
    "tem",
    "hạn sử dụng",
    "hút",
    "độ hút",
    "tác dụng",
    "chất lượng",
    "hoạt động",
    "độ chính xác",
    "độ nhạy",
    "khả năng",
)

EXPERIENCE_TERMS = (
    "mình dùng",
    "tôi dùng",
    "đã dùng",
    "sử dụng",
    "dùng được",
    "mua về",
    "mặc",
    "uống",
    "ăn",
    "đọc",
    "thử",
    "trải nghiệm",
    "sau khi",
    "nhận hàng",
    "lắp",
    "giặt",
    "sạc",
    "chạy",
)

DECISION_TERMS = (
    "phù hợp",
    "không hợp",
    "nên",
    "không nên",
    "cân nhắc",
    "so với",
    "rẻ hơn",
    "đắt hơn",
    "hời hơn",
    "giống",
    "khác",
    "ưu điểm",
    "nhược điểm",
    "nhưng",
    "tuy nhiên",
    "lỗi",
    "hỏng",
    "không hoạt động",
    "không lên nguồn",
    "sai mẫu",
    "thiếu",
    "bị móp",
    "bị vỡ",
    "bị chảy",
    "bị cong",
    "bị gập",
    "móp méo",
    "yếu",
    "không vừa",
    "vừa vặn",
    "không đúng",
    "đúng mô tả",
    "đúng hãng",
)

SHIPPING_TERMS = (
    "giao hàng",
    "shipper",
    "giao nhanh",
    "giao chậm",
    "đóng gói",
    "shop",
    "phục vụ",
    "tư vấn",
)

NOISE_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)
NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:phút|giờ|ngày|tuần|tháng|lần|cm|mm|kg|g|ml|mah|%)?\b",
    flags=re.IGNORECASE,
)

DEFECT_TERMS = (
    "lỗi",
    "hỏng",
    "bể",
    "vỡ",
    "móp",
    "móp méo",
    "cong",
    "gập",
    "rách",
    "rè",
    "yếu",
    "không hoạt động",
    "không lên nguồn",
    "sai mẫu",
    "không đúng",
    "thiếu",
    "chảy",
    "tụt pin",
)


def normalized_text(title: Any, content: Any) -> str:
    value = f"{clean_text(title)}. {clean_text(content)}".strip(". ")
    return unicodedata.normalize("NFC", value).lower()


def content_only(title: Any, content: Any) -> str:
    return unicodedata.normalize("NFC", clean_text(content)).lower()


def is_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return not compact or bool(NOISE_RE.fullmatch(compact))


def is_generic(text: str) -> bool:
    cleaned = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
    cleaned = " ".join(cleaned.split())
    if cleaned in GENERIC_PATTERNS:
        return True
    words = cleaned.split()
    generic_words = {
        "tốt",
        "rất",
        "quá",
        "ok",
        "hay",
        "đẹp",
        "ngon",
        "thơm",
        "ổn",
        "hài",
        "lòng",
        "ưng",
        "ý",
        "sản",
        "phẩm",
        "sp",
        "shop",
    }
    return len(words) <= 5 and bool(words) and all(word in generic_words for word in words)


def score_review(row: pd.Series) -> dict[str, Any]:
    text = normalized_text(row["title"], row["content"])
    content = content_only(row["title"], row["content"])
    words = re.findall(r"\w+", content, flags=re.UNICODE)
    attributes = [term for term in PRODUCT_ATTRIBUTES if term in content]
    experience_hits = [term for term in EXPERIENCE_TERMS if term in content]
    decision_hits = [term for term in DECISION_TERMS if term in content]
    defect_hits = [term for term in DEFECT_TERMS if term in content]
    shipping_hits = [term for term in SHIPPING_TERMS if term in content]
    has_number = bool(NUMBER_RE.search(content))
    noise = is_noise(content)
    generic = is_generic(content)
    shipping_only = (
        bool(shipping_hits)
        and not attributes
        and not decision_hits
        and not experience_hits
        and not defect_hits
    )

    if noise:
        specificity = 0
        product_experience = 0
        decision_value = 0
        clarity = 0
        noise_penalty = -2
    else:
        specificity = 0
        if attributes or has_number or decision_hits or defect_hits:
            specificity = 1
        if (
            has_number
            or len(set(attributes)) >= 2
            or len(decision_hits) >= 2
            or (defect_hits and len(words) >= 4)
        ):
            specificity = 2

        product_experience = 0
        if experience_hits:
            product_experience = 1
        if experience_hits and (has_number or attributes or len(words) >= 20):
            product_experience = 2

        decision_value = 0
        if attributes or decision_hits or defect_hits:
            decision_value = 1
        if (
            len(set(attributes)) >= 2
            or decision_hits
            or defect_hits
            or (attributes and has_number)
        ):
            decision_value = 2

        clarity = 1 if len(words) >= 1 else 0
        noise_penalty = 0
        if generic or shipping_only:
            specificity = min(specificity, 1)
            product_experience = 0
            decision_value = 0
        if generic and len(words) <= 3:
            specificity = 0
        spam_terms = ("nhận xu", "đánh giá nhận xu", "ủng hộ shop", "mua ngay")
        if any(term in content for term in spam_terms) and not attributes:
            noise_penalty = -1

    score = (
        specificity
        + product_experience
        + decision_value
        + clarity
        + noise_penalty
    )

    flags: set[str] = set()
    if noise:
        flags.add("noise_text")
    if generic:
        flags.add("generic_only")
    if shipping_only:
        flags.add("shipping_only")
    if "shop" in content and not attributes:
        flags.add("seller_service_only")
    if len(words) <= 5 and attributes and not generic:
        flags.add("too_short_but_specific")
    if defect_hits:
        flags.add("contains_product_defect")
    if experience_hits or has_number:
        flags.add("contains_usage_context")
    if any(term in content for term in ("so với", "rẻ hơn", "đắt hơn", "hời hơn", "giống loại")):
        flags.add("contains_comparison")
    if any(term in content for term in ("size", "kích thước", "rộng", "chật", "dài", "ngắn")):
        flags.add("contains_size_fit_info")
    if any(term in content for term in ("độ bền", "bền", "dùng lâu", "lâu dài")):
        flags.add("contains_durability_info")
    if any(term in content for term in ("đóng gói", "bao bì", "hộp", "móp")):
        flags.add("contains_packaging_info")

    original = str(row.get("manual_label", "uncertain"))
    if score >= 4:
        label = 1
    elif score <= 1:
        label = 0
    else:
        flags.add("ambiguous")
        concrete_signal = bool(attributes or decision_hits or defect_hits or has_number)
        if concrete_signal:
            label = 1
        elif original in {"0", "1"}:
            label = int(original)
        else:
            label = 0

    if score in (2, 3):
        confidence = 0.62 if original == "uncertain" else 0.68
    elif score in (1, 4):
        confidence = 0.82
    else:
        confidence = 0.92
    if original in {"0", "1"} and int(original) == label:
        confidence = min(0.96, confidence + 0.04)
    elif original in {"0", "1"} and int(original) != label:
        confidence = min(confidence, 0.72)
        flags.add("ambiguous")

    reason_parts = []
    if noise:
        reason_parts.append("Nội dung rỗng hoặc chỉ chứa ký tự không mang thông tin.")
    elif generic:
        reason_parts.append("Review chỉ khen/chê chung chung, thiếu chi tiết sản phẩm.")
    else:
        if attributes:
            reason_parts.append(
                "Review đề cập thuộc tính cụ thể: " + ", ".join(attributes[:3]) + "."
            )
        if has_number:
            reason_parts.append("Review có thông tin định lượng hoặc thời gian cụ thể.")
        if experience_hits:
            reason_parts.append("Review thể hiện trải nghiệm sử dụng thực tế.")
        if decision_hits:
            reason_parts.append("Review nêu điều kiện, so sánh hoặc hạn chế giúp cân nhắc.")
        if defect_hits:
            reason_parts.append(
                "Review nêu lỗi hoặc hạn chế cụ thể: "
                + ", ".join(defect_hits[:3])
                + "."
            )
        if shipping_only:
            reason_parts.append(
                "Nội dung chủ yếu nói về giao hàng/shop, không đánh giá sản phẩm."
            )
    if not reason_parts:
        reason_parts.append(
            "Review có một phần thông tin nhưng giá trị hỗ trợ quyết định còn hạn chế."
        )

    item = {
        "annotation_id": str(row["annotation_id"]),
        "is_helpful": label,
        "helpfulness_score": score,
        "confidence": round(confidence, 3),
        "reason": " ".join(reason_parts),
        "specificity": specificity,
        "product_experience": product_experience,
        "decision_value": decision_value,
        "clarity": clarity,
        "noise_penalty": noise_penalty,
        "quality_flags": sorted(flags),
        "label_version": "llm_helpfulness_v1_1",
        "label_source": "codex_rule_adjudicated",
        "adjudicated": True,
        "needs_human_review": (
            confidence < 0.7
            or "ambiguous" in flags
            or "possible_fake_or_spam" in flags
        ),
    }
    return refine_flags(item, text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = load_source()
    existing = load_checkpoint()
    completed = dict(existing)

    for _, row in source.iterrows():
        annotation_id = str(row["annotation_id"])
        if annotation_id in completed:
            continue
        completed[annotation_id] = score_review(row)

    create_outputs(source, completed)

    output = pd.read_csv(GOLD_PATH)
    report = json.loads(REPORT_PATH.read_text())
    report["source_distribution"] = {
        str(key): int(value)
        for key, value in output["label_source"].value_counts().items()
    }
    report["completion_method"] = (
        "840 Gemini labels retained; remaining rows scored with the PDF rubric "
        "and locally adjudicated. Low-confidence/disagreement cases are queued."
    )
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(output)}")
    print(f"Labels: {output['is_helpful'].value_counts().sort_index().to_dict()}")
    print(f"Sources: {output['label_source'].value_counts().to_dict()}")
    print(f"Human review queue: {len(pd.read_csv(REVIEW_QUEUE_PATH))}")
    print(f"Gold data: {GOLD_PATH}")


if __name__ == "__main__":
    main()
