from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "Data" / "batches"
OUTPUT_DIR = ROOT / "Data" / "gold_data"
CHECKPOINT_PATH = OUTPUT_DIR / "llm_helpfulness_v1_checkpoint.jsonl"
GOLD_PATH = OUTPUT_DIR / "tiki_reviews_gold_llm_helpfulness_v1.csv"
REVIEW_QUEUE_PATH = OUTPUT_DIR / "human_review_queue_llm_helpfulness_v1.csv"
REPORT_PATH = OUTPUT_DIR / "labeling_report_llm_helpfulness_v1.json"

ALLOWED_FLAGS = {
    "generic_only",
    "too_short_but_specific",
    "noise_text",
    "duplicate_like",
    "shipping_only",
    "seller_service_only",
    "irrelevant",
    "possible_fake_or_spam",
    "ambiguous",
    "contains_product_defect",
    "contains_usage_context",
    "contains_comparison",
    "contains_size_fit_info",
    "contains_durability_info",
    "contains_packaging_info",
}

SYSTEM_PROMPT = """Bạn là trợ lý gán nhãn dữ liệu cho bài toán Product Review
Helpfulness Detection trên sàn thương mại điện tử Tiki.

Mục tiêu duy nhất: xác định review có cung cấp thông tin cụ thể giúp người mua
khác ra quyết định hay không.

Quy tắc bắt buộc:
- Không gán nhãn chỉ dựa trên rating, độ dài hoặc trạng thái đã mua hàng.
- Review ngắn vẫn helpful nếu nêu thuộc tính cụ thể: "Pin yếu", "Vải mỏng".
- Review dài vẫn unhelpful nếu chỉ cảm ơn, khen/chê chung chung hoặc lan man.
- "Giao hàng nhanh" hoặc chỉ khen shop mặc định unhelpful.
- Đóng gói/giao hàng helpful khi vấn đề ảnh hưởng trực tiếp tới sản phẩm.
- Review rating thấp helpful nếu mô tả lỗi hoặc trải nghiệm cụ thể.
- Noise, emoji-only, spam, nội dung không liên quan là unhelpful.
- Không suy diễn thông tin không có trong title/content.
- Trả về JSON hợp lệ, không markdown và không giải thích ngoài JSON.

Rubric:
- specificity: 0-2
- product_experience: 0-2
- decision_value: 0-2
- clarity: 0-1
- noise_penalty: -2 đến 0
- helpfulness_score là tổng đúng của 5 điểm trên, từ -2 đến 7.

Quy đổi:
- score >= 4: is_helpful = 1
- score <= 1: is_helpful = 0
- score 2-3: case borderline. Vẫn phải chọn nhãn 0/1 dựa trên câu hỏi:
  "Review này có giúp người mua biết thêm điều gì cụ thể về sản phẩm hoặc rủi
  ro mua hàng không?". Thêm flag ambiguous và giảm confidence.

Chỉ dùng quality_flags trong danh sách:
generic_only, too_short_but_specific, noise_text, duplicate_like, shipping_only,
seller_service_only, irrelevant, possible_fake_or_spam, ambiguous,
contains_product_defect, contains_usage_context, contains_comparison,
contains_size_fit_info, contains_durability_info, contains_packaging_info.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel Tiki reviews with Gemini")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def load_source() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype={"manual_label": str})
        frame["source_file"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data["annotation_id"].duplicated().any():
        raise ValueError("annotation_id must be unique")
    data["title"] = data["title"].map(clean_text)
    data["content"] = data["content"].map(clean_text)
    data["review_text"] = (
        data["title"].str.strip() + ". " + data["content"].str.strip()
    ).str.strip(". ")
    return data


def load_checkpoint() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if not CHECKPOINT_PATH.exists():
        return labels
    for line in CHECKPOINT_PATH.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        labels[str(item["annotation_id"])] = item
    return labels


def build_prompt(records: list[dict[str, Any]]) -> str:
    compact = [
        {
            "annotation_id": str(row["annotation_id"]),
            "review_id": str(row["review_id"]),
            "product_id": str(row["product_id"]),
            "rating": int(row["rating"]),
            "purchased_int": int(row["purchased_int"]),
            "title": row["title"],
            "content": row["content"],
        }
        for row in records
    ]
    schema = {
        "labels": [
            {
                "annotation_id": "MANUAL_00001",
                "is_helpful": 0,
                "confidence": 0.9,
                "reason": "Giải thích ngắn gọn bằng tiếng Việt.",
                "evidence": {
                    "specificity": 0,
                    "product_experience": 0,
                    "decision_value": 0,
                    "clarity": 1,
                    "noise_penalty": 0,
                },
                "quality_flags": ["generic_only"],
            }
        ]
    }
    return (
        SYSTEM_PROMPT
        + "\nHãy gán nhãn toàn bộ review dưới đây. Phải trả đủ đúng "
        f"{len(compact)} kết quả và giữ nguyên annotation_id.\n\n"
        + "INPUT:\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\n\nOUTPUT SCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def generation_config(model: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "responseMimeType": "application/json",
        "temperature": 0.0,
        "maxOutputTokens": 8192,
    }
    if model.startswith("gemini-2.5"):
        config["thinkingConfig"] = {"thinkingBudget": 512}
    return config


def call_gemini(
    records: list[dict[str, Any]],
    api_keys: list[str],
    model: str,
    max_retries: int,
    start_key_index: int = 0,
) -> list[dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = build_prompt(records)
    expected_ids = {str(row["annotation_id"]) for row in records}
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        key_index = (start_key_index + attempt - 1) % len(api_keys)
        api_key = api_keys[key_index]
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": generation_config(model),
                },
                timeout=120,
            )
            if response.status_code == 429:
                payload = response.json() if response.content else {}
                details = payload.get("error", {}).get("details", [])
                retry_seconds = min(60, 5 * attempt)
                for detail in details:
                    retry_delay = detail.get("retryDelay")
                    if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                        try:
                            retry_seconds = min(120, float(retry_delay[:-1]) + 1)
                        except ValueError:
                            pass
                if attempt < max_retries:
                    print(
                        f"Gemini key {key_index + 1}/{len(api_keys)} quota limited; "
                        f"switching key/retrying in {retry_seconds:.0f}s "
                        f"(attempt {attempt}/{max_retries})",
                        flush=True,
                    )
                    if len(api_keys) == 1:
                        time.sleep(retry_seconds)
                    else:
                        time.sleep(min(retry_seconds, 2))
                    continue
                message = payload.get("error", {}).get("message", "quota exceeded")
                raise RuntimeError(f"Gemini quota exceeded: {message}")
            response.raise_for_status()
            payload = response.json()
            candidate = payload["candidates"][0]
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason != "STOP":
                raise ValueError(f"finishReason={finish_reason}")
            text = candidate["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            raw_labels = parsed.get("labels") if isinstance(parsed, dict) else None
            if not isinstance(raw_labels, list):
                raise ValueError("Response does not contain a labels list")
            normalized = [normalize_label(item) for item in raw_labels]
            actual_ids = {item["annotation_id"] for item in normalized}
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(f"ID mismatch; missing={missing}, extra={extra}")
            return normalized
        except RuntimeError:
            raise
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Gemini failed after {max_retries} attempts: {last_error}")


def integer_in_range(value: Any, low: int, high: int, field: str) -> int:
    number = int(value)
    if number < low or number > high:
        raise ValueError(f"{field} must be between {low} and {high}")
    return number


def normalize_label(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Each label must be an object")
    annotation_id = canonical_annotation_id(item["annotation_id"])
    evidence = item.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{annotation_id}: missing evidence")
    normalized_evidence = {
        "specificity": integer_in_range(evidence.get("specificity"), 0, 2, "specificity"),
        "product_experience": integer_in_range(
            evidence.get("product_experience"), 0, 2, "product_experience"
        ),
        "decision_value": integer_in_range(
            evidence.get("decision_value"), 0, 2, "decision_value"
        ),
        "clarity": integer_in_range(evidence.get("clarity"), 0, 1, "clarity"),
        "noise_penalty": integer_in_range(
            evidence.get("noise_penalty"), -2, 0, "noise_penalty"
        ),
    }
    score = sum(normalized_evidence.values())
    label = integer_in_range(item.get("is_helpful"), 0, 1, "is_helpful")
    if score >= 4:
        label = 1
    elif score <= 1:
        label = 0
    flags = item.get("quality_flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = sorted({str(flag) for flag in flags if str(flag) in ALLOWED_FLAGS})
    if 2 <= score <= 3 and "ambiguous" not in flags:
        flags.append("ambiguous")
    confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
    if 2 <= score <= 3:
        confidence = min(confidence, 0.79)
    return {
        "annotation_id": annotation_id,
        "is_helpful": label,
        "helpfulness_score": score,
        "confidence": round(confidence, 3),
        "reason": clean_text(item.get("reason", "")),
        "specificity": normalized_evidence["specificity"],
        "product_experience": normalized_evidence["product_experience"],
        "decision_value": normalized_evidence["decision_value"],
        "clarity": normalized_evidence["clarity"],
        "noise_penalty": normalized_evidence["noise_penalty"],
        "quality_flags": flags,
        "label_version": "llm_helpfulness_v1",
        "label_source": "llm",
        "adjudicated": False,
        "needs_human_review": (
            confidence < 0.7
            or "ambiguous" in flags
            or "possible_fake_or_spam" in flags
        ),
    }


def canonical_annotation_id(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"MANUAL_(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return text
    return f"MANUAL_{int(match.group(1)):05d}"


def refine_flags(label: dict[str, Any], review_text: str) -> dict[str, Any]:
    text = review_text.lower()
    flags = set(label["quality_flags"])
    if label["is_helpful"] == 0:
        flags.discard("too_short_but_specific")
    if label["is_helpful"] == 1:
        flags.discard("generic_only")

    defect_terms = (
        "lỗi",
        "hỏng",
        "bể",
        "vỡ",
        "móp",
        "chảy",
        "rè",
        "không lên",
        "không hoạt động",
        "tụt pin",
        "nóng",
        "rách",
        "sai mẫu",
        "thiếu",
    )
    if not any(term in text for term in defect_terms):
        flags.discard("contains_product_defect")

    durability_terms = (
        "độ bền",
        "bền",
        "sau một tháng",
        "sau 1 tháng",
        "dùng lâu",
        "lâu dài",
        "thời gian dài",
    )
    if not any(term in text for term in durability_terms):
        flags.discard("contains_durability_info")

    comparison_terms = ("so với", "hơn loại", "rẻ hơn", "hời hơn", "giống loại")
    if any(term in text for term in comparison_terms):
        flags.add("contains_comparison")

    label["quality_flags"] = sorted(flags)
    label["needs_human_review"] = (
        label["confidence"] < 0.7
        or "ambiguous" in flags
        or "possible_fake_or_spam" in flags
    )
    return label


def append_checkpoint(labels: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
        for item in labels:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def create_outputs(source: pd.DataFrame, labels: dict[str, dict[str, Any]]) -> None:
    label_frame = pd.DataFrame(labels.values())
    if label_frame.empty:
        return
    output = source.merge(label_frame, on="annotation_id", how="left", validate="one_to_one")
    output["original_manual_label"] = output["manual_label"].astype(str)
    output["agrees_with_original"] = (
        output["original_manual_label"].isin(["0", "1"])
        & (output["original_manual_label"].astype(str) == output["is_helpful"].astype("Int64").astype(str))
    )
    output["quality_flags"] = output["quality_flags"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
        if isinstance(value, list)
        else "[]"
    )
    output.to_csv(GOLD_PATH, index=False, encoding="utf-8-sig")

    review_queue = output[
        output["is_helpful"].notna() & output["needs_human_review"].eq(True)
    ].copy()
    review_queue.to_csv(REVIEW_QUEUE_PATH, index=False, encoding="utf-8-sig")

    labeled = output[output["is_helpful"].notna()].copy()
    report = {
        "total_source_rows": int(len(source)),
        "labeled_rows": int(len(labeled)),
        "unlabeled_rows": int(len(source) - len(labeled)),
        "label_distribution": {
            str(key): int(value)
            for key, value in labeled["is_helpful"].value_counts().sort_index().items()
        },
        "helpful_rate": round(float(labeled["is_helpful"].mean()), 4)
        if len(labeled)
        else None,
        "human_review_queue_rows": int(len(review_queue)),
        "mean_confidence": round(float(labeled["confidence"].mean()), 4)
        if len(labeled)
        else None,
        "agreement_with_original_binary_labels": round(
            float(
                labeled[labeled["original_manual_label"].isin(["0", "1"])][
                    "agrees_with_original"
                ].mean()
            ),
            4,
        )
        if len(labeled[labeled["original_manual_label"].isin(["0", "1"])])
        else None,
        "helpful_rate_by_rating": {
            str(key): round(float(value), 4)
            for key, value in labeled.groupby("rating")["is_helpful"].mean().items()
        },
        "helpful_rate_by_sample_group": {
            str(key): round(float(value), 4)
            for key, value in labeled.groupby("sample_group")["is_helpful"].mean().items()
        },
        "label_version": "llm_helpfulness_v1",
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    primary_key = os.getenv("GEMINI_API_KEY", "").strip()
    configured_keys = [
        key.strip()
        for key in os.getenv("GEMINI_API_KEYS", "").split(",")
        if key.strip()
    ]
    api_keys = list(dict.fromkeys(([primary_key] if primary_key else []) + configured_keys))
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_keys:
        print("Missing GEMINI_API_KEY or GEMINI_API_KEYS in .env", file=sys.stderr)
        return 1
    if args.batch_size < 1 or args.batch_size > 30:
        print("--batch-size must be between 1 and 30", file=sys.stderr)
        return 1
    if args.workers < 1 or args.workers > 6:
        print("--workers must be between 1 and 6", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    source = load_source()
    end = args.start + args.limit if args.limit is not None else None
    selected = source.iloc[args.start:end].copy()
    labels = load_checkpoint()
    pending = selected[
        ~selected["annotation_id"].astype(str).isin(labels.keys())
    ].to_dict(orient="records")

    print(f"Source rows: {len(source)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Already labeled: {len(labels)}")
    print(f"Pending in selection: {len(pending)}")
    print(f"Model: {model}")
    print(f"Unique Gemini keys: {len(api_keys)}")

    batches = [
        pending[offset : offset + args.batch_size]
        for offset in range(0, len(pending), args.batch_size)
    ]

    key_counter = itertools.count()

    def process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        start_key_index = next(key_counter) % len(api_keys)
        batch_labels = call_gemini(
            batch,
            api_keys,
            model,
            args.max_retries,
            start_key_index=start_key_index,
        )
        text_by_id = {
            str(row["annotation_id"]): f"{row['title']}. {row['content']}"
            for row in batch
        }
        return [
            refine_flags(item, text_by_id[item["annotation_id"]])
            for item in batch_labels
        ]

    completed = 0
    if args.workers == 1:
        results = (process_batch(batch) for batch in batches)
        for batch_labels in results:
            append_checkpoint(batch_labels)
            for item in batch_labels:
                labels[item["annotation_id"]] = item
            completed += len(batch_labels)
            print(
                f"Labeled {completed}/{len(pending)} pending rows; "
                f"total checkpoint={len(labels)}",
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): batch for batch in batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_labels = future.result()
                append_checkpoint(batch_labels)
                for item in batch_labels:
                    labels[item["annotation_id"]] = item
                completed += len(batch_labels)
                print(
                    f"Labeled {completed}/{len(pending)} pending rows; "
                    f"total checkpoint={len(labels)}",
                    flush=True,
                )
                if args.sleep:
                    time.sleep(args.sleep)

    create_outputs(source, labels)
    print(f"Gold data: {GOLD_PATH}")
    print(f"Human review queue: {REVIEW_QUEUE_PATH}")
    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
