from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "kaggle_gemini_helpfulness_labeling.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def main() -> None:
    nb = {
        "cells": [
            md(
                """
# Gemini LLM Labeling cho Tiki Review Helpfulness

Notebook này dùng Gemini API để gán nhãn review theo rubric `helpfulness`.
Thiết kế cho Kaggle:

- Đọc `data_labeling_raw_9k.csv` nếu có trong `/kaggle/input`.
- Nếu không có, đọc `data_labeling.csv`.
- Nếu không có, đọc toàn bộ CSV trong folder `batches`.
- Ghi checkpoint JSONL sau mỗi batch để có thể resume khi Kaggle dừng session hoặc hết quota.
- Xuất `data_labeling_labeled.csv`, `llm_helpfulness_labels.csv`, `human_review_queue.csv` và `labeling_report.json`.

Trước khi chạy:

1. Bật Internet trong Kaggle Notebook.
2. Thêm Kaggle Secrets:
   - `GEMINI_API_KEY`
   - tuỳ chọn thêm `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, ...
   - hoặc `GEMINI_API_KEYS` là nhiều key phân cách bằng dấu phẩy.
3. Upload `data_labeling_raw_9k.csv`, `data_labeling.csv` hoặc các file batch CSV thành Kaggle Dataset rồi Add Input vào notebook.
"""
            ),
            md("## 1. Cài đặt và import"),
            code(
                """
!pip install -q requests
"""
            ),
            code(
                """
import concurrent.futures
import itertools
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from kaggle_secrets import UserSecretsClient

pd.set_option("display.max_colwidth", 160)
"""
            ),
            md("## 2. Cấu hình"),
            code(
                """
# Gemini model.
GEMINI_MODEL = "gemini-2.5-flash"

# Batch nhỏ giúp JSON ổn định hơn. Có thể tăng lên 20 nếu review ngắn và quota khỏe.
BATCH_SIZE = 12

# Chạy song song khi có nhiều API key. Nếu chỉ có 1 key, nên để 1.
MAX_WORKERS = 2

# Dùng LIMIT để test nhanh trước. Đặt None để chạy toàn bộ.
START = 0
LIMIT = 50

MAX_RETRIES = 5
SLEEP_BETWEEN_COMPLETED_BATCHES = 0.25

OUTPUT_DIR = Path("/kaggle/working/gemini_labeling_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "llm_helpfulness_checkpoint.jsonl"
LABELS_PATH = OUTPUT_DIR / "llm_helpfulness_labels.csv"
LABELED_DATA_PATH = OUTPUT_DIR / "data_labeling_labeled.csv"
HUMAN_REVIEW_PATH = OUTPUT_DIR / "human_review_queue.csv"
REPORT_PATH = OUTPUT_DIR / "labeling_report.json"

print("OUTPUT_DIR:", OUTPUT_DIR)
"""
            ),
            md("## 3. Lấy Gemini API key từ Kaggle Secrets"),
            code(
                """
def get_secret(name: str) -> str:
    try:
        return UserSecretsClient().get_secret(name) or ""
    except Exception:
        return ""


raw_keys = []
primary_key = get_secret("GEMINI_API_KEY").strip()
if primary_key:
    raw_keys.append(primary_key)

multi_key_value = get_secret("GEMINI_API_KEYS").strip()
if multi_key_value:
    raw_keys.extend([item.strip() for item in multi_key_value.split(",") if item.strip()])

for index in range(2, 11):
    value = get_secret(f"GEMINI_API_KEY_{index}").strip()
    if value:
        raw_keys.append(value)

API_KEYS = list(dict.fromkeys(raw_keys))
if not API_KEYS:
    raise ValueError(
        "Chưa cấu hình Kaggle Secret GEMINI_API_KEY hoặc GEMINI_API_KEYS."
    )

print("Số Gemini API key:", len(API_KEYS))
"""
            ),
            md("## 4. Đọc dữ liệu input"),
            code(
                """
def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def find_input_file() -> Path | None:
    roots = [Path("/kaggle/input"), Path(".")]
    preferred_names = ["data_labeling_raw_9k.csv", "data_labeling.csv", "data_gold.csv"]
    for root in roots:
        if not root.exists():
            continue
        for name in preferred_names:
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    return None


def load_input_data() -> pd.DataFrame:
    input_file = find_input_file()
    if input_file is not None:
        print("Đọc input file:", input_file)
        df = pd.read_csv(input_file)
        df["source_file"] = df.get("source_file", input_file.name)
    else:
        batch_files = sorted(Path("/kaggle/input").rglob("*.csv"))
        if not batch_files:
            batch_files = sorted(Path(".").rglob("Data/batches/*.csv"))
        if not batch_files:
            raise FileNotFoundError(
                "Không tìm thấy data_labeling.csv hoặc batch CSV trong /kaggle/input."
            )
        frames = []
        for path in batch_files:
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        print("Đọc batch files:", len(batch_files))

    if "annotation_id" not in df.columns:
        df["annotation_id"] = [f"ROW_{i:06d}" for i in range(len(df))]

    for col in ["title", "content", "review_text"]:
        if col in df.columns:
            df[col] = df[col].map(clean_text)

    if "review_text" not in df.columns:
        title = df["title"] if "title" in df.columns else ""
        content = df["content"] if "content" in df.columns else ""
        df["review_text"] = (
            title.fillna("").astype(str).str.strip()
            + ". "
            + content.fillna("").astype(str).str.strip()
        ).str.strip(". ")

    for col, default in [
        ("review_id", ""),
        ("product_id", ""),
        ("user_id", ""),
        ("rating", 0),
        ("purchased_int", 0),
        ("word_count", None),
        ("sample_group", ""),
        ("title", ""),
        ("content", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df["annotation_id"] = df["annotation_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
    df["purchased_int"] = (
        pd.to_numeric(df["purchased_int"], errors="coerce").fillna(0).astype(int)
    )
    if df["word_count"].isna().any():
        df["word_count"] = df["review_text"].str.split().str.len()
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)

    before = len(df)
    df = df[df["review_text"].str.len() > 0].drop_duplicates("annotation_id").reset_index(drop=True)
    print(f"Rows hợp lệ: {len(df)}/{before}")
    return df


source_df = load_input_data()
selected_df = source_df.iloc[START : START + LIMIT if LIMIT is not None else None].copy()
print("Selected rows:", len(selected_df))
display(selected_df.head(3))
"""
            ),
            md("## 5. Rubric và prompt gán nhãn"),
            code(
                '''
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
- Không suy diễn thông tin không có trong title/content/review_text.
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


def build_prompt(records: list[dict[str, Any]]) -> str:
    compact = []
    for row in records:
        compact.append(
            {
                "annotation_id": str(row["annotation_id"]),
                "review_id": str(row.get("review_id", "")),
                "product_id": str(row.get("product_id", "")),
                "rating": int(row.get("rating", 0)),
                "purchased_int": int(row.get("purchased_int", 0)),
                "title": clean_text(row.get("title", "")),
                "content": clean_text(row.get("content", "")),
                "review_text": clean_text(row.get("review_text", "")),
            }
        )

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
        + "\\nHãy gán nhãn toàn bộ review dưới đây. Phải trả đủ đúng "
        f"{len(compact)} kết quả và giữ nguyên annotation_id.\\n\\n"
        + "INPUT:\\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\\n\\nOUTPUT SCHEMA:\\n"
        + json.dumps(schema, ensure_ascii=False)
    )
'''
            ),
            md("## 6. Chuẩn hoá response và gọi Gemini"),
            code(
                '''
def canonical_annotation_id(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"MANUAL_(\\d+)", text, flags=re.IGNORECASE)
    if not match:
        return text
    return f"MANUAL_{int(match.group(1)):05d}"


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
        "candidate_llm_is_helpful": label,
        "candidate_llm_helpfulness_score": score,
        "candidate_llm_confidence": round(confidence, 3),
        "candidate_llm_reason": clean_text(item.get("reason", "")),
        "candidate_llm_specificity": normalized_evidence["specificity"],
        "candidate_llm_product_experience": normalized_evidence["product_experience"],
        "candidate_llm_decision_value": normalized_evidence["decision_value"],
        "candidate_llm_clarity": normalized_evidence["clarity"],
        "candidate_llm_noise_penalty": normalized_evidence["noise_penalty"],
        "candidate_llm_quality_flags": flags,
        "candidate_llm_label_version": "gemini_helpfulness_v1",
        "candidate_llm_model": GEMINI_MODEL,
    }


def generation_config() -> dict[str, Any]:
    config = {
        "responseMimeType": "application/json",
        "temperature": 0.0,
        "maxOutputTokens": 8192,
    }
    if GEMINI_MODEL.startswith("gemini-2.5"):
        config["thinkingConfig"] = {"thinkingBudget": 512}
    return config


def parse_gemini_labels(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)

    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("labels"), list):
        return parsed["labels"]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "annotation_id" in parsed:
        return [parsed]

    # Gemini đôi khi bọc response trong một key khác dù đã yêu cầu schema.
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if "annotation_id" in value[0]:
                    return value

    preview = json.dumps(parsed, ensure_ascii=False)[:500]
    raise ValueError(f"Response does not contain labels; preview={preview}")


def call_gemini(
    records: list[dict[str, Any]],
    api_keys: list[str],
    start_key_index: int = 0,
) -> list[dict[str, Any]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    prompt = build_prompt(records)
    expected_ids = {canonical_annotation_id(row["annotation_id"]) for row in records}
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        key_index = (start_key_index + attempt - 1) % len(api_keys)
        api_key = api_keys[key_index]
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": generation_config(),
                },
                timeout=120,
            )
            if response.status_code == 429:
                retry_seconds = min(60, 5 * attempt)
                if attempt < MAX_RETRIES:
                    print(
                        f"Quota limited key {key_index + 1}/{len(api_keys)}; retry in {retry_seconds}s",
                        flush=True,
                    )
                    time.sleep(2 if len(api_keys) > 1 else retry_seconds)
                    continue
            response.raise_for_status()
            payload = response.json()
            candidate = payload["candidates"][0]
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason != "STOP":
                raise ValueError(f"finishReason={finish_reason}")
            text = candidate["content"]["parts"][0]["text"]
            raw_labels = parse_gemini_labels(text)
            normalized = [normalize_label(item) for item in raw_labels]
            actual_ids = {item["annotation_id"] for item in normalized}
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(f"ID mismatch; missing={missing}, extra={extra}")
            return normalized
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(min(2**attempt, 20))

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_error}")
'''
            ),
            md("## 7. Checkpoint, chạy labeling và resume"),
            code(
                '''
def load_checkpoint() -> dict[str, dict[str, Any]]:
    labels = {}
    if not CHECKPOINT_PATH.exists():
        return labels
    for line in CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        labels[str(item["annotation_id"])] = item
    return labels


def append_checkpoint(labels: list[dict[str, Any]]) -> None:
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
        for item in labels:
            handle.write(json.dumps(item, ensure_ascii=False) + "\\n")
        handle.flush()


def save_outputs(labels_by_id: dict[str, dict[str, Any]]) -> pd.DataFrame:
    label_df = pd.DataFrame(labels_by_id.values())
    if label_df.empty:
        print("Chưa có label để lưu.")
        return pd.DataFrame()

    label_df["candidate_llm_quality_flags"] = label_df["candidate_llm_quality_flags"].map(
        lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else "[]"
    )
    label_df.to_csv(LABELS_PATH, index=False, encoding="utf-8-sig")

    output = source_df.merge(label_df, on="annotation_id", how="left", validate="one_to_one")
    output.to_csv(LABELED_DATA_PATH, index=False, encoding="utf-8-sig")

    review_queue = output[
        output["candidate_llm_is_helpful"].notna()
        & (
            output["candidate_llm_confidence"].lt(0.7)
            | output["candidate_llm_quality_flags"].str.contains("ambiguous", na=False)
            | output["candidate_llm_quality_flags"].str.contains("possible_fake_or_spam", na=False)
        )
    ].copy()
    review_queue.to_csv(HUMAN_REVIEW_PATH, index=False, encoding="utf-8-sig")

    labeled = output[output["candidate_llm_is_helpful"].notna()].copy()
    report = {
        "total_rows": int(len(source_df)),
        "selected_rows": int(len(selected_df)),
        "labeled_rows": int(len(labeled)),
        "label_distribution": {
            str(k): int(v)
            for k, v in labeled["candidate_llm_is_helpful"].value_counts().sort_index().items()
        },
        "helpful_rate": round(float(labeled["candidate_llm_is_helpful"].mean()), 4)
        if len(labeled) else None,
        "mean_confidence": round(float(labeled["candidate_llm_confidence"].mean()), 4)
        if len(labeled) else None,
        "human_review_queue_rows": int(len(review_queue)),
        "model": GEMINI_MODEL,
        "batch_size": BATCH_SIZE,
        "max_workers": MAX_WORKERS,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print("-", LABELS_PATH)
    print("-", LABELED_DATA_PATH)
    print("-", HUMAN_REVIEW_PATH)
    print("-", REPORT_PATH)
    print(report)
    return output


labels_by_id = load_checkpoint()
pending_df = selected_df[~selected_df["annotation_id"].astype(str).isin(labels_by_id.keys())]
pending_records = pending_df.to_dict(orient="records")

print("Already labeled from checkpoint:", len(labels_by_id))
print("Pending rows:", len(pending_records))

batches = [
    pending_records[offset : offset + BATCH_SIZE]
    for offset in range(0, len(pending_records), BATCH_SIZE)
]
print("Pending batches:", len(batches))

key_counter = itertools.count()


def process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_key_index = next(key_counter) % len(API_KEYS)
    try:
        return call_gemini(batch, API_KEYS, start_key_index=start_key_index)
    except Exception as exc:
        if len(batch) == 1:
            raise
        print(
            "Batch schema failed; retry từng review. "
            f"First annotation_id={batch[0]['annotation_id']}. Error={exc}",
            flush=True,
        )
        recovered = []
        for row in batch:
            recovered.extend(
                call_gemini([row], API_KEYS, start_key_index=next(key_counter) % len(API_KEYS))
            )
            time.sleep(0.2)
        return recovered


if batches:
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_labels = future.result()
            except Exception as exc:
                print("Batch failed. First annotation_id:", batch[0]["annotation_id"])
                print("Error:", repr(exc))
                raise

            append_checkpoint(batch_labels)
            for item in batch_labels:
                labels_by_id[item["annotation_id"]] = item
            completed += 1
            print(
                f"Completed {completed}/{len(batches)} batches; total labels={len(labels_by_id)}",
                flush=True,
            )
            time.sleep(SLEEP_BETWEEN_COMPLETED_BATCHES)

output_df = save_outputs(labels_by_id)
display(output_df.head(5) if not output_df.empty else output_df)
'''
            ),
            md("## 8. Kiểm tra nhanh kết quả"),
            code(
                """
if not output_df.empty:
    labeled = output_df[output_df["candidate_llm_is_helpful"].notna()].copy()
    display(labeled["candidate_llm_is_helpful"].value_counts(dropna=False).rename("count").to_frame())
    display(
        labeled[
            [
                "annotation_id",
                "rating",
                "review_text",
                "candidate_llm_is_helpful",
                "candidate_llm_helpfulness_score",
                "candidate_llm_confidence",
                "candidate_llm_reason",
                "candidate_llm_quality_flags",
            ]
        ].sample(min(10, len(labeled)), random_state=42)
    )
"""
            ),
            md(
                """
## 9. Chạy toàn bộ dữ liệu

Sau khi test ổn với `LIMIT = 50`, quay lại cell cấu hình và đổi:

```python
LIMIT = None
BATCH_SIZE = 12  # hoặc 20 nếu response ổn định
MAX_WORKERS = min(so_api_key, 3)
```

Nếu notebook bị dừng, chạy lại từ đầu. Cell labeling sẽ đọc
`llm_helpfulness_checkpoint.jsonl` trong `/kaggle/working/gemini_labeling_outputs`
và bỏ qua các dòng đã gán nhãn.
"""
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "none"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
