from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_SOURCE = ROOT / "Data" / "raw_reviews.csv"
LABEL_SOURCE = ROOT / "Data" / "gold_data" / "data_gold.csv"
OUTPUT = ROOT / "Data" / "gold_data" / "tiki_reviews_full_labeled_9k.csv"

MERGE_KEYS = ["review_id", "product_id", "user_id"]

LABEL_COLUMNS = [
    "annotation_id",
    "sample_group",
    "review_id",
    "product_id",
    "user_id",
    "is_helpful",
    "helpfulness_score",
    "confidence",
    "reason",
    "specificity",
    "product_experience",
    "decision_value",
    "clarity",
    "noise_penalty",
    "quality_flags",
    "label_version",
    "label_source",
    "adjudicated",
    "needs_human_review",
]


def build_review_text(frame: pd.DataFrame) -> pd.Series:
    title = frame["title"].fillna("").astype(str).str.strip()
    content = frame["content"].fillna("").astype(str).str.strip()
    return (title + ". " + content).str.strip(". ")


def main() -> None:
    if not RAW_SOURCE.exists():
        raise FileNotFoundError(f"Missing raw source: {RAW_SOURCE}")
    if not LABEL_SOURCE.exists():
        raise FileNotFoundError(f"Missing label source: {LABEL_SOURCE}")

    raw = pd.read_csv(RAW_SOURCE)
    labels = pd.read_csv(LABEL_SOURCE)

    missing_raw = [column for column in MERGE_KEYS if column not in raw.columns]
    missing_labels = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing_raw:
        raise ValueError(f"Raw source missing merge keys: {missing_raw}")
    if missing_labels:
        raise ValueError(f"Label source missing columns: {missing_labels}")

    if raw["review_id"].duplicated().any():
        raise ValueError("raw_reviews.csv contains duplicated review_id")
    if labels["review_id"].duplicated().any():
        raise ValueError("data_gold.csv contains duplicated review_id")

    label_frame = labels[LABEL_COLUMNS].copy()
    merged = raw.merge(
        label_frame,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(labels):
        raw_keys = set(map(tuple, raw[MERGE_KEYS].to_numpy()))
        label_keys = set(map(tuple, labels[MERGE_KEYS].to_numpy()))
        missing = sorted(label_keys - raw_keys)[:10]
        raise ValueError(
            f"Expected {len(labels)} labeled rows, got {len(merged)}. "
            f"Missing key examples: {missing}"
        )

    merged["review_text"] = build_review_text(merged)
    merged["word_count"] = merged["review_text"].str.split().str.len().astype(int)
    merged["purchased_int"] = merged["purchased"].astype(bool).astype(int)
    merged["help"] = pd.to_numeric(
        merged["is_helpful"],
        errors="raise",
    ).astype(int)

    output_columns = [
        "review_id",
        "product_id",
        "user_id",
        "rating",
        "title",
        "content",
        "review_text",
        "created_at",
        "helpful_count",
        "purchased",
        "purchased_int",
        "crawled_at",
        "word_count",
        "annotation_id",
        "sample_group",
        "help",
        "helpfulness_score",
        "confidence",
        "reason",
        "specificity",
        "product_experience",
        "decision_value",
        "clarity",
        "noise_penalty",
        "quality_flags",
        "label_version",
        "label_source",
        "adjudicated",
        "needs_human_review",
    ]
    merged = merged[output_columns].sort_values("annotation_id").reset_index(drop=True)
    merged.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    manifest = {
        "raw_source": str(RAW_SOURCE.relative_to(ROOT)),
        "label_source": str(LABEL_SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "merge_keys": MERGE_KEYS,
        "raw_rows": int(len(raw)),
        "label_rows": int(len(labels)),
        "output_rows": int(len(merged)),
        "columns": output_columns,
        "target_column": "help",
        "target_source_column": "is_helpful",
        "help_distribution": {
            str(key): int(value)
            for key, value in merged["help"].value_counts().sort_index().items()
        },
    }
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Help distribution: {manifest['help_distribution']}")


if __name__ == "__main__":
    main()
