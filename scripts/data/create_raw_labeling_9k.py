from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data" / "gold_data" / "data_labeling.csv"
OUTPUT = ROOT / "Data" / "gold_data" / "data_labeling_raw_9k.csv"

RAW_COLUMNS = [
    "annotation_id",
    "sample_group",
    "review_id",
    "product_id",
    "user_id",
    "rating",
    "purchased_int",
    "word_count",
    "title",
    "content",
    "review_text",
    "source_file",
]

LLM_OUTPUT_COLUMNS = [
    "llm_is_helpful",
    "llm_helpfulness_score",
    "llm_confidence",
    "llm_reason",
    "llm_specificity",
    "llm_product_experience",
    "llm_decision_value",
    "llm_clarity",
    "llm_noise_penalty",
    "llm_quality_flags",
    "llm_label_version",
    "llm_model",
]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    data = pd.read_csv(SOURCE, dtype={"annotation_id": str})
    missing = [column for column in RAW_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    output = data[RAW_COLUMNS].copy()
    for column in LLM_OUTPUT_COLUMNS:
        output[column] = ""

    if output["annotation_id"].duplicated().any():
        duplicates = output.loc[
            output["annotation_id"].duplicated(), "annotation_id"
        ].head(10)
        raise ValueError(f"annotation_id must be unique. Examples: {duplicates}")

    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "rows": int(len(output)),
        "raw_columns": RAW_COLUMNS,
        "llm_output_columns": LLM_OUTPUT_COLUMNS,
        "removed_columns": [
            column for column in data.columns if column not in RAW_COLUMNS
        ],
        "note": (
            "Raw 9k labeling file. Human labels, reference labels, Gemini/Codex "
            "labels and explanations are intentionally removed."
        ),
    }
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")


if __name__ == "__main__":
    main()
