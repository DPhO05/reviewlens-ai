from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data" / "gold_data" / "tiki_reviews_gold_llm_helpfulness_v1.csv"
OUTPUT = ROOT / "Data" / "gold_data" / "data_labeling.csv"

HUMAN_COLUMNS = {
    "annotator_id",
    "manual_label",
    "label_confidence",
    "label_notes",
    "original_manual_label",
    "agrees_with_original",
}

REFERENCE_RENAME = {
    "is_helpful": "reference_is_helpful",
    "helpfulness_score": "reference_helpfulness_score",
    "confidence": "reference_confidence",
    "reason": "reference_reason",
    "specificity": "reference_specificity",
    "product_experience": "reference_product_experience",
    "decision_value": "reference_decision_value",
    "clarity": "reference_clarity",
    "noise_penalty": "reference_noise_penalty",
    "quality_flags": "reference_quality_flags",
    "label_version": "reference_label_version",
    "label_source": "reference_label_source",
    "adjudicated": "reference_adjudicated",
    "needs_human_review": "reference_needs_human_review",
}

CANDIDATE_COLUMNS = [
    "candidate_llm_is_helpful",
    "candidate_llm_helpfulness_score",
    "candidate_llm_confidence",
    "candidate_llm_reason",
    "candidate_llm_specificity",
    "candidate_llm_product_experience",
    "candidate_llm_decision_value",
    "candidate_llm_clarity",
    "candidate_llm_noise_penalty",
    "candidate_llm_quality_flags",
    "candidate_llm_label_version",
    "candidate_llm_model",
]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    data = pd.read_csv(SOURCE, dtype={"annotation_id": str})
    output = data.drop(
        columns=[column for column in HUMAN_COLUMNS if column in data.columns]
    ).rename(columns=REFERENCE_RENAME)

    input_columns = [
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
    reference_columns = list(REFERENCE_RENAME.values())
    for column in CANDIDATE_COLUMNS:
        output[column] = ""

    output = output[input_columns + reference_columns + CANDIDATE_COLUMNS]
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    assert len(output) == len(data)
    assert output["annotation_id"].nunique() == len(output)
    assert not HUMAN_COLUMNS.intersection(output.columns)
    assert set(reference_columns).issubset(output.columns)
    assert set(CANDIDATE_COLUMNS).issubset(output.columns)
    for column in CANDIDATE_COLUMNS:
        assert output[column].fillna("").eq("").all()

    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "rows": len(output),
        "input_columns": input_columns,
        "reference_columns": reference_columns,
        "candidate_columns": CANDIDATE_COLUMNS,
        "removed_human_columns": sorted(HUMAN_COLUMNS),
        "note": (
            "reference_* contains Codex/Gemini labels for later comparison. "
            "candidate_llm_* is intentionally empty for an independent LLM."
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
