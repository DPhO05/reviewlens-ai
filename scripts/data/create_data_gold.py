from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data" / "gold_data" / "tiki_reviews_gold_llm_helpfulness_v1.csv"
OUTPUT = ROOT / "Data" / "gold_data" / "data_gold.csv"

HUMAN_COLUMNS = {
    "annotator_id",
    "manual_label",
    "label_confidence",
    "label_notes",
    "original_manual_label",
    "agrees_with_original",
}


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    source = pd.read_csv(SOURCE, dtype={"annotation_id": str})
    output = source.drop(
        columns=[column for column in HUMAN_COLUMNS if column in source.columns]
    )
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    assert len(output) == len(source)
    assert output["annotation_id"].nunique() == len(output)
    assert not HUMAN_COLUMNS.intersection(output.columns)
    assert output["is_helpful"].notna().all()
    assert output["label_source"].isin(
        ["llm", "codex_rule_adjudicated"]
    ).all()

    print(f"Created: {OUTPUT}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")
    print(f"Label sources: {output['label_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
