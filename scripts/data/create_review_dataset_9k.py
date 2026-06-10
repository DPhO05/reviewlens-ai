from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data" / "gold_data" / "data_gold.csv"
OUTPUT = ROOT / "Data" / "gold_data" / "tiki_reviews_helpfulness_9k.csv"

OUTPUT_COLUMNS = [
    "review_id",
    "product_id",
    "product_name",
    "category",
    "rating",
    "review_text",
    "helpful_count",
    "created_at",
    "help",
]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    data = pd.read_csv(SOURCE)
    required = {"review_id", "product_id", "rating", "review_text", "is_helpful"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    output = pd.DataFrame(
        {
            "review_id": data["review_id"],
            "product_id": data["product_id"],
            "product_name": data["product_name"] if "product_name" in data else "",
            "category": data["category"] if "category" in data else "",
            "rating": data["rating"],
            "review_text": data["review_text"],
            "helpful_count": (
                data["helpful_count"] if "helpful_count" in data else ""
            ),
            "created_at": data["created_at"] if "created_at" in data else "",
            "help": pd.to_numeric(data["is_helpful"], errors="raise").astype(int),
        }
    )
    output = output[OUTPUT_COLUMNS]
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "rows": int(len(output)),
        "columns": OUTPUT_COLUMNS,
        "target_column": "help",
        "target_source_column": "is_helpful",
        "empty_columns_due_to_missing_source": [
            column
            for column in ["product_name", "category", "helpful_count", "created_at"]
            if column not in data.columns
        ],
    }
    OUTPUT.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")
    print(f"Help distribution: {output['help'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
