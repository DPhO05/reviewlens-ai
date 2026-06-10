from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Data" / "gold_data" / "data_gold.csv"
EDA_DIR = ROOT / "Data" / "gold_data" / "eda"
CLEAN_PATH = ROOT / "Data" / "gold_data" / "data_gold_eda.csv"

DROP_COLUMNS = ["adjudicated", "needs_human_review"]


def percentage(series: pd.Series) -> pd.Series:
    return (series * 100).round(2)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(EDA_DIR / name, index=False, encoding="utf-8-sig")


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, dtype={"annotation_id": str})
    clean = data.drop(columns=DROP_COLUMNS)
    clean_column_count = len(clean.columns)
    clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

    label = (
        clean.groupby("is_helpful")
        .size()
        .rename("count")
        .reset_index()
        .assign(
            label=lambda x: x["is_helpful"].map({0: "Not Helpful", 1: "Helpful"}),
            percentage=lambda x: percentage(x["count"] / len(clean)),
        )[["is_helpful", "label", "count", "percentage"]]
    )
    save_csv(label, "label_distribution.csv")

    rating = (
        clean.groupby("rating")
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_score=("helpfulness_score", "mean"),
            avg_confidence=("confidence", "mean"),
            avg_word_count=("word_count", "mean"),
        )
        .reset_index()
    )
    rating["helpful_rate"] = percentage(rating["helpful_rate"])
    rating[["avg_score", "avg_confidence", "avg_word_count"]] = rating[
        ["avg_score", "avg_confidence", "avg_word_count"]
    ].round(2)
    save_csv(rating, "rating_analysis.csv")

    bins = [-1, 5, 10, 20, 50, np.inf]
    labels = ["1-5", "6-10", "11-20", "21-50", "51+"]
    clean["word_count_bucket"] = pd.cut(
        clean["word_count"], bins=bins, labels=labels, ordered=True
    )
    length = (
        clean.groupby("word_count_bucket", observed=False)
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_score=("helpfulness_score", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    length["helpful_rate"] = percentage(length["helpful_rate"])
    length[["avg_score", "avg_confidence"]] = length[
        ["avg_score", "avg_confidence"]
    ].round(2)
    save_csv(length, "length_analysis.csv")

    sample_group = (
        clean.groupby("sample_group")
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_score=("helpfulness_score", "mean"),
            avg_confidence=("confidence", "mean"),
            avg_word_count=("word_count", "mean"),
        )
        .reset_index()
        .sort_values("review_count", ascending=False)
    )
    sample_group["helpful_rate"] = percentage(sample_group["helpful_rate"])
    sample_group[["avg_score", "avg_confidence", "avg_word_count"]] = sample_group[
        ["avg_score", "avg_confidence", "avg_word_count"]
    ].round(2)
    save_csv(sample_group, "sample_group_analysis.csv")

    score = (
        clean.groupby("helpfulness_score")
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    score["percentage"] = percentage(score["review_count"] / len(clean))
    score["helpful_rate"] = percentage(score["helpful_rate"])
    score["avg_confidence"] = score["avg_confidence"].round(3)
    save_csv(score, "score_distribution.csv")

    confidence_bins = [0, 0.7, 0.8, 0.9, 1.000001]
    confidence_labels = ["<0.70", "0.70-0.79", "0.80-0.89", "0.90-1.00"]
    clean["confidence_bucket"] = pd.cut(
        clean["confidence"],
        bins=confidence_bins,
        labels=confidence_labels,
        right=False,
        include_lowest=True,
    )
    confidence = (
        clean.groupby("confidence_bucket", observed=False)
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_score=("helpfulness_score", "mean"),
        )
        .reset_index()
    )
    confidence["percentage"] = percentage(confidence["review_count"] / len(clean))
    confidence["helpful_rate"] = percentage(confidence["helpful_rate"])
    confidence["avg_score"] = confidence["avg_score"].round(2)
    save_csv(confidence, "confidence_analysis.csv")

    source = (
        clean.groupby("label_source")
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_score=("helpfulness_score", "mean"),
            avg_confidence=("confidence", "mean"),
            avg_word_count=("word_count", "mean"),
        )
        .reset_index()
    )
    source["percentage"] = percentage(source["review_count"] / len(clean))
    source["helpful_rate"] = percentage(source["helpful_rate"])
    source[["avg_score", "avg_confidence", "avg_word_count"]] = source[
        ["avg_score", "avg_confidence", "avg_word_count"]
    ].round(2)
    save_csv(source, "source_analysis.csv")

    purchased = (
        clean.groupby("purchased_int")
        .agg(
            review_count=("review_id", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
            avg_score=("helpfulness_score", "mean"),
            avg_word_count=("word_count", "mean"),
        )
        .reset_index()
    )
    purchased["purchase_status"] = purchased["purchased_int"].map(
        {0: "Không xác minh", 1: "Đã mua"}
    )
    purchased["helpful_rate"] = percentage(purchased["helpful_rate"])
    purchased[["avg_score", "avg_word_count"]] = purchased[
        ["avg_score", "avg_word_count"]
    ].round(2)
    save_csv(purchased, "purchase_analysis.csv")

    rubric_columns = [
        "specificity",
        "product_experience",
        "decision_value",
        "clarity",
        "noise_penalty",
    ]
    rubric = (
        clean.groupby("is_helpful")[rubric_columns]
        .mean()
        .round(3)
        .reset_index()
        .melt(
            id_vars="is_helpful",
            var_name="criterion",
            value_name="average_score",
        )
    )
    rubric["label"] = rubric["is_helpful"].map(
        {0: "Not Helpful", 1: "Helpful"}
    )
    save_csv(rubric, "rubric_by_label.csv")

    flag_rows = []
    for _, row in clean[["is_helpful", "quality_flags"]].iterrows():
        try:
            flags = json.loads(row["quality_flags"])
        except (TypeError, json.JSONDecodeError):
            flags = []
        if not flags:
            flag_rows.append({"flag": "no_flag", "is_helpful": row["is_helpful"]})
        else:
            for flag in flags:
                flag_rows.append({"flag": flag, "is_helpful": row["is_helpful"]})
    flags = pd.DataFrame(flag_rows)
    flag_summary = (
        flags.groupby("flag")
        .agg(
            occurrence_count=("is_helpful", "size"),
            helpful_count=("is_helpful", "sum"),
            helpful_rate=("is_helpful", "mean"),
        )
        .reset_index()
        .sort_values("occurrence_count", ascending=False)
    )
    flag_summary["helpful_rate"] = percentage(flag_summary["helpful_rate"])
    save_csv(flag_summary, "quality_flags_analysis.csv")

    numeric_columns = [
        "rating",
        "purchased_int",
        "word_count",
        "is_helpful",
        "helpfulness_score",
        "confidence",
        *rubric_columns,
    ]
    correlation = clean[numeric_columns].corr().round(3)
    correlation.insert(0, "variable", correlation.index)
    save_csv(correlation.reset_index(drop=True), "correlation_matrix.csv")

    label_stats = (
        clean.groupby("is_helpful")
        .agg(
            review_count=("review_id", "size"),
            avg_rating=("rating", "mean"),
            avg_word_count=("word_count", "mean"),
            median_word_count=("word_count", "median"),
            avg_score=("helpfulness_score", "mean"),
            avg_confidence=("confidence", "mean"),
            verified_purchase_rate=("purchased_int", "mean"),
        )
        .round(3)
        .reset_index()
    )
    label_stats["label"] = label_stats["is_helpful"].map(
        {0: "Not Helpful", 1: "Helpful"}
    )
    label_stats["verified_purchase_rate"] = percentage(
        label_stats["verified_purchase_rate"]
    )
    save_csv(label_stats, "label_profile.csv")

    summary = {
        "rows": int(len(clean)),
        "columns": int(clean_column_count),
        "duplicate_annotation_ids": int(clean["annotation_id"].duplicated().sum()),
        "duplicate_review_ids": int(clean["review_id"].duplicated().sum()),
        "missing_content": int(clean["content"].isna().sum()),
        "helpful_count": int(clean["is_helpful"].sum()),
        "not_helpful_count": int((clean["is_helpful"] == 0).sum()),
        "helpful_rate": round(float(clean["is_helpful"].mean() * 100), 2),
        "average_rating": round(float(clean["rating"].mean()), 2),
        "average_word_count": round(float(clean["word_count"].mean()), 2),
        "median_word_count": round(float(clean["word_count"].median()), 2),
        "average_score": round(float(clean["helpfulness_score"].mean()), 2),
        "average_confidence": round(float(clean["confidence"].mean()), 3),
        "verified_purchase_rate": round(float(clean["purchased_int"].mean() * 100), 2),
        "removed_columns": DROP_COLUMNS,
    }
    (EDA_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Clean data: {CLEAN_PATH}")
    print(f"EDA directory: {EDA_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
