from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Data" / "gold_data" / "tiki_reviews_helpfulness_9k.csv"
OUTPUT_DIR = ROOT / "Data" / "gold_data" / "eda_phobert"
PHOBERT_READY_PATH = OUTPUT_DIR / "tiki_reviews_helpfulness_9k_phobert_ready.csv"
EDA_SUMMARY_PATH = OUTPUT_DIR / "eda_summary.json"
FIGURE_DIR = OUTPUT_DIR / "figures"

EXPECTED_COLUMNS = [
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

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_phobert(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")


def save_current_figure(name: str) -> None:
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    data = pd.read_csv(DATA_PATH)
    missing = [column for column in EXPECTED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[EXPECTED_COLUMNS].copy()
    data["help"] = pd.to_numeric(data["help"], errors="raise").astype(int)
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data["review_text"] = data["review_text"].fillna("").astype(str)

    missing_report = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": data.isna().sum().values,
            "missing_rate_pct": (data.isna().mean().values * 100).round(2),
        }
    ).sort_values("missing_count", ascending=False)
    write_csv(missing_report, "missing_report.csv")

    label_distribution = (
        data["help"]
        .value_counts()
        .sort_index()
        .rename_axis("help")
        .reset_index(name="count")
    )
    label_distribution["percentage"] = (
        label_distribution["count"] / len(data) * 100
    ).round(2)
    write_csv(label_distribution, "label_distribution.csv")

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=label_distribution,
        x="help",
        y="count",
        hue="help",
        palette={0: "#fc8d62", 1: "#66c2a5"},
        legend=False,
    )
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Label Distribution")
    ax.set_xlabel("help")
    ax.set_ylabel("Review count")
    save_current_figure("01_label_distribution.png")

    rating_analysis = (
        data.groupby("rating", dropna=False)
        .agg(
            review_count=("help", "size"),
            helpful_count=("help", "sum"),
            helpful_rate=("help", "mean"),
        )
        .reset_index()
    )
    rating_analysis["helpful_rate_pct"] = (
        rating_analysis["helpful_rate"] * 100
    ).round(2)
    write_csv(rating_analysis, "rating_analysis.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=data, x="rating", color="#80b1d3", ax=axes[0])
    axes[0].set_title("Rating Distribution")
    axes[0].set_xlabel("rating")
    axes[0].set_ylabel("Review count")

    sns.barplot(
        data=rating_analysis,
        x="rating",
        y="helpful_rate_pct",
        color="#8dd3c7",
        ax=axes[1],
    )
    axes[1].set_title("Helpful Rate by Rating")
    axes[1].set_xlabel("rating")
    axes[1].set_ylabel("Helpful rate (%)")
    save_current_figure("02_rating_analysis.png")

    processed = data.copy()
    processed["char_len"] = processed["review_text"].str.len()
    processed["word_count_computed"] = processed["review_text"].str.split().str.len()
    processed["phobert_text"] = processed["review_text"].map(normalize_for_phobert)
    processed["phobert_char_len"] = processed["phobert_text"].str.len()
    processed["phobert_word_count"] = processed["phobert_text"].str.split().str.len()

    length_by_label = (
        processed.groupby("help")
        .agg(
            review_count=("help", "size"),
            avg_word_count=("phobert_word_count", "mean"),
            median_word_count=("phobert_word_count", "median"),
            avg_char_len=("phobert_char_len", "mean"),
            median_char_len=("phobert_char_len", "median"),
        )
        .round(2)
        .reset_index()
    )
    write_csv(length_by_label, "length_by_label.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(processed["phobert_word_count"], bins=60, color="#bebada", ax=axes[0])
    axes[0].set_title("Review Word Count Distribution")
    axes[0].set_xlabel("PhoBERT-ready word count")
    axes[0].set_ylabel("Review count")

    sns.boxplot(
        data=processed,
        x="help",
        y="phobert_word_count",
        hue="help",
        palette={0: "#fc8d62", 1: "#66c2a5"},
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Word Count by Label")
    axes[1].set_xlabel("help")
    axes[1].set_ylabel("PhoBERT-ready word count")
    save_current_figure("03_length_distribution.png")

    clipped = processed.copy()
    upper = clipped["phobert_word_count"].quantile(0.99)
    clipped["word_count_clipped_99p"] = clipped["phobert_word_count"].clip(upper=upper)
    plt.figure(figsize=(7, 4))
    sns.violinplot(
        data=clipped,
        x="help",
        y="word_count_clipped_99p",
        hue="help",
        palette={0: "#fc8d62", 1: "#66c2a5"},
        legend=False,
        cut=0,
    )
    plt.title("Word Count by Label (clipped at 99th percentile)")
    plt.xlabel("help")
    plt.ylabel("PhoBERT-ready word count")
    save_current_figure("04_word_count_violin_99p.png")

    metadata_cols = ["product_name", "category", "helpful_count", "created_at"]
    metadata_missing = pd.DataFrame(
        {
            "column": metadata_cols,
            "empty_or_missing_count": [
                int(processed[col].fillna("").astype(str).str.strip().eq("").sum())
                for col in metadata_cols
            ],
            "non_empty_count": [
                int(processed[col].fillna("").astype(str).str.strip().ne("").sum())
                for col in metadata_cols
            ],
        }
    )
    metadata_missing["empty_or_missing_rate_pct"] = (
        metadata_missing["empty_or_missing_count"] / len(processed) * 100
    ).round(2)
    write_csv(metadata_missing, "metadata_missing.csv")

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(
        data=metadata_missing,
        x="column",
        y="empty_or_missing_rate_pct",
        color="#fb8072",
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    ax.set_title("Missing Rate of Metadata Columns")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing or empty rate (%)")
    plt.xticks(rotation=20, ha="right")
    save_current_figure("05_metadata_missing_rate.png")

    cross_tab = pd.crosstab(
        processed["rating"],
        processed["help"],
        normalize="index",
    ).mul(100)
    plt.figure(figsize=(7, 4))
    sns.heatmap(
        cross_tab,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Percentage within rating"},
    )
    plt.title("Label Share within Each Rating")
    plt.xlabel("help")
    plt.ylabel("rating")
    save_current_figure("06_rating_label_heatmap.png")

    export_columns = [
        "review_id",
        "product_id",
        "product_name",
        "category",
        "rating",
        "review_text",
        "phobert_text",
        "helpful_count",
        "created_at",
        "help",
        "char_len",
        "word_count_computed",
        "phobert_char_len",
        "phobert_word_count",
    ]
    processed[export_columns].to_csv(
        PHOBERT_READY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "source": str(DATA_PATH.relative_to(ROOT)),
        "rows": int(len(processed)),
        "columns": EXPECTED_COLUMNS,
        "label_distribution": {
            str(key): int(value)
            for key, value in processed["help"].value_counts().sort_index().items()
        },
        "helpful_rate_pct": round(float(processed["help"].mean() * 100), 2),
        "duplicate_review_id": int(processed["review_id"].duplicated().sum()),
        "duplicate_review_text": int(processed["review_text"].duplicated().sum()),
        "empty_after_clean": int(processed["phobert_text"].eq("").sum()),
        "rows_changed_by_normalization": int(
            (processed["phobert_text"] != processed["review_text"]).sum()
        ),
        "avg_phobert_word_count_by_label": {
            str(key): round(float(value), 2)
            for key, value in processed.groupby("help")["phobert_word_count"]
            .mean()
            .items()
        },
        "metadata_empty_or_missing": {
            row["column"]: int(row["empty_or_missing_count"])
            for _, row in metadata_missing.iterrows()
        },
        "normalization": [
            "html.unescape",
            "Unicode NFC",
            "remove control characters",
            "normalize whitespace",
            "keep Vietnamese accents",
            "no stopword removal",
            "no stemming",
            "no lowercasing",
        ],
        "outputs": {
            "phobert_ready_csv": str(PHOBERT_READY_PATH.relative_to(ROOT)),
            "missing_report": str((OUTPUT_DIR / "missing_report.csv").relative_to(ROOT)),
            "label_distribution": str(
                (OUTPUT_DIR / "label_distribution.csv").relative_to(ROOT)
            ),
            "rating_analysis": str((OUTPUT_DIR / "rating_analysis.csv").relative_to(ROOT)),
            "length_by_label": str((OUTPUT_DIR / "length_by_label.csv").relative_to(ROOT)),
            "metadata_missing": str((OUTPUT_DIR / "metadata_missing.csv").relative_to(ROOT)),
            "figures": str(FIGURE_DIR.relative_to(ROOT)),
        },
    }
    EDA_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(processed)}")
    print(f"Help distribution: {summary['label_distribution']}")
    print(f"Saved: {PHOBERT_READY_PATH}")
    print(f"Saved: {EDA_SUMMARY_PATH}")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
