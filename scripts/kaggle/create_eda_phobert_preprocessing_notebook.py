from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "eda_phobert_preprocessing_9k.ipynb"


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
# EDA và chuẩn hoá dữ liệu cho PhoBERT

Notebook này xử lý file `tiki_reviews_helpfulness_9k.csv` với 2 mục tiêu:

1. **EDA dữ liệu**: kiểm tra schema, missing values, phân phối nhãn `help`,
   rating, độ dài review và các quan hệ cơ bản.
2. **Chuẩn hoá đầu vào PhoBERT**: làm sạch text ở mức nhẹ, giữ nguyên dấu tiếng
   Việt và ngữ nghĩa review, tạo file `tiki_reviews_helpfulness_9k_phobert_ready.csv`.

Nguyên tắc chuẩn hoá cho PhoBERT:

- Chuẩn hoá Unicode về NFC.
- Xoá ký tự điều khiển, HTML entities và khoảng trắng dư.
- Không bỏ dấu tiếng Việt.
- Không lowercase bắt buộc.
- Không xoá stopword.
- Không stemming/lemmatization.
- Không viết lại nội dung review.
"""
            ),
            md("## 1. Import và cấu hình"),
            code(
                """
import html
import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

pd.set_option("display.max_colwidth", 180)
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42

OUTPUT_DIR = (
    Path("/kaggle/working/eda_phobert_outputs")
    if Path("/kaggle/working").exists()
    else Path("eda_phobert_outputs")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PHOBERT_READY_PATH = OUTPUT_DIR / "tiki_reviews_helpfulness_9k_phobert_ready.csv"
EDA_SUMMARY_PATH = OUTPUT_DIR / "eda_summary.json"

print("OUTPUT_DIR:", OUTPUT_DIR)
print("FIGURE_DIR:", FIGURE_DIR)


def save_and_show(filename: str) -> None:
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved figure:", path)
"""
            ),
            md("## 2. Load dữ liệu"),
            code(
                """
def find_dataset() -> Path:
    candidate_names = ["tiki_reviews_helpfulness_9k.csv"]
    roots = [Path("/kaggle/input"), Path("."), Path("Data/gold_data")]
    for root in roots:
        if not root.exists():
            continue
        for name in candidate_names:
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Không tìm thấy tiki_reviews_helpfulness_9k.csv. "
        "Hãy upload file này lên Kaggle Dataset hoặc đặt trong Data/gold_data."
    )


DATA_PATH = find_dataset()
df = pd.read_csv(DATA_PATH)

print("DATA_PATH:", DATA_PATH)
print("Shape:", df.shape)
display(df.head())
"""
            ),
            md("## 3. Kiểm tra schema và kiểu dữ liệu"),
            code(
                """
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

missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]

print("Missing columns:", missing)
print("Extra columns:", extra)
display(df.dtypes.rename("dtype").to_frame())

if missing:
    raise ValueError(f"Thiếu cột bắt buộc: {missing}")

df = df[EXPECTED_COLUMNS].copy()
df["help"] = pd.to_numeric(df["help"], errors="raise").astype(int)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["review_text"] = df["review_text"].fillna("").astype(str)
"""
            ),
            md("## 4. Missing values và duplicate"),
            code(
                """
missing_report = pd.DataFrame(
    {
        "missing_count": df.isna().sum(),
        "missing_rate_pct": (df.isna().mean() * 100).round(2),
    }
).sort_values("missing_count", ascending=False)

duplicate_review_id = int(df["review_id"].duplicated().sum())
duplicate_full_text = int(df["review_text"].duplicated().sum())

print("Duplicate review_id:", duplicate_review_id)
print("Duplicate review_text:", duplicate_full_text)
display(missing_report)
"""
            ),
            md("## 5. Phân phối nhãn `help`"),
            code(
                """
label_counts = (
    df["help"]
    .value_counts()
    .sort_index()
    .rename_axis("help")
    .reset_index(name="count")
)
label_counts["percentage"] = (label_counts["count"] / len(df) * 100).round(2)
display(label_counts)

plt.figure(figsize=(6, 4))
ax = sns.barplot(data=label_counts, x="help", y="count", palette="Set2")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Phân phối nhãn help")
plt.xlabel("help")
plt.ylabel("Số review")
save_and_show("01_label_distribution.png")
"""
            ),
            md("## 6. Phân phối rating và quan hệ với nhãn"),
            code(
                """
rating_counts = (
    df["rating"]
    .value_counts(dropna=False)
    .sort_index()
    .rename_axis("rating")
    .reset_index(name="count")
)
display(rating_counts)

rating_help = (
    df.groupby("rating", dropna=False)
    .agg(
        review_count=("help", "size"),
        helpful_count=("help", "sum"),
        helpful_rate=("help", "mean"),
    )
    .reset_index()
)
rating_help["helpful_rate"] = (rating_help["helpful_rate"] * 100).round(2)
display(rating_help)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x="rating", ax=axes[0], color="#6baed6")
axes[0].set_title("Phân phối rating")
axes[0].set_xlabel("rating")
axes[0].set_ylabel("Số review")

sns.barplot(data=rating_help, x="rating", y="helpful_rate", ax=axes[1], color="#74c476")
axes[1].set_title("Tỷ lệ helpful theo rating")
axes[1].set_xlabel("rating")
axes[1].set_ylabel("Helpful rate (%)")
save_and_show("02_rating_analysis.png")
"""
            ),
            md("## 7. Độ dài review"),
            code(
                """
df["char_len"] = df["review_text"].str.len()
df["word_count_computed"] = df["review_text"].str.split().str.len()

display(df[["char_len", "word_count_computed"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df["word_count_computed"], bins=60, ax=axes[0])
axes[0].set_title("Phân phối số từ")
axes[0].set_xlabel("word_count")

sns.boxplot(data=df, x="help", y="word_count_computed", ax=axes[1])
axes[1].set_title("Số từ theo nhãn help")
axes[1].set_xlabel("help")
axes[1].set_ylabel("word_count")
save_and_show("03_length_distribution.png")
"""
            ),
            md("## 8. Missing metadata trong file hiện tại"),
            code(
                """
metadata_cols = ["product_name", "category", "helpful_count", "created_at"]
metadata_missing = pd.DataFrame(
    {
        "missing_count": [df[col].fillna("").astype(str).str.strip().eq("").sum() for col in metadata_cols],
        "non_empty_count": [df[col].fillna("").astype(str).str.strip().ne("").sum() for col in metadata_cols],
    },
    index=metadata_cols,
)
metadata_missing["missing_rate_pct"] = (metadata_missing["missing_count"] / len(df) * 100).round(2)
display(metadata_missing)

plt.figure(figsize=(8, 4))
ax = sns.barplot(
    data=metadata_missing.reset_index(names="column"),
    x="column",
    y="missing_rate_pct",
    color="#fb8072",
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%")
plt.title("Tỷ lệ thiếu/trống của metadata")
plt.xlabel("Cột")
plt.ylabel("Missing rate (%)")
plt.xticks(rotation=20, ha="right")
save_and_show("04_metadata_missing_rate.png")

print(
    "Lưu ý: nếu các cột metadata này trống 100%, không nên dùng chúng làm feature train "
    "trừ khi bổ sung lại từ nguồn crawl ban đầu."
)
"""
            ),
            md("## 9. Sample review theo nhãn"),
            code(
                """
for label in [0, 1]:
    print(f"\\n===== help={label} =====")
    sample = df[df["help"] == label].sample(
        min(5, (df["help"] == label).sum()),
        random_state=RANDOM_STATE,
    )
    display(sample[["review_id", "product_id", "rating", "review_text", "help"]])
"""
            ),
            md("## 10. Chuẩn hoá text nhẹ cho PhoBERT"),
            code(
                r"""
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_phobert(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


processed = df.copy()
processed["phobert_text"] = processed["review_text"].map(normalize_for_phobert)
processed["phobert_char_len"] = processed["phobert_text"].str.len()
processed["phobert_word_count"] = processed["phobert_text"].str.split().str.len()

empty_after_clean = int(processed["phobert_text"].eq("").sum())
changed_text = int((processed["phobert_text"] != processed["review_text"]).sum())

print("Empty after clean:", empty_after_clean)
print("Rows changed by normalization:", changed_text)
display(processed[["review_text", "phobert_text"]].head())
"""
            ),
            md(
                """
### Vì sao không lowercase/bỏ dấu/xoá stopword?

PhoBERT có tokenizer và phân phối pretraining riêng cho tiếng Việt. Các thao tác
như bỏ dấu, xoá stopword, stemming hoặc viết lại câu có thể làm mất tín hiệu ngữ
nghĩa trong review. Do đó notebook chỉ chuẩn hoá kỹ thuật ở mức tối thiểu.
"""
            ),
            md("## 11. Kiểm tra độ dài token bằng PhoBERT tokenizer (tuỳ chọn)"),
            code(
                """
# Cell này cần Internet lần đầu để tải tokenizer từ Hugging Face.
# Nếu chỉ cần file CSV đã chuẩn hoá, có thể bỏ qua cell này.

RUN_TOKENIZER_CHECK = False

if RUN_TOKENIZER_CHECK:
    !pip install -q transformers sentencepiece

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

    token_lengths = []
    for text in processed["phobert_text"].tolist():
        token_lengths.append(
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
        )

    processed["phobert_token_len"] = token_lengths
    display(processed["phobert_token_len"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    plt.figure(figsize=(8, 4))
    sns.histplot(processed["phobert_token_len"], bins=60)
    plt.axvline(192, color="red", linestyle="--", label="max_length=192")
    plt.title("Phân phối độ dài token PhoBERT")
    plt.xlabel("Số token")
    plt.legend()
    save_and_show("07_phobert_token_length.png")

    over_192 = (processed["phobert_token_len"] > 192).mean() * 100
    print(f"Tỷ lệ review dài hơn 192 token: {over_192:.2f}%")
else:
    print("Tokenizer check đang tắt. Đổi RUN_TOKENIZER_CHECK = True nếu cần kiểm tra token length.")
"""
            ),
            md("## 12. Visualize bổ sung cho báo cáo"),
            code(
                """
# Violin plot độ dài review theo nhãn, clip ở percentile 99 để tránh outlier làm méo hình.
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
plt.title("Phân phối số từ theo nhãn help (clip 99th percentile)")
plt.xlabel("help")
plt.ylabel("PhoBERT-ready word count")
save_and_show("05_word_count_violin_99p.png")

# Heatmap tỷ trọng nhãn trong từng mức rating.
rating_label_share = pd.crosstab(
    processed["rating"],
    processed["help"],
    normalize="index",
).mul(100)

plt.figure(figsize=(7, 4))
sns.heatmap(
    rating_label_share,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    cbar_kws={"label": "Percentage within rating"},
)
plt.title("Tỷ trọng nhãn help trong từng rating")
plt.xlabel("help")
plt.ylabel("rating")
save_and_show("06_rating_label_heatmap.png")
"""
            ),
            md("## 13. Hiển thị toàn bộ figure đã lưu"),
            code(
                """
figure_files = sorted(FIGURE_DIR.glob("*.png"))
print("Saved figures:", len(figure_files))
for path in figure_files:
    print(path.name)
    display(Image(filename=str(path)))
"""
            ),
            md("## 14. Xuất file PhoBERT-ready"),
            code(
                """
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
if "phobert_token_len" in processed.columns:
    export_columns.append("phobert_token_len")

processed[export_columns].to_csv(PHOBERT_READY_PATH, index=False, encoding="utf-8-sig")

summary = {
    "source": str(DATA_PATH),
    "rows": int(len(processed)),
    "label_distribution": {
        str(k): int(v)
        for k, v in processed["help"].value_counts().sort_index().items()
    },
    "empty_after_clean": empty_after_clean,
    "rows_changed_by_normalization": changed_text,
    "output": str(PHOBERT_READY_PATH),
    "figure_dir": str(FIGURE_DIR),
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
}
EDA_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("Saved:", PHOBERT_READY_PATH)
print("Saved:", EDA_SUMMARY_PATH)
display(processed[export_columns].head())
"""
            ),
            md("## 15. Kết luận EDA ngắn cho báo cáo"),
            code(
                """
help_dist = processed["help"].value_counts(normalize=True).sort_index() * 100
avg_words_by_label = processed.groupby("help")["phobert_word_count"].mean().round(2)
rating_help_rate = (processed.groupby("rating")["help"].mean() * 100).round(2)

print("Gợi ý mô tả báo cáo:")
print(f"- Bộ dữ liệu gồm {len(processed)} review với nhãn nhị phân help.")
print(
    f"- Tỷ lệ Not Helpful/Helpful lần lượt là "
    f"{help_dist.get(0, 0):.2f}% và {help_dist.get(1, 0):.2f}%."
)
print("- Số từ trung bình theo nhãn:")
print(avg_words_by_label.to_string())
print("- Tỷ lệ helpful theo rating:")
print(rating_help_rate.to_string())
print(
    "- Text được chuẩn hoá nhẹ trước khi embedding bằng PhoBERT: chuẩn hoá Unicode, "
    "loại khoảng trắng/ký tự điều khiển, giữ nguyên dấu và nội dung gốc."
)
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
