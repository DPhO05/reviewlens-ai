from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "EDA"
DATA_PATH = ROOT / "Data" / "gold_data" / "tiki_reviews_full_labeled_9k.csv"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.9"},
    }
    return nb


COMMON_SETUP = """
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

%matplotlib inline
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
pd.set_option("display.max_colwidth", 160)

DATA_PATH = Path("Data/gold_data/tiki_reviews_full_labeled_9k.csv")
if not DATA_PATH.exists():
    matches = list(Path(".").rglob("tiki_reviews_full_labeled_9k.csv"))
    if not matches:
        raise FileNotFoundError("Không tìm thấy tiki_reviews_full_labeled_9k.csv")
    DATA_PATH = matches[0]

print("Dữ liệu:", DATA_PATH.resolve())
"""


LOAD_AND_FEATURES = """
df = pd.read_csv(DATA_PATH)

available_columns = [
    "review_id", "product_id", "user_id", "rating", "title", "content",
    "created_at", "helpful_count", "purchased", "crawled_at", "help"
]
missing = [col for col in available_columns if col not in df.columns]
if missing:
    raise ValueError(f"Thiếu cột cần thiết: {missing}")

df = df[available_columns].copy()
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["content"] = df["content"].fillna("").astype(str).str.strip()
df["review_context"] = (
    df["title"] + ". " + df["content"]
).str.strip(". ")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["crawled_at"] = pd.to_datetime(df["crawled_at"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["helpful_count"] = pd.to_numeric(df["helpful_count"], errors="coerce").fillna(0)
df["help"] = pd.to_numeric(df["help"], errors="raise").astype(int)
df["purchased"] = df["purchased"].astype(bool)

# Feature engineering chỉ từ dữ liệu quan sát được.
df["review_char_length"] = df["review_context"].str.len()
df["review_word_length"] = df["review_context"].str.split().str.len()
df["title_char_length"] = df["title"].str.len()
df["content_char_length"] = df["content"].str.len()
df["sentence_count"] = df["review_context"].str.count(r"[.!?]+").clip(lower=1)
df["exclamation_count"] = df["review_context"].str.count("!")
df["question_count"] = df["review_context"].str.count(r"\\?")
df["digit_count"] = df["review_context"].str.count(r"\\d")
df["created_year"] = df["created_at"].dt.year
df["created_month"] = df["created_at"].dt.month
df["created_hour"] = df["created_at"].dt.hour
df["created_weekday"] = df["created_at"].dt.day_name()
df["has_helpful_vote"] = (df["helpful_count"] > 0).astype(int)

print(f"Kích thước sau khi tạo feature: {df.shape[0]:,} dòng x {df.shape[1]} cột")
display(df.head())
"""


def exploration_notebook() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
# KHAI PHÁ DỮ LIỆU REVIEW TIKI 9K ĐÃ GÁN NHÃN

Notebook được xây dựng theo cấu trúc của `Khai_pha_du_lieu.ipynb`, tập trung:

- Đọc và kiểm tra dữ liệu review đã gán nhãn.
- Chỉ sử dụng những trường thực tế có trong raw data.
- Tạo feature độ dài review và đặc trưng thời gian.
- Phân tích ảnh hưởng của rating, độ dài nội dung, helpful vote và thời gian đến nhãn `help`.
"""
            ),
            md("## Bước 1: Chuẩn bị môi trường và dữ liệu"),
            code(COMMON_SETUP),
            code(LOAD_AND_FEATURES),
            md("## Bước 2: Tổng quan cấu trúc dữ liệu"),
            code(
                """
print("THÔNG TIN TỔNG QUAN")
print("=" * 70)
print(f"Số review: {len(df):,}")
print(f"Số cột sau feature engineering: {df.shape[1]}")
print(f"Dung lượng bộ nhớ: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Khoảng thời gian review: {df['created_at'].min()} -> {df['created_at'].max()}")
print(f"Số sản phẩm: {df['product_id'].nunique():,}")
print(f"Số người dùng: {df['user_id'].nunique():,}")
df.info()
"""
            ),
            md("## Bước 3: Kiểm tra chất lượng dữ liệu"),
            code(
                """
quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_rate_pct": (df.isna().mean() * 100).round(2),
    "unique_count": df.nunique(dropna=True),
})
display(quality)

print("Duplicate review_id:", int(df["review_id"].duplicated().sum()))
print("Duplicate review_context:", int(df["review_context"].duplicated().sum()))
print("Review rỗng:", int(df["review_context"].eq("").sum()))
print("Nhãn ngoài 0/1:", int((~df["help"].isin([0, 1])).sum()))
"""
            ),
            md("## Bước 4: Thống kê mô tả các biến số"),
            code(
                """
numeric_columns = [
    "rating", "helpful_count", "help", "review_char_length",
    "review_word_length", "title_char_length", "content_char_length",
    "sentence_count", "exclamation_count", "question_count", "digit_count",
]
display(df[numeric_columns].describe(
    percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
).T)
"""
            ),
            md("## Bước 5: Phân phối nhãn và rating"),
            code(
                """
label_summary = (
    df["help"].value_counts().sort_index()
    .rename_axis("help").reset_index(name="count")
)
label_summary["percentage"] = (label_summary["count"] / len(df) * 100).round(2)
display(label_summary)

rating_summary = (
    df.groupby("rating")
    .agg(
        review_count=("review_id", "size"),
        helpful_count=("help", "sum"),
        helpful_rate=("help", "mean"),
        avg_word_length=("review_word_length", "mean"),
    )
    .reset_index()
)
rating_summary["helpful_rate_pct"] = (rating_summary["helpful_rate"] * 100).round(2)
display(rating_summary)
"""
            ),
            md("## Bước 6: Phân tích độ dài review theo nhãn"),
            code(
                """
length_summary = (
    df.groupby("help")
    .agg(
        review_count=("review_id", "size"),
        avg_char_length=("review_char_length", "mean"),
        median_char_length=("review_char_length", "median"),
        avg_word_length=("review_word_length", "mean"),
        median_word_length=("review_word_length", "median"),
        avg_sentence_count=("sentence_count", "mean"),
    )
    .round(2)
)
display(length_summary)

length_bins = pd.cut(
    df["review_word_length"],
    bins=[-1, 5, 10, 20, 40, 80, np.inf],
    labels=["1-5", "6-10", "11-20", "21-40", "41-80", ">80"],
)
length_effect = (
    df.assign(length_group=length_bins)
    .groupby("length_group", observed=False)
    .agg(
        review_count=("review_id", "size"),
        helpful_count=("help", "sum"),
        helpful_rate=("help", "mean"),
    )
    .reset_index()
)
length_effect["helpful_rate_pct"] = (length_effect["helpful_rate"] * 100).round(2)
display(length_effect)
"""
            ),
            md("## Bước 7: Helpful vote, purchased và nhãn help"),
            code(
                """
vote_summary = (
    df.groupby("has_helpful_vote")
    .agg(
        review_count=("review_id", "size"),
        avg_platform_helpful_count=("helpful_count", "mean"),
        labeled_helpful_count=("help", "sum"),
        labeled_helpful_rate=("help", "mean"),
    )
    .round(3)
)
display(vote_summary)

purchased_summary = (
    df.groupby("purchased")
    .agg(
        review_count=("review_id", "size"),
        helpful_rate=("help", "mean"),
        avg_word_length=("review_word_length", "mean"),
    )
    .round(3)
)
display(purchased_summary)
"""
            ),
            md("## Bước 8: Phân tích thời gian tạo review"),
            code(
                """
year_summary = (
    df.dropna(subset=["created_year"])
    .groupby("created_year")
    .agg(
        review_count=("review_id", "size"),
        helpful_rate=("help", "mean"),
        avg_word_length=("review_word_length", "mean"),
    )
    .reset_index()
)
year_summary["helpful_rate_pct"] = (year_summary["helpful_rate"] * 100).round(2)
display(year_summary)

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
weekday_summary = (
    df.groupby("created_weekday")
    .agg(review_count=("review_id", "size"), helpful_rate=("help", "mean"))
    .reindex(weekday_order)
)
weekday_summary["helpful_rate_pct"] = (weekday_summary["helpful_rate"] * 100).round(2)
display(weekday_summary)
"""
            ),
            md("## Bước 9: Tương quan giữa các biến số"),
            code(
                """
correlation_columns = [
    "rating", "helpful_count", "purchased", "review_char_length",
    "review_word_length", "sentence_count", "exclamation_count",
    "question_count", "digit_count", "created_year", "created_month",
    "created_hour", "help",
]
correlation_matrix = df[correlation_columns].astype(float).corr()
display(correlation_matrix.round(3))

target_corr = (
    correlation_matrix["help"]
    .drop("help")
    .sort_values(key=lambda series: series.abs(), ascending=False)
    .rename("correlation_with_help")
)
display(target_corr.to_frame())
"""
            ),
            md("## Bước 10: Quan sát các review đại diện"),
            code(
                """
for label in [0, 1]:
    print(f"\\n===== SAMPLE help={label} =====")
    sample = df[df["help"] == label].sample(5, random_state=42)
    display(sample[
        [
            "review_id", "rating", "review_context", "review_word_length",
            "helpful_count", "created_at", "help",
        ]
    ])
"""
            ),
            md(
                """
## Kết luận

- Nhãn `help` cần được phân tích cùng với **nội dung review**, không chỉ rating.
- Độ dài review là feature quan trọng để khảo sát, nhưng review dài không mặc định hữu ích.
- `helpful_count` của nền tảng là tín hiệu tham khảo, không phải nhãn thay thế.
- Các feature thời gian mô tả hành vi dữ liệu nhưng cần tránh diễn giải thành quan hệ nhân quả.
"""
            ),
        ]
    )


def visualization_notebook() -> nbf.NotebookNode:
    return notebook(
        [
            md(
                """
# TRỰC QUAN HÓA DỮ LIỆU REVIEW TIKI 9K ĐÃ GÁN NHÃN

Notebook được xây dựng theo cấu trúc của `Truc_quan_du_lieu.ipynb`.
Các biểu đồ tập trung vào quan hệ giữa nhãn `help` và:

- Rating.
- Độ dài review context.
- Helpful count của nền tảng.
- Trạng thái purchased.
- Thời gian tạo review.
"""
            ),
            md("## Bước 1: Chuẩn bị dữ liệu và feature"),
            code(COMMON_SETUP),
            code(LOAD_AND_FEATURES),
            code(
                """
FIGURE_DIR = Path("EDA/figures_9k_labeled")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def save_and_show(filename):
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", path)

print("Thư mục figure:", FIGURE_DIR.resolve())
"""
            ),
            md("## Bước 2: Phân phối nhãn và rating"),
            code(
                """
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

label_counts = df["help"].value_counts().sort_index()
sns.barplot(
    x=label_counts.index, y=label_counts.values,
    hue=label_counts.index, palette={0: "#e76f51", 1: "#2a9d8f"},
    legend=False, ax=axes[0],
)
axes[0].set_title("Phân phối nhãn Help")
axes[0].set_xlabel("help")
axes[0].set_ylabel("Số review")
for container in axes[0].containers:
    axes[0].bar_label(container)

sns.countplot(data=df, x="rating", hue="help", palette="Set2", ax=axes[1])
axes[1].set_title("Phân phối rating theo nhãn Help")
axes[1].set_xlabel("Rating")
axes[1].set_ylabel("Số review")
axes[1].legend(title="help")
save_and_show("01_label_and_rating_distribution.png")
"""
            ),
            md("## Bước 3: Tỷ lệ helpful theo rating"),
            code(
                """
rating_effect = (
    df.groupby("rating")
    .agg(review_count=("review_id", "size"), helpful_rate=("help", "mean"))
    .reset_index()
)
rating_effect["helpful_rate_pct"] = rating_effect["helpful_rate"] * 100

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=rating_effect, x="rating", y="helpful_rate_pct", color="#457b9d")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%")
plt.title("Tỷ lệ review Helpful theo Rating")
plt.xlabel("Rating")
plt.ylabel("Helpful rate (%)")
save_and_show("02_helpful_rate_by_rating.png")
display(rating_effect.round(3))
"""
            ),
            md("## Bước 4: Độ dài review theo nhãn"),
            code(
                """
upper_word = df["review_word_length"].quantile(0.99)
plot_df = df.assign(
    word_length_99p=df["review_word_length"].clip(upper=upper_word)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(
    data=plot_df, x="help", y="word_length_99p",
    hue="help", palette={0: "#e76f51", 1: "#2a9d8f"},
    legend=False, ax=axes[0],
)
axes[0].set_title("Boxplot số từ theo nhãn (clip P99)")
axes[0].set_xlabel("help")
axes[0].set_ylabel("Số từ")

sns.violinplot(
    data=plot_df, x="help", y="word_length_99p",
    hue="help", palette={0: "#e76f51", 1: "#2a9d8f"},
    inner="quartile", legend=False, cut=0, ax=axes[1],
)
axes[1].set_title("Violin plot số từ theo nhãn (clip P99)")
axes[1].set_xlabel("help")
axes[1].set_ylabel("Số từ")
save_and_show("03_review_length_by_label.png")
"""
            ),
            md("## Bước 5: Ảnh hưởng của nhóm độ dài review"),
            code(
                """
df["length_group"] = pd.cut(
    df["review_word_length"],
    bins=[-1, 5, 10, 20, 40, 80, np.inf],
    labels=["1-5", "6-10", "11-20", "21-40", "41-80", ">80"],
)
length_effect = (
    df.groupby("length_group", observed=False)
    .agg(review_count=("review_id", "size"), helpful_rate=("help", "mean"))
    .reset_index()
)
length_effect["helpful_rate_pct"] = length_effect["helpful_rate"] * 100

fig, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=length_effect, x="length_group", y="helpful_rate_pct",
    color="#8ecae6", ax=ax1,
)
ax1.set_title("Tỷ lệ Helpful theo nhóm độ dài review")
ax1.set_xlabel("Nhóm số từ")
ax1.set_ylabel("Helpful rate (%)")
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.1f%%")
save_and_show("04_helpful_rate_by_length_group.png")
display(length_effect.round(3))
"""
            ),
            md("## Bước 6: Rating, độ dài và nhãn Help"),
            code(
                """
sample = df.sample(min(3000, len(df)), random_state=42)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=sample,
    x="rating",
    y="review_word_length",
    hue="help",
    size="helpful_count",
    sizes=(25, 180),
    alpha=0.55,
    palette={0: "#e76f51", 1: "#2a9d8f"},
)
plt.ylim(0, df["review_word_length"].quantile(0.99))
plt.title("Rating và độ dài review theo nhãn Help")
plt.xlabel("Rating")
plt.ylabel("Số từ của review")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
save_and_show("05_rating_length_scatter.png")
"""
            ),
            md("## Bước 7: Helpful count của nền tảng và nhãn gán"),
            code(
                """
vote_plot = df.assign(
    platform_vote_group=pd.cut(
        df["helpful_count"],
        bins=[-1, 0, 1, 2, 5, 10, np.inf],
        labels=["0", "1", "2", "3-5", "6-10", ">10"],
    )
)
vote_effect = (
    vote_plot.groupby("platform_vote_group", observed=False)
    .agg(review_count=("review_id", "size"), helpful_rate=("help", "mean"))
    .reset_index()
)
vote_effect["helpful_rate_pct"] = vote_effect["helpful_rate"] * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=vote_plot, x="platform_vote_group", color="#ffb703", ax=axes[0])
axes[0].set_title("Phân phối Helpful Count của nền tảng")
axes[0].set_xlabel("Nhóm helpful_count")
axes[0].set_ylabel("Số review")

sns.barplot(
    data=vote_effect, x="platform_vote_group", y="helpful_rate_pct",
    color="#219ebc", ax=axes[1],
)
axes[1].set_title("Tỷ lệ nhãn Helpful theo Helpful Count")
axes[1].set_xlabel("Nhóm helpful_count")
axes[1].set_ylabel("Helpful rate (%)")
save_and_show("06_platform_helpful_count_analysis.png")
display(vote_effect.round(3))
"""
            ),
            md("## Bước 8: Purchased và nhãn Help"),
            code(
                """
purchased_effect = (
    df.groupby("purchased")
    .agg(
        review_count=("review_id", "size"),
        helpful_rate=("help", "mean"),
        avg_word_length=("review_word_length", "mean"),
    )
    .reset_index()
)
purchased_effect["helpful_rate_pct"] = purchased_effect["helpful_rate"] * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=df, x="purchased", hue="help", palette="Set2", ax=axes[0])
axes[0].set_title("Purchased theo nhãn Help")
axes[0].set_xlabel("Purchased")
axes[0].set_ylabel("Số review")

sns.barplot(
    data=purchased_effect, x="purchased", y="helpful_rate_pct",
    color="#90be6d", ax=axes[1],
)
axes[1].set_title("Tỷ lệ Helpful theo Purchased")
axes[1].set_xlabel("Purchased")
axes[1].set_ylabel("Helpful rate (%)")
save_and_show("07_purchased_analysis.png")
display(purchased_effect.round(3))
"""
            ),
            md("## Bước 9: Xu hướng theo thời gian"),
            code(
                """
time_effect = (
    df.dropna(subset=["created_year"])
    .groupby("created_year")
    .agg(
        review_count=("review_id", "size"),
        helpful_rate=("help", "mean"),
        avg_word_length=("review_word_length", "mean"),
    )
    .reset_index()
)
time_effect["helpful_rate_pct"] = time_effect["helpful_rate"] * 100

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
sns.lineplot(
    data=time_effect, x="created_year", y="review_count",
    marker="o", color="#264653", ax=axes[0],
)
axes[0].set_title("Số lượng review theo năm")
axes[0].set_ylabel("Số review")

sns.lineplot(
    data=time_effect, x="created_year", y="helpful_rate_pct",
    marker="o", color="#e76f51", ax=axes[1],
)
axes[1].set_title("Tỷ lệ Helpful theo năm")
axes[1].set_xlabel("Năm tạo review")
axes[1].set_ylabel("Helpful rate (%)")
save_and_show("08_time_trend.png")
display(time_effect.round(3))
"""
            ),
            md("## Bước 10: Heatmap tương quan"),
            code(
                """
corr_columns = [
    "rating", "helpful_count", "purchased", "review_char_length",
    "review_word_length", "title_char_length", "content_char_length",
    "sentence_count", "exclamation_count", "question_count",
    "digit_count", "created_year", "created_month", "created_hour", "help",
]
corr = df[corr_columns].astype(float).corr()

plt.figure(figsize=(13, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=False,
    cbar_kws={"label": "Hệ số tương quan Pearson"},
)
plt.title("Ma trận tương quan các feature và nhãn Help")
save_and_show("09_correlation_heatmap.png")
"""
            ),
            md("## Bước 11: FacetGrid độ dài review theo rating và nhãn"),
            code(
                """
facet_df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()
facet_df["word_length_99p"] = facet_df["review_word_length"].clip(
    upper=facet_df["review_word_length"].quantile(0.99)
)
g = sns.FacetGrid(
    facet_df,
    col="rating",
    hue="help",
    col_wrap=3,
    height=3.3,
    aspect=1.15,
    palette={0: "#e76f51", 1: "#2a9d8f"},
)
g.map(sns.histplot, "word_length_99p", bins=25, alpha=0.55)
g.add_legend(title="help")
g.fig.suptitle("Phân phối độ dài review theo Rating và Help", y=1.03)
g.savefig(FIGURE_DIR / "10_length_facet_by_rating.png", dpi=160, bbox_inches="tight")
plt.show()
print("Saved:", FIGURE_DIR / "10_length_facet_by_rating.png")
"""
            ),
            md("## Bước 12: Word Cloud của nội dung review theo nhãn Help"),
            code(
                """
# Word Cloud giúp quan sát nhanh các từ xuất hiện thường xuyên trong review.
# Tách riêng help=0 và help=1 để so sánh nội dung giữa hai nhóm.
try:
    from wordcloud import WordCloud
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
    from wordcloud import WordCloud

VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "cho", "được", "mình", "tôi", "thì", "mà",
    "rất", "quá", "này", "đó", "với", "khi", "đã", "cũng", "không", "ko",
    "k", "nhưng", "nên", "ở", "về", "một", "như", "cái", "các", "để",
    "sản", "phẩm", "sp", "shop", "tiki", "hàng", "ạ", "nha", "nhé",
}

def prepare_wordcloud_text(series):
    text = " ".join(series.fillna("").astype(str)).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

wordcloud_config = {
    "width": 1000,
    "height": 500,
    "background_color": "white",
    "max_words": 180,
    "collocations": False,
    "stopwords": VIETNAMESE_STOPWORDS,
    "random_state": 42,
}

text_not_helpful = prepare_wordcloud_text(df.loc[df["help"] == 0, "review_context"])
text_helpful = prepare_wordcloud_text(df.loc[df["help"] == 1, "review_context"])

wc_not_helpful = WordCloud(
    colormap="OrRd",
    **wordcloud_config,
).generate(text_not_helpful)

wc_helpful = WordCloud(
    colormap="YlGnBu",
    **wordcloud_config,
).generate(text_helpful)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
axes[0].imshow(wc_not_helpful, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title(
    "Word Cloud - Review Not Helpful (help=0)",
    fontsize=16,
    fontweight="bold",
)

axes[1].imshow(wc_helpful, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title(
    "Word Cloud - Review Helpful (help=1)",
    fontsize=16,
    fontweight="bold",
)

save_and_show("11_wordcloud_by_help_label.png")
"""
            ),
            md(
                """
## Kết luận trực quan

- Review được gán `help=1` có xu hướng dài và giàu thông tin hơn, nhưng độ dài không đủ để quyết định nhãn.
- Rating thấp có thể vẫn hữu ích nếu review mô tả lỗi hoặc trải nghiệm cụ thể.
- Helpful count của nền tảng có liên hệ với nhãn nhưng không hoàn toàn tương đương nhãn LLM.
- Các biểu đồ thời gian phản ánh phân bố mẫu thu thập, không chứng minh quan hệ nhân quả.
- Word Cloud chỉ phản ánh tần suất xuất hiện của từ, không đo trực tiếp mức độ quan trọng
  hoặc quan hệ nhân quả giữa từ và nhãn.
"""
            ),
        ]
    )


def execute_and_save(nb: nbf.NotebookNode, path: Path) -> None:
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    executed = client.execute()
    nbf.write(executed, path)
    print(f"Created and executed: {path}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/demo_ds_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/demo_ds_xdg")

    outputs = [
        (exploration_notebook(), EDA_DIR / "Khai_pha_du_lieu_9k_da_gan_nhan.ipynb"),
        (
            visualization_notebook(),
            EDA_DIR / "Truc_quan_du_lieu_9k_da_gan_nhan.ipynb",
        ),
    ]
    for nb, path in outputs:
        execute_and_save(nb, path)


if __name__ == "__main__":
    main()
