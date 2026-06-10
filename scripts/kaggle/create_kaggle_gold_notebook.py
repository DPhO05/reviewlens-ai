from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "notebook_data_gold_phobert_berf.ipynb"


def md(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str):
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
        "cells": [],
        "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "GPU"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    cells = [
        md(
            """
# PhoBERT + Leakage-Safe BERF trên `data_gold.csv`

Notebook này kế thừa cấu trúc của `notebook22f71cd157.ipynb`, nhưng dùng tập
gold mới và sửa hai vấn đề quan trọng:

1. **Stacking leakage:** probability của Random Forest dùng cho final model
   được tạo bằng inner out-of-fold, không dùng prediction in-sample.
2. **Annotation leakage:** `helpfulness_score` và `confidence` có tương quan
   mạnh với nhãn và không tồn tại tự nhiên khi inference. Notebook không đưa
   giá trị thật trực tiếp vào model. Thay vào đó:
   - `confidence` và độ xa biên của `helpfulness_score` được dùng làm
     **sample weight** trong lúc train.
   - Hai mô hình phụ dự đoán `helpfulness_score` và `confidence` bằng nested
     OOF. Chỉ **giá trị dự đoán** được ghép vào final BERF feature.

Pipeline triển khai:

```text
review -> PhoBERT embedding
       -> RF OOF class probabilities
       -> predicted helpfulness score + predicted confidence
       -> safe metadata
       -> LightGBM final classifier
```
"""
        ),
        md("## 1. Cài đặt và import"),
        code(
            """
# Bật Internet trong Kaggle Notebook trước khi chạy.
!pip install -q transformers sentencepiece lightgbm accelerate huggingface_hub
"""
        ),
        code(
            """
import gc
import json
import os
import random
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from transformers import AutoModel, AutoTokenizer

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
"""
        ),
        md("## 2. Cấu hình Kaggle và đường dẫn dữ liệu"),
        code(
            """
# Notebook tự tìm data_gold.csv trong Kaggle Input.
# Upload file thành một Kaggle Dataset rồi Add Input vào notebook.
candidate_paths = list(Path("/kaggle/input").rglob("data_gold.csv"))

if candidate_paths:
    GOLD_DATA_PATH = candidate_paths[0]
else:
    local_candidates = [
        Path("Data/gold_data/data_gold.csv"),
        Path("data_gold.csv"),
    ]
    GOLD_DATA_PATH = next((p for p in local_candidates if p.exists()), local_candidates[0])

OUTPUT_DIR = (
    Path("/kaggle/working/phobert_berf_gold_outputs")
    if Path("/kaggle/working").exists()
    else Path("phobert_berf_gold_outputs")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_PATH = OUTPUT_DIR / "phobert_embeddings.npy"
METRICS_PATH = OUTPUT_DIR / "kfold_metrics.csv"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "kfold_metrics_summary.csv"
CONFIG_PATH = OUTPUT_DIR / "config.json"

# FAST_MODE=True để kiểm tra pipeline nhanh trước khi chạy đầy đủ.
FAST_MODE = False
N_OUTER_SPLITS = 3 if FAST_MODE else 5
N_INNER_SPLITS = 2 if FAST_MODE else 3
RF_ESTIMATORS = 100 if FAST_MODE else 300

print("GOLD_DATA_PATH:", GOLD_DATA_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
"""
        ),
        md("## 3. Load và kiểm tra `data_gold.csv`"),
        code(
            """
df = pd.read_csv(GOLD_DATA_PATH)

required_columns = {
    "annotation_id", "review_id", "title", "content", "rating",
    "word_count", "is_helpful", "helpfulness_score", "confidence",
}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

for col in ["title", "content", "review_text"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)

df["text"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["content"].fillna("").astype(str)
).str.replace(r"\\s+", " ", regex=True).str.strip()

df["label"] = pd.to_numeric(df["is_helpful"], errors="raise").astype(int)
df["helpfulness_score"] = pd.to_numeric(
    df["helpfulness_score"], errors="coerce"
).fillna(0.0)
df["confidence"] = pd.to_numeric(
    df["confidence"], errors="coerce"
).fillna(0.5).clip(0, 1)

df = df[df["text"].str.len() > 0].drop_duplicates("annotation_id").reset_index(drop=True)

print("Shape:", df.shape)
display(df["label"].value_counts().rename("count").to_frame())
display(df[["label", "helpfulness_score", "confidence", "rating", "word_count"]].describe())
"""
        ),
        md(
            """
### Kiểm tra leakage

`helpfulness_score` là sản phẩm của rubric labeling và đã được dùng để quyết
định nhãn. Vì vậy không được dùng giá trị thật này như feature deployable.
Cell dưới đây chỉ đo mức liên hệ để chứng minh rủi ro.
"""
        ),
        code(
            """
leakage_corr = df[
    ["label", "helpfulness_score", "confidence", "rating", "word_count"]
].corr()["label"].sort_values(ascending=False)
display(leakage_corr.to_frame("correlation_with_label"))

known_boundary = ~df["helpfulness_score"].between(2, 3)
rule_pred = (df.loc[known_boundary, "helpfulness_score"] >= 4).astype(int)
rule_accuracy = accuracy_score(df.loc[known_boundary, "label"], rule_pred)
print(f"Score-rule accuracy ngoài vùng borderline: {rule_accuracy:.4f}")
print("Kết luận: raw helpfulness_score chỉ dùng làm supervision/weight, không dùng trực tiếp khi inference.")
"""
        ),
        md("## 4. Safe metadata và sample weights"),
        code(
            """
def add_metadata_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["title"] = data["title"].fillna("").astype(str)
    data["content"] = data["content"].fillna("").astype(str)
    data["text"] = (
        data["title"] + " " + data["content"]
    ).str.replace(r"\\s+", " ", regex=True).str.strip()

    data["content_len"] = data["content"].str.len()
    data["title_len"] = data["title"].str.len()
    data["text_len"] = data["text"].str.len()
    data["word_count_calc"] = data["text"].str.split().str.len().fillna(0)
    data["has_title"] = (data["title"].str.strip().str.len() > 0).astype(int)
    data["has_content"] = (data["content"].str.strip().str.len() > 0).astype(int)
    data["exclamation_count"] = data["text"].str.count("!")
    data["question_count"] = data["text"].str.count(r"\\?")
    data["digit_count"] = data["text"].str.count(r"\\d")
    data["uppercase_ratio"] = data["text"].apply(
        lambda x: sum(ch.isupper() for ch in str(x)) / max(len(str(x)), 1)
    )
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce").fillna(0)
    data["word_count"] = pd.to_numeric(
        data.get("word_count", data["word_count_calc"]), errors="coerce"
    ).fillna(data["word_count_calc"])
    data["is_low_rating"] = (data["rating"] <= 2).astype(int)
    data["is_high_rating"] = (data["rating"] >= 4).astype(int)
    data["is_extreme_rating"] = data["rating"].isin([1, 5]).astype(int)
    return data


metadata_cols = [
    "rating", "word_count", "content_len", "title_len", "text_len",
    "has_title", "has_content", "exclamation_count", "question_count",
    "digit_count", "uppercase_ratio", "is_low_rating", "is_high_rating",
    "is_extreme_rating",
]

df = add_metadata_features(df)

# purchased_int không được dùng vì EDA cho thấy toàn bộ giá trị bằng 0.
constant_cols = [c for c in metadata_cols if df[c].nunique(dropna=False) <= 1]
metadata_cols = [c for c in metadata_cols if c not in constant_cols]

X_meta = df[metadata_cols].fillna(0).astype("float32").values
y = df["label"].astype(int).values
annotation_targets = df[["helpfulness_score", "confidence"]].astype("float32").values

# Confidence và độ xa vùng score borderline chỉ ảnh hưởng trọng số train.
score_certainty = np.clip(np.abs(df["helpfulness_score"].values - 2.5) / 4.5, 0, 1)
sample_weights = (
    np.clip(df["confidence"].values, 0.5, 1.0)
    * (0.85 + 0.15 * score_certainty)
).astype("float32")

print("metadata_cols:", metadata_cols)
print("constant metadata removed:", constant_cols)
print("sample weight range:", sample_weights.min(), sample_weights.max())
"""
        ),
        md("## 5. Tạo PhoBERT embedding"),
        code(
            """
PHOBERT_MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 192
BATCH_SIZE = 32


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts_phobert(
    texts,
    model_name=PHOBERT_MODEL_NAME,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu().numpy().astype("float32"))

            if start % (batch_size * 20) == 0:
                print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(embeddings)


if EMBEDDING_PATH.exists():
    X_emb = np.load(EMBEDDING_PATH)
    if len(X_emb) != len(df):
        raise ValueError("Embedding cache không khớp số dòng data_gold.csv")
    print("Loaded cached embeddings:", X_emb.shape)
else:
    X_emb = encode_texts_phobert(df["text"].tolist())
    np.save(EMBEDDING_PATH, X_emb)
    print("Saved embeddings:", EMBEDDING_PATH, X_emb.shape)
"""
        ),
        md("## 6. Model factories và evaluation helpers"),
        code(
            """
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


def make_final_model():
    if HAS_LIGHTGBM:
        return LGBMClassifier(
            n_estimators=500 if not FAST_MODE else 150,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=300 if not FAST_MODE else 100,
        learning_rate=0.04,
        l2_regularization=0.01,
        random_state=SEED,
    )


def make_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def make_aux_model():
    # Ridge hỗ trợ multi-output: [helpfulness_score, confidence].
    return Ridge(alpha=10.0)


def predict_proba_positive(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    return result
"""
        ),
        md(
            """
## 7. Nested OOF stack features

Trong mỗi outer fold:

- RF probabilities cho outer-train được tạo bằng inner OOF.
- Mô hình phụ dự đoán score/confidence cho outer-train bằng inner OOF.
- Outer-validation chỉ được dự đoán bởi model fit trên toàn outer-train.

Nhờ đó final classifier không nhìn thấy in-sample stack features.
"""
        ),
        code(
            """
def build_nested_stack_features(
    X_emb_train,
    X_meta_train,
    y_train,
    annotation_train,
    X_emb_valid,
    X_meta_valid,
    seed,
):
    inner = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    n_train = len(y_train)
    rf_oof = np.zeros((n_train, 2), dtype="float32")
    aux_oof = np.zeros((n_train, 2), dtype="float32")

    for inner_fold, (fit_idx, hold_idx) in enumerate(
        inner.split(X_emb_train, y_train), start=1
    ):
        rf_inner = make_rf(seed + inner_fold)
        rf_inner.fit(
            X_emb_train[fit_idx],
            y_train[fit_idx],
        )
        rf_oof[hold_idx] = rf_inner.predict_proba(X_emb_train[hold_idx])

        meta_scaler_inner = StandardScaler()
        meta_fit = meta_scaler_inner.fit_transform(X_meta_train[fit_idx])
        meta_hold = meta_scaler_inner.transform(X_meta_train[hold_idx])
        aux_X_fit = np.hstack([X_emb_train[fit_idx], meta_fit])
        aux_X_hold = np.hstack([X_emb_train[hold_idx], meta_hold])

        aux_inner = make_aux_model()
        aux_inner.fit(aux_X_fit, annotation_train[fit_idx])
        aux_oof[hold_idx] = aux_inner.predict(aux_X_hold)

    # Models fit trên toàn outer-train để dự đoán outer-valid.
    rf_full = make_rf(seed + 100)
    rf_full.fit(X_emb_train, y_train)
    rf_valid = rf_full.predict_proba(X_emb_valid).astype("float32")

    meta_scaler_full = StandardScaler()
    meta_train_scaled = meta_scaler_full.fit_transform(X_meta_train)
    meta_valid_scaled = meta_scaler_full.transform(X_meta_valid)

    aux_full = make_aux_model()
    aux_full.fit(
        np.hstack([X_emb_train, meta_train_scaled]),
        annotation_train,
    )
    aux_valid = aux_full.predict(
        np.hstack([X_emb_valid, meta_valid_scaled])
    ).astype("float32")

    # Giữ prediction trong miền hợp lệ.
    aux_oof[:, 0] = np.clip(aux_oof[:, 0], -2, 7)
    aux_oof[:, 1] = np.clip(aux_oof[:, 1], 0, 1)
    aux_valid[:, 0] = np.clip(aux_valid[:, 0], -2, 7)
    aux_valid[:, 1] = np.clip(aux_valid[:, 1], 0, 1)

    base_train = np.hstack([rf_oof, meta_train_scaled]).astype("float32")
    base_valid = np.hstack([rf_valid, meta_valid_scaled]).astype("float32")
    extended_train = np.hstack([base_train, aux_oof]).astype("float32")
    extended_valid = np.hstack([base_valid, aux_valid]).astype("float32")

    artifacts = {
        "rf": rf_full,
        "metadata_scaler": meta_scaler_full,
        "aux_model": aux_full,
    }
    return base_train, base_valid, extended_train, extended_valid, aux_valid, artifacts
"""
        ),
        md("## 8. Outer K-Fold model comparison"),
        code(
            """
outer = StratifiedKFold(
    n_splits=N_OUTER_SPLITS,
    shuffle=True,
    random_state=SEED,
)

metrics_rows = []
oof_rows = []

for fold, (train_idx, valid_idx) in enumerate(outer.split(X_emb, y), start=1):
    print(f"\\n===== Outer Fold {fold}/{N_OUTER_SPLITS} =====")
    y_train, y_valid = y[train_idx], y[valid_idx]
    weight_train = sample_weights[train_idx]

    # 1. Safe metadata-only.
    meta_model = make_logreg()
    meta_model.fit(X_meta[train_idx], y_train)

    # 2. PhoBERT-only.
    emb_model = make_logreg()
    emb_model.fit(X_emb[train_idx], y_train)

    # 3. PhoBERT + safe metadata.
    comb_scaler = StandardScaler()
    meta_train_comb = comb_scaler.fit_transform(X_meta[train_idx])
    meta_valid_comb = comb_scaler.transform(X_meta[valid_idx])
    comb_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=SEED,
    )
    comb_model.fit(
        np.hstack([X_emb[train_idx], meta_train_comb]),
        y_train,
        sample_weight=weight_train,
    )

    # 4–5. Leakage-safe BERF stack.
    (
        berf_train,
        berf_valid,
        berf_aux_train,
        berf_aux_valid,
        aux_valid_pred,
        _,
    ) = build_nested_stack_features(
        X_emb[train_idx],
        X_meta[train_idx],
        y_train,
        annotation_targets[train_idx],
        X_emb[valid_idx],
        X_meta[valid_idx],
        seed=SEED + fold * 1000,
    )

    berf_model = make_final_model()
    berf_model.fit(berf_train, y_train, sample_weight=weight_train)

    berf_aux_model = make_final_model()
    berf_aux_model.fit(berf_aux_train, y_train, sample_weight=weight_train)

    candidates = [
        ("metadata_logreg", meta_model, X_meta[valid_idx]),
        ("phobert_logreg", emb_model, X_emb[valid_idx]),
        (
            "phobert_metadata_logreg",
            comb_model,
            np.hstack([X_emb[valid_idx], meta_valid_comb]),
        ),
        ("berf_nested_oof", berf_model, berf_valid),
        ("berf_nested_oof_aux_pred", berf_aux_model, berf_aux_valid),
    ]

    for model_name, model, X_valid_model in candidates:
        proba = predict_proba_positive(model, X_valid_model)
        pred = (proba >= 0.5).astype(int)
        metrics_rows.append({
            "fold": fold,
            "model": model_name,
            **compute_metrics(y_valid, pred, proba),
        })

        if model_name == "berf_nested_oof_aux_pred":
            fold_oof = df.iloc[valid_idx][[
                "annotation_id", "review_id", "sample_group",
                "title", "content", "label",
            ]].copy()
            fold_oof["fold"] = fold
            fold_oof["proba_helpful"] = proba
            fold_oof["pred"] = pred
            fold_oof["pred_helpfulness_score"] = aux_valid_pred[:, 0]
            fold_oof["pred_confidence"] = aux_valid_pred[:, 1]
            oof_rows.append(fold_oof)

    gc.collect()

metrics_df = pd.DataFrame(metrics_rows)
oof_df = pd.concat(oof_rows, ignore_index=True)
metrics_df.to_csv(METRICS_PATH, index=False)
oof_df.to_csv(OOF_PATH, index=False)
display(metrics_df)
"""
        ),
        md("## 9. Tổng hợp kết quả và ablation"),
        code(
            """
summary = (
    metrics_df.groupby("model")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
    )
    .sort_values("f1_mean", ascending=False)
)
summary.to_csv(SUMMARY_PATH)
display(summary)

plot_df = summary.reset_index()
plt.figure(figsize=(11, 5))
sns.barplot(data=plot_df, x="f1_mean", y="model", color="#2563EB")
plt.xlim(0, 1)
plt.title("Mean F1 across outer folds")
plt.xlabel("F1")
plt.ylabel("")
plt.tight_layout()
plt.show()

base_f1 = summary.loc["berf_nested_oof", "f1_mean"]
aux_f1 = summary.loc["berf_nested_oof_aux_pred", "f1_mean"]
print(f"Ablation delta from predicted score/confidence: {aux_f1 - base_f1:+.4f} F1")
"""
        ),
        md("## 10. OOF evaluation cho model mở rộng"),
        code(
            """
y_true_oof = oof_df["label"].astype(int).values
y_pred_oof = oof_df["pred"].astype(int).values
y_proba_oof = oof_df["proba_helpful"].values

oof_metrics = compute_metrics(y_true_oof, y_pred_oof, y_proba_oof)
print(json.dumps(oof_metrics, indent=2))
print("\\nClassification report:")
print(classification_report(y_true_oof, y_pred_oof, digits=4))

cm = confusion_matrix(y_true_oof, y_pred_oof)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Helpful", "Helpful"],
    yticklabels=["Not Helpful", "Helpful"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("OOF confusion matrix")
plt.tight_layout()
plt.show()

aux_quality = oof_df[
    ["pred_helpfulness_score", "pred_confidence"]
].describe()
display(aux_quality)
"""
        ),
        md(
            """
## 11. Train final deployable pipeline

Final classifier được train bằng:

- RF OOF probabilities trên toàn training set.
- Predicted score/confidence OOF trên toàn training set.
- Metadata đã scale.

Các artifact full-data được lưu riêng để tạo đúng feature khi inference.
"""
        ),
        code(
            """
# Tạo OOF stack cho toàn bộ data, đồng thời nhận full-data artifacts.
(
    final_base_train,
    _,
    final_extended_train,
    _,
    _,
    final_stack_artifacts,
) = build_nested_stack_features(
    X_emb,
    X_meta,
    y,
    annotation_targets,
    X_emb[:1],   # dummy valid row; artifacts full-data vẫn hợp lệ
    X_meta[:1],
    seed=SEED + 9000,
)

final_classifier = make_final_model()
final_classifier.fit(
    final_extended_train,
    y,
    sample_weight=sample_weights,
)

FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    final_stack_artifacts["rf"],
    FINAL_MODEL_DIR / "rf_embedding_probability_model.joblib",
)
joblib.dump(
    final_stack_artifacts["metadata_scaler"],
    FINAL_MODEL_DIR / "metadata_scaler.joblib",
)
joblib.dump(
    final_stack_artifacts["aux_model"],
    FINAL_MODEL_DIR / "annotation_feature_predictor.joblib",
)
joblib.dump(
    final_classifier,
    FINAL_MODEL_DIR / "final_ml_model.joblib",
)

config = {
    "embedding_model": PHOBERT_MODEL_NAME,
    "max_length": MAX_LENGTH,
    "metadata_cols": metadata_cols,
    "target": "is_helpful",
    "label_mapping": {"0": "Not Helpful", "1": "Helpful"},
    "stacking": "nested_oof",
    "final_features": [
        "rf_proba_not_helpful",
        "rf_proba_helpful",
        *metadata_cols,
        "predicted_helpfulness_score",
        "predicted_confidence",
    ],
    "annotation_features": {
        "raw_values_used_as_inputs": False,
        "training_supervision": ["helpfulness_score", "confidence"],
        "inference_features": [
            "predicted_helpfulness_score",
            "predicted_confidence",
        ],
        "confidence_used_as_sample_weight": True,
    },
    "has_lightgbm": HAS_LIGHTGBM,
}
CONFIG_PATH.write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("Saved final artifacts to:", FINAL_MODEL_DIR)
"""
        ),
        md("## 12. Inference helper"),
        code(
            """
def build_final_features_for_inference(embedding, metadata):
    rf_model = final_stack_artifacts["rf"]
    meta_scaler = final_stack_artifacts["metadata_scaler"]
    aux_model = final_stack_artifacts["aux_model"]

    metadata_scaled = meta_scaler.transform(metadata)
    rf_proba = rf_model.predict_proba(embedding)
    aux_pred = aux_model.predict(np.hstack([embedding, metadata_scaled]))
    aux_pred[:, 0] = np.clip(aux_pred[:, 0], -2, 7)
    aux_pred[:, 1] = np.clip(aux_pred[:, 1], 0, 1)
    return np.hstack([rf_proba, metadata_scaled, aux_pred])


# Smoke test bằng dòng đầu tiên.
example_features = build_final_features_for_inference(X_emb[:1], X_meta[:1])
example_proba = final_classifier.predict_proba(example_features)[0, 1]
print("Final feature shape:", example_features.shape)
print("Example helpful probability:", float(example_proba))
"""
        ),
        md("## 13. Lưu report và model card"),
        code(
            """
report = {
    "dataset_rows": int(len(df)),
    "label_distribution": {
        str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
    },
    "best_model": str(summary.index[0]),
    "best_metrics": {
        key: float(value)
        for key, value in summary.iloc[0].to_dict().items()
    },
    "oof_extended_berf_metrics": oof_metrics,
    "leakage_controls": {
        "raw_helpfulness_score_as_feature": False,
        "raw_confidence_as_feature": False,
        "nested_oof_rf_probabilities": True,
        "nested_oof_predicted_annotation_features": True,
    },
}
(OUTPUT_DIR / "training_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

model_card = f'''---
language:
- vi
license: mit
pipeline_tag: text-classification
tags:
- phobert
- berf
- nested-oof
- review-helpfulness
---

# Tiki Review Helpfulness — PhoBERT BERF Gold v2

## Dataset

- Rows: {len(df)}
- Target: `is_helpful`
- Helpful: {int(df["label"].sum())}
- Not Helpful: {int((df["label"] == 0).sum())}

## Leakage-safe pipeline

```text
review -> PhoBERT
       -> nested-OOF RF probabilities
       -> nested-OOF predicted helpfulness_score/confidence
       -> safe metadata
       -> final classifier
```

Raw `helpfulness_score` and `confidence` are annotation outputs and are not
used as inference inputs. They only supervise auxiliary predictions and sample
weights during training.

## Best cross-validation model

`{summary.index[0]}`

See `kfold_metrics_summary.csv`, `oof_predictions.csv`, `training_report.json`
and `config.json` for full results.
'''
(OUTPUT_DIR / "README.md").write_text(model_card, encoding="utf-8")
print("Saved report and model card to:", OUTPUT_DIR)
"""
        ),
        md("## 14. Optional: upload artifacts lên Hugging Face"),
        code(
            """
# Chỉ chạy cell này sau khi thêm HF_TOKEN vào Kaggle Secrets.
# Không hard-code token trong notebook.

UPLOAD_TO_HF = False
HF_REPO_ID = "DPhO05/tiki-review-helpfulness-phobert-berf-gold-v2"

if UPLOAD_TO_HF:
    from huggingface_hub import HfApi, create_repo
    from kaggle_secrets import UserSecretsClient

    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    create_repo(
        repo_id=HF_REPO_ID,
        token=hf_token,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    HfApi(token=hf_token).upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Upload leakage-safe PhoBERT BERF gold v2",
    )
    print("Uploaded to:", f"https://huggingface.co/{HF_REPO_ID}")
else:
    print("UPLOAD_TO_HF=False — skipped.")
"""
        ),
        md(
            """
## 15. Ghi chú diễn giải kết quả

- So sánh `berf_nested_oof` với `berf_nested_oof_aux_pred` để đo đóng góp thực
  của predicted score/confidence.
- Nếu model mở rộng không cải thiện ổn định qua các fold, loại hai feature phụ.
- Không báo cáo kết quả từ một model dùng trực tiếp raw `helpfulness_score` hoặc
  raw `confidence`; đó là label leakage.
- Nên giữ một human-labeled holdout riêng nếu muốn đánh giá khả năng tổng quát
  ngoài nhãn Codex/Gemini.
"""
        ),
    ]

    nb["cells"] = cells
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Created: {OUTPUT}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
