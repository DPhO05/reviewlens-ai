from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.demo_models import DemoEmbeddingModel  # noqa: E402
from src.preprocessing import clean_text  # noqa: E402


HELPFUL = [
    "Dùng 2 tuần, pin được khoảng 6 giờ. Âm thanh rõ trong phòng nhưng mic hơi rè ngoài đường.",
    "Máy chạy mượt các tác vụ văn phòng, màn hình sáng và pin dùng từ 8 giờ đến 17 giờ còn 20%.",
    "Kem thấm sau khoảng 3 phút, không nhờn. Da mình bớt khô sau 10 ngày nhưng mùi hơi nồng.",
    "Áo đúng bảng size, vải dày vừa và đường may chắc. Mình cao 1m68 nặng 58kg mặc size M vừa.",
    "Đóng gói chắc, giao trong 2 ngày. Sản phẩm dễ lắp, hướng dẫn rõ nhưng dây nguồn hơi ngắn.",
    "Giá hợp lý so với chất lượng. Camera ban ngày tốt, ban đêm nhiễu nhẹ và lấy nét chậm.",
    "Sau một tháng sử dụng chưa gặp lỗi. Bàn phím êm, touchpad chính xác, loa hơi nhỏ.",
    "Vị ít ngọt, thơm nhẹ. Hộp có 20 gói, mỗi gói pha được khoảng 250ml.",
]

NOT_HELPFUL = [
    "Ok shop.",
    "Hàng tốt.",
    "Đẹp lắm mọi người ơi.",
    "Giao nhanh.",
    "Sản phẩm tuyệt vời.",
    "Không thích.",
    "Tạm được.",
    "Năm sao nhé.",
]


def build_dataset() -> tuple[list[str], np.ndarray]:
    random.seed(42)
    texts: list[str] = []
    labels: list[int] = []
    prefixes = ["", "Mình thấy ", "Theo trải nghiệm của mình, ", "Mua cho gia đình, "]
    suffixes = ["", " Mọi người có thể tham khảo.", " Đây là trải nghiệm cá nhân."]
    for _ in range(18):
        for label, samples in ((1, HELPFUL), (0, NOT_HELPFUL)):
            for sample in samples:
                texts.append(random.choice(prefixes) + sample + random.choice(suffixes))
                labels.append(label)
    return texts, np.asarray(labels)


def main() -> None:
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    texts, labels = build_dataset()
    embedding_model = DemoEmbeddingModel(384)
    embeddings = embedding_model.encode([clean_text(text) for text in texts])

    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(embeddings, labels)
    rf_proba = rf.predict_proba(embeddings)
    berf_features = np.concatenate([embeddings, rf_proba], axis=1)

    final_model = GradientBoostingClassifier(
        n_estimators=90,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )
    final_model.fit(berf_features, labels)

    joblib.dump(rf, model_dir / "rf_model.pkl")
    joblib.dump(final_model, model_dir / "lgbm_model.pkl")
    config = {
        "embedding_model_name": "demo-hashing-char-ngram",
        "embedding_dim": 384,
        "rf_probability_dim": 2,
        "final_feature_dim": 386,
        "threshold": 0.5,
        "helpful_class": 1,
        "label_mapping": {"0": "Not Helpful", "1": "Helpful"},
        "model_variant": "offline-demo-berf-style",
        "warning": "Synthetic demo artifacts; replace with trained BERF-LGBM artifacts for research claims.",
    }
    (model_dir / "model_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    print(f"Saved demo artifacts to {model_dir}")


if __name__ == "__main__":
    main()
