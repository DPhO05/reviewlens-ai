from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gemini_service import GeminiReviewExplainer
from src.predictor import ReviewHelpfulnessPredictor

load_dotenv()


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


API_BASE_URL = get_setting("API_BASE_URL").rstrip("/")
GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(
    page_title="ReviewLens AI",
    page_icon="RL",
    layout="wide",
)


@st.cache_resource
def local_services() -> tuple[ReviewHelpfulnessPredictor, GeminiReviewExplainer]:
    predictor = ReviewHelpfulnessPredictor(str(ROOT / "models"))
    explainer = GeminiReviewExplainer(
        api_key=GEMINI_API_KEY or None,
        model=GEMINI_MODEL,
    )
    return predictor, explainer


def analyze_locally(payload: dict[str, Any]) -> dict[str, Any]:
    predictor, explainer = local_services()
    model_result = predictor.predict(payload["review_text"])
    explanation = None
    if payload.get("use_gemini", True):
        explanation = explainer.explain(
            product_info={
                "product_name": payload["product_name"],
                "category": payload["category"],
                "rating": payload["rating"],
                "price": payload.get("price"),
                "brand": payload.get("brand"),
                "verified_purchase": payload.get("verified_purchase"),
            },
            review_text=payload["review_text"],
            model_result=model_result,
        )
    return {**model_result, "gemini_explanation": explanation}


def rank_locally(payload: dict[str, Any]) -> dict[str, Any]:
    ranked = []
    for index, review in enumerate(payload["reviews"]):
        result = analyze_locally({**review, "use_gemini": False})
        ranked.append({"original_index": index, **review, **result})
    ranked.sort(key=lambda item: item["helpfulness_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {"count": len(ranked), "reviews": ranked}


def call_service(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not API_BASE_URL:
        if path == "/analyze-review":
            return analyze_locally(payload)
        if path == "/rank-reviews":
            return rank_locally(payload)
        raise ValueError(f"Unsupported local path: {path}")

    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def sidebar() -> str:
    st.sidebar.title("ReviewLens AI")
    st.sidebar.caption("Review helpfulness intelligence")
    page = st.sidebar.radio(
        "Chức năng",
        ["Single Review Analyzer", "Batch Review Ranking", "Model Evaluation"],
    )
    st.sidebar.divider()
    st.sidebar.info(
        "Model quyết định nhãn là pipeline BERF-style. Gemini chỉ diễn giải kết quả."
    )
    st.sidebar.caption(
        "Runtime: "
        + (f"FastAPI · {API_BASE_URL}" if API_BASE_URL else "Streamlit direct mode")
    )
    return page


def explanation_card(explanation: dict[str, Any] | None) -> None:
    if not explanation:
        st.info("Gemini explanation đang tắt.")
        return
    st.subheader(explanation.get("short_assessment", "Giải thích"))

    score_info = explanation.get("score_interpretation", {})
    if score_info:
        st.markdown("### Diễn giải kết quả model")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Label", score_info.get("label", "N/A"))
        score = score_info.get("helpfulness_score")
        metric_columns[1].metric(
            "Score",
            f"{float(score) * 100:.1f}%" if isinstance(score, (int, float)) else "N/A",
        )
        metric_columns[2].metric(
            "Confidence", score_info.get("confidence_level", "N/A")
        )
        st.write(score_info.get("explanation", ""))

    st.markdown("### Vì sao model đưa ra kết quả này?")
    for reason in explanation.get("why", []):
        st.write(f"- {reason}")

    evidence = explanation.get("evidence_analysis", [])
    if evidence:
        st.markdown("### Phân tích bằng chứng")
        for index, item in enumerate(evidence, start=1):
            criterion = item.get("criterion", f"Tiêu chí {index}")
            status = item.get("status", "N/A")
            with st.expander(f"{index}. {criterion} · {status}", expanded=index <= 2):
                quote = item.get("evidence")
                if quote:
                    st.markdown(f'**Bằng chứng trong review:** “{quote}”')
                else:
                    st.markdown("**Bằng chứng trong review:** Chưa có.")
                st.markdown("**Phân tích**")
                st.write(item.get("analysis", ""))
                st.markdown("**Tác động tới người mua**")
                st.write(item.get("buyer_impact", ""))

    strengths, limitations = st.columns(2)
    with strengths:
        st.markdown("### Điểm mạnh")
        items = explanation.get("strengths", [])
        if items:
            for item in items:
                st.write(f"- {item}")
        else:
            st.write("Chưa có điểm mạnh nổi bật được xác định.")
    with limitations:
        st.markdown("### Hạn chế")
        items = explanation.get("limitations", [])
        if items:
            for item in items:
                st.write(f"- {item}")
        else:
            st.write("Không có hạn chế đáng kể được xác định.")

    missing_information = explanation.get("missing_information", [])
    if missing_information:
        st.markdown("### Thông tin còn thiếu")
        for item in missing_information:
            st.write(f"- {item}")

    st.markdown("**Gợi ý cải thiện**")
    st.write(explanation.get("improvement_suggestion", ""))
    st.markdown("**Phiên bản viết lại đề xuất**")
    st.info(explanation.get("suggested_rewrite", ""))
    st.markdown("**Giá trị cho người mua**")
    st.write(explanation.get("buyer_value", ""))
    st.markdown("**Mức độ bám sát model**")
    st.write(explanation.get("model_alignment", ""))
    if explanation.get("source") != "gemini":
        reason = explanation.get("fallback_reason")
        detail = f" Mã lỗi: `{reason}`." if reason else ""
        st.caption(
            "Đang dùng giải thích local vì chưa cấu hình Gemini hoặc API lỗi."
            + detail
        )


def single_analyzer() -> None:
    st.title("Single Review Analyzer")
    st.write("Chấm điểm mức độ hữu ích của một bình luận sản phẩm.")
    with st.form("single-review"):
        left, right = st.columns(2)
        product_name = left.text_input(
            "Tên sản phẩm", "Tai nghe Bluetooth Xiaomi Redmi Buds 6"
        )
        category = left.selectbox(
            "Danh mục", ["electronics", "beauty", "food", "fashion", "other"]
        )
        rating = left.slider("Rating", 1, 5, 4)
        brand = right.text_input("Thương hiệu", "Xiaomi")
        price = right.number_input("Giá", min_value=0.0, value=650000.0, step=10000.0)
        verified = right.checkbox("Verified purchase", value=True)
        review_text = st.text_area(
            "Nội dung review",
            "Mình dùng được 2 tuần, pin khoảng 5-6 tiếng, âm thanh ổn trong "
            "tầm giá. Mic gọi trong phòng rõ nhưng ra đường hơi rè.",
            height=150,
        )
        use_gemini = st.checkbox("Bật lớp giải thích Gemini", value=True)
        submitted = st.form_submit_button("Phân tích review", type="primary")

    if submitted:
        if not product_name.strip() or not review_text.strip():
            st.error("Tên sản phẩm và nội dung review không được để trống.")
            return
        payload = {
            "product_name": product_name,
            "category": category,
            "rating": rating,
            "price": price,
            "brand": brand or None,
            "verified_purchase": verified,
            "review_text": review_text,
            "use_gemini": use_gemini,
        }
        try:
            with st.spinner("Đang phân tích..."):
                result = call_service("/analyze-review", payload)
        except (requests.RequestException, ValueError, OSError) as exc:
            target = API_BASE_URL or "local Streamlit runtime"
            st.error(f"Không chạy được inference tại {target}: {exc}")
            return

        score = result["helpfulness_score"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Helpfulness score", f"{score * 100:.1f}%")
        col2.metric("Label", result["label"])
        col3.metric("Confidence", result["confidence_level"])
        st.progress(score)
        reason_codes = result.get("model_reason_codes", [])
        if reason_codes:
            st.caption("Model reason codes: " + ", ".join(reason_codes))
        st.divider()
        explanation_card(result.get("gemini_explanation"))


def batch_ranking() -> None:
    st.title("Batch Review Ranking")
    st.write("Upload CSV và xếp hạng review theo helpfulness score.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        st.download_button(
            "Tải CSV mẫu",
            data=(
                "product_name,category,rating,price,review_text\n"
                'Tai nghe A,electronics,4,650000,"Pin tốt, mic hơi rè khi gọi ngoài đường."\n'
                'Tai nghe A,electronics,5,650000,"Ok shop."\n'
            ),
            file_name="sample_reviews.csv",
            mime="text/csv",
        )
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Không đọc được CSV: {exc}")
        return

    required = {"product_name", "category", "rating", "review_text"}
    missing = required - set(frame.columns)
    if missing:
        st.error("CSV thiếu cột: " + ", ".join(sorted(missing)))
        return
    st.dataframe(frame.head(20), use_container_width=True)

    if st.button("Xếp hạng", type="primary"):
        reviews = []
        for row in frame.fillna("").to_dict(orient="records"):
            reviews.append(
                {
                    "product_name": str(row["product_name"]),
                    "category": str(row["category"]),
                    "rating": int(row["rating"]),
                    "price": float(row["price"]) if row.get("price") != "" else None,
                    "review_text": str(row["review_text"]),
                    "use_gemini": False,
                }
            )
        try:
            result = call_service("/rank-reviews", {"reviews": reviews})
        except (requests.RequestException, ValueError, OSError) as exc:
            st.error(f"Không chạy được batch inference: {exc}")
            return
        ranked = pd.DataFrame(result["reviews"])
        visible = [
            "rank", "product_name", "rating", "review_text", "label",
            "helpfulness_score", "confidence_level",
        ]
        st.dataframe(ranked[visible], use_container_width=True, hide_index=True)
        st.download_button(
            "Tải kết quả",
            ranked.to_csv(index=False).encode("utf-8-sig"),
            "ranked_reviews.csv",
            "text/csv",
        )


def evaluation_dashboard() -> None:
    st.title("Model Evaluation Dashboard")
    st.warning(
        "Các số liệu dưới đây là dữ liệu minh họa giao diện, không phải kết quả "
        "đánh giá của model nghiên cứu."
    )
    metrics = {"Accuracy": 0.86, "Precision": 0.88, "Recall": 0.84, "F1-score": 0.86}
    cols = st.columns(4)
    for column, (name, value) in zip(cols, metrics.items()):
        column.metric(name, f"{value:.2f}")

    left, right = st.columns(2)
    confusion = pd.DataFrame(
        [[420, 68], [74, 438]],
        index=["Actual Not Helpful", "Actual Helpful"],
        columns=["Pred Not Helpful", "Pred Helpful"],
    )
    figure = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion matrix (demo)",
    )
    left.plotly_chart(figure, use_container_width=True)

    comparison = pd.DataFrame(
        {
            "Model": ["Embedding only", "BERF-style"],
            "F1-score": [0.81, 0.86],
        }
    )
    right.plotly_chart(
        px.bar(
            comparison,
            x="Model",
            y="F1-score",
            color="Model",
            range_y=[0, 1],
            title="Model comparison (demo)",
        ),
        use_container_width=True,
    )
    st.caption(
        "Thay dữ liệu minh họa bằng metrics.json và confusion matrix từ test set "
        "khi có model đã train chính thức."
    )


page = sidebar()
if page == "Single Review Analyzer":
    single_analyzer()
elif page == "Batch Review Ranking":
    batch_ranking()
else:
    evaluation_dashboard()
