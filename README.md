# ReviewLens AI

Product demo đánh giá mức độ hữu ích của review thương mại điện tử.

## Kiến trúc

```text
Streamlit -> FastAPI -> embedding -> Random Forest predict_proba
                              -> concat features -> final classifier
                              -> Gemini/local explanation
```

Gemini chỉ là lớp diễn giải. Nhãn `Helpful`/`Not Helpful` do pipeline ML quyết định.

Model mặc định là
[`DPhO05/tiki-review-helpfulness-phobert-berf-gold-v2`](https://huggingface.co/DPhO05/tiki-review-helpfulness-phobert-berf-gold-v2).
Pipeline chạy PhoBERT, Random Forest probability features, metadata đã scale,
predicted rubric score/confidence và final LightGBM classifier. Artifact được tải
một lần vào `models/hf_gold_v2` và tái sử dụng ở các lần chạy sau.

## Chạy local

Yêu cầu Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Terminal 1:

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:

```bash
python3 -m streamlit run app/streamlit_app.py
```

Mở `http://localhost:8501`. API docs ở `http://localhost:8000/docs`.

## Gemini

```bash
cp .env.example .env
```

Điền `GEMINI_API_KEY`. Nếu không có key hoặc Gemini lỗi, hệ thống tự dùng giải
thích local dựa trên reason codes và vẫn hoàn thành prediction.

## API

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/analyze-review \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Tai nghe Bluetooth",
    "category": "electronics",
    "rating": 4,
    "review_text": "Dùng 2 tuần, pin 6 giờ, âm thanh rõ nhưng mic hơi rè.",
    "use_gemini": false
  }'
```

Endpoint:

- `GET /health`
- `POST /analyze-review`
- `POST /rank-reviews`

## Docker

```bash
docker compose up --build
```

Frontend: `http://localhost:8501`, backend: `http://localhost:8000`.

## Streamlit Community Cloud

Ứng dụng hỗ trợ hai chế độ runtime:

- `API_BASE_URL` để trống: Streamlit chạy predictor và Gemini trực tiếp trong
  cùng process. Đây là chế độ phù hợp với Community Cloud.
- `API_BASE_URL` có giá trị: Streamlit gọi FastAPI được deploy ở URL đó.

Model backend:

- `MODEL_BACKEND=hf_gold_v2`: model PhoBERT BERF gold-v2, mặc định.
- `MODEL_BACKEND=demo`: artifact demo nhẹ, chỉ nên dùng cho test/offline fallback.

1. Push toàn bộ project lên một GitHub repository. Không push `.env`.
2. Vào `https://share.streamlit.io`, chọn **Create app**.
3. Chọn repository, branch và entrypoint `app/streamlit_app.py`.
4. Chọn Python 3.11 trong **Advanced settings**.
5. Điền secrets:

```toml
GEMINI_API_KEY = "your-new-gemini-key"
GEMINI_MODEL = "gemini-2.5-flash"
API_BASE_URL = ""
MODEL_BACKEND = "hf_gold_v2"
HF_MODEL_ID = "DPhO05/tiki-review-helpfulness-phobert-berf-gold-v2"
HF_TOKEN = "your-hugging-face-read-token"
```

Repo model đang public nên `HF_TOKEN` có thể để trống, nhưng nên cấu hình token
read để tránh rate limit khi tải artifact/PhoBERT.

Kiểm tra token, model repo và endpoint:

```bash
python3 scripts/model/test_hf_config.py
```

Nếu chưa có `HF_ENDPOINT_URL`, script vẫn kiểm tra token và các artifact trên
model repo, sau đó bỏ qua inference.

## Test

```bash
python3 -m pytest -q
```

Chạy hoặc tái tạo notebook EDA:

```bash
pip install -r requirements-notebooks.txt
python3 scripts/kaggle/create_and_run_labeled_9k_eda_notebooks.py
```

## Cấu trúc project

Xem [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) để biết vị trí app code,
script xử lý dữ liệu, notebook Kaggle và artifact.

## Gold-v2 inference features

```text
review + rating
-> vinai/phobert-base-v2 embedding
-> Random Forest predict_proba
-> 13 metadata features + StandardScaler
-> predicted helpfulness_score + predicted confidence
-> final LightGBM predict_proba
```

Raw labeling score/confidence không được truyền vào inference; auxiliary model
tự dự đoán hai feature này từ embedding và metadata.
