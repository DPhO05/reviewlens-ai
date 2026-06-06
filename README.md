# ReviewLens AI

Product demo đánh giá mức độ hữu ích của review thương mại điện tử.

## Kiến trúc

```text
Streamlit -> FastAPI -> embedding -> Random Forest predict_proba
                              -> concat features -> final classifier
                              -> Gemini/local explanation
```

Gemini chỉ là lớp diễn giải. Nhãn `Helpful`/`Not Helpful` do pipeline ML quyết định.

Repo đi kèm artifact **demo tổng hợp** để có thể chạy offline. Artifact này dùng
HashingVectorizer 384 chiều, Random Forest và Gradient Boosting final classifier,
theo đúng thứ tự feature của BERF. Không dùng các metric demo để đưa ra kết luận
nghiên cứu. Khi có model thật, thay artifacts trong `models/` và adapter embedding.

## Chạy local

Yêu cầu Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_demo_models.py
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

Ứng dụng hỗ trợ hai chế độ:

- `API_BASE_URL` để trống: Streamlit chạy predictor và Gemini trực tiếp trong
  cùng process. Đây là chế độ phù hợp với Community Cloud.
- `API_BASE_URL` có giá trị: Streamlit gọi FastAPI được deploy ở URL đó.

1. Push toàn bộ project lên một GitHub repository. Không push `.env`.
2. Vào `https://share.streamlit.io`, chọn **Create app**.
3. Chọn repository, branch và entrypoint `app/streamlit_app.py`.
4. Chọn Python 3.11 trong **Advanced settings**.
5. Điền secrets:

```toml
GEMINI_API_KEY = "your-new-gemini-key"
GEMINI_MODEL = "gemini-2.5-flash"
API_BASE_URL = ""
```

Không cần điền `HF_TOKEN` cho artifact demo hiện tại. Khi có Hugging Face
Inference Endpoint, thêm `HF_TOKEN`, `HF_ENDPOINT_URL` và adapter gọi endpoint.

## Test

```bash
python3 -m pytest -q
```

## Thay bằng model nghiên cứu

1. Lưu đúng embedding model đã dùng lúc training.
2. Thay `models/rf_model.pkl` và `models/lgbm_model.pkl`.
3. Cập nhật dimension, threshold và class mapping trong `model_config.json`.
4. Thay `DemoEmbeddingModel` bằng adapter Sentence-BERT tương ứng.
5. Xuất metrics test set thật để thay dashboard minh họa.
