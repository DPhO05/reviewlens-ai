# ReviewLens AI - Instruction triển khai Product Demo

## 1. Mục tiêu demo

**ReviewLens AI** là một demo sản phẩm dùng để đánh giá mức độ hữu ích của bình luận sản phẩm trên sàn thương mại điện tử.

Người dùng nhập thông tin sản phẩm và nội dung bình luận. Hệ thống sẽ:

1. Tiền xử lý nội dung bình luận.
2. Tạo embedding từ mô hình BERT hoặc Sentence-BERT.
3. Đưa embedding qua Random Forest để sinh ra `class probability features`.
4. Ghép embedding với probability features để tạo tập đặc trưng mới theo hướng BERF.
5. Đưa tập feature mới vào mô hình final, ví dụ LGBM, để dự đoán review là `Helpful` hoặc `Not Helpful`.
6. Gọi Gemini API để sinh giải thích bằng ngôn ngữ tự nhiên, giúp người dùng hiểu vì sao review đó hữu ích hoặc chưa hữu ích.

Điểm quan trọng khi thuyết trình:

> Gemini không phải model phân loại chính. Model chính là pipeline ML đã train: BERT embedding -> Random Forest probability feature -> LGBM classifier. Gemini chỉ đóng vai trò explanation layer để diễn giải kết quả model cho người dùng cuối.

---

## 2. Cấu trúc thư mục dự án

```text
reviewlens-ai/
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── src/
│   ├── preprocessing.py
│   ├── feature_pipeline.py
│   ├── predictor.py
│   └── gemini_service.py
├── models/
│   ├── rf_model.pkl
│   ├── lgbm_model.pkl
│   └── embedding_model/
├── requirements.txt
├── Dockerfile
└── README.md
```

Ý nghĩa các thư mục:

| Thành phần | Vai trò |
|---|---|
| `app/` | Chứa giao diện Streamlit cho người dùng demo |
| `api/` | Chứa FastAPI backend để phục vụ endpoint inference |
| `src/` | Chứa logic xử lý chính: preprocessing, tạo feature, predict, gọi Gemini |
| `models/` | Chứa model đã train và các artifact liên quan |
| `requirements.txt` | Danh sách thư viện Python cần cài |
| `Dockerfile` | File đóng gói ứng dụng để deploy bằng Docker |
| `README.md` | Hướng dẫn chạy project |

---

## 3. Input đầu vào của sản phẩm

MVP nên dùng các input sau:

| Input | Bắt buộc | Ý nghĩa |
|---|---:|---|
| `product_name` | Có | Tên sản phẩm được đánh giá |
| `category` | Có | Danh mục sản phẩm, ví dụ electronics, beauty, food |
| `rating` | Có | Số sao người dùng đánh giá, từ 1 đến 5 |
| `review_text` | Có | Nội dung bình luận cần đánh giá |
| `price` | Không | Giá sản phẩm, giúp Gemini hiểu thêm ngữ cảnh |
| `brand` | Không | Thương hiệu sản phẩm |
| `verified_purchase` | Không | Người đánh giá có phải người mua thật hay không |

Input tối thiểu để model chạy:

```json
{
  "product_name": "Tai nghe Bluetooth Xiaomi Redmi Buds 6",
  "category": "electronics",
  "rating": 4,
  "review_text": "Mình dùng được 2 tuần, pin khoảng 5-6 tiếng, âm thanh ổn trong tầm giá. Mic gọi trong phòng rõ nhưng ra đường hơi rè."
}
```

---

## 4. Output đầu ra của sản phẩm

Output nên có cả kết quả định lượng từ model và phần giải thích từ Gemini.

Ví dụ response chuẩn:

```json
{
  "label": "Helpful",
  "helpfulness_score": 0.91,
  "confidence_level": "High",
  "model_reason_codes": [
    "specific_experience",
    "mentions_product_features",
    "mentions_pros_and_cons"
  ],
  "gemini_explanation": "Review này hữu ích vì người dùng nêu trải nghiệm thực tế sau 2 tuần sử dụng, cung cấp thông tin cụ thể về pin, âm thanh và mic, đồng thời chỉ ra cả ưu điểm và nhược điểm.",
  "improvement_suggestion": "Có thể bổ sung thêm ảnh thực tế hoặc trải nghiệm khi chơi game/nghe gọi lâu để review thuyết phục hơn."
}
```

Ý nghĩa:

| Field | Ý nghĩa |
|---|---|
| `label` | Nhãn dự đoán: `Helpful` hoặc `Not Helpful` |
| `helpfulness_score` | Xác suất review hữu ích, lấy từ model final |
| `confidence_level` | Mức độ tự tin: Low, Medium, High |
| `model_reason_codes` | Các lý do dạng rule-based/heuristic để hỗ trợ explain |
| `gemini_explanation` | Giải thích tự nhiên do Gemini sinh ra |
| `improvement_suggestion` | Gợi ý cách cải thiện review |

---

## 5. Pipeline inference sau khi đã train model

Sau khi train xong, inference phải chạy đúng thứ tự như lúc training.

```text
User input
    ↓
Preprocess review text
    ↓
Generate BERT/Sentence-BERT embedding
    ↓
Random Forest predict_proba
    ↓
Concat embedding + RF probability features
    ↓
LGBM predict_proba
    ↓
Convert probability to label and confidence
    ↓
Call Gemini API for explanation
    ↓
Return final response to UI
```

Pseudo-code:

```python
clean_text = preprocess_text(review_text)

embedding = embedding_model.encode([clean_text])
# shape: (1, 768) nếu dùng BERT/Sentence-BERT thường gặp

rf_proba = rf_model.predict_proba(embedding)
# shape: (1, 2), ví dụ [P(Not Helpful), P(Helpful)]

berf_feature = np.concatenate([embedding, rf_proba], axis=1)
# shape: (1, 770) nếu embedding là 768 chiều và RF probability có 2 chiều

final_proba = lgbm_model.predict_proba(berf_feature)
helpfulness_score = final_proba[0][1]

label = "Helpful" if helpfulness_score >= threshold else "Not Helpful"
```

Lưu ý rất quan trọng:

- Nếu lúc train dùng embedding 768 chiều, lúc inference cũng phải dùng đúng embedding model đó.
- Nếu lúc train dùng `embedding + rf_proba`, lúc inference cũng phải concat đúng thứ tự đó.
- Nếu có scaler, PCA hoặc label encoder trong training, bắt buộc save lại và load lại khi inference.
- Không được train lại Random Forest trong lúc inference. RF chỉ được dùng để `predict_proba`.

---

## 6. Các model artifact cần lưu

Sau training, cần lưu ít nhất:

```text
models/
├── rf_model.pkl
├── lgbm_model.pkl
├── embedding_model/
└── model_config.json
```

Nếu có dùng thêm preprocessing artifact thì lưu thêm:

```text
models/
├── scaler.pkl
├── pca.pkl
├── label_encoder.pkl
└── threshold.json
```

Ví dụ `model_config.json`:

```json
{
  "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dim": 384,
  "rf_probability_dim": 2,
  "final_feature_dim": 386,
  "threshold": 0.5,
  "label_mapping": {
    "0": "Not Helpful",
    "1": "Helpful"
  }
}
```

---

## 7. Nhiệm vụ từng file trong project

### 7.1. `src/preprocessing.py`

File này chịu trách nhiệm làm sạch text.

Nên có các hàm:

```python
def clean_text(text: str) -> str:
    """Làm sạch review text."""
    pass
```

Các bước xử lý đề xuất:

- Lowercase text.
- Xóa HTML tag nếu có.
- Xóa URL.
- Chuẩn hóa khoảng trắng.
- Giữ lại dấu tiếng Việt nếu dùng dữ liệu tiếng Việt.
- Không nên xóa quá mạnh vì BERT/Sentence-BERT cần ngữ cảnh tự nhiên.

Ví dụ:

```python
import re


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

### 7.2. `src/feature_pipeline.py`

File này tạo BERF feature.

Nên có class:

```python
class BERFFeaturePipeline:
    def __init__(self, embedding_model, rf_model):
        self.embedding_model = embedding_model
        self.rf_model = rf_model

    def transform(self, texts: list[str]):
        pass
```

Logic:

```python
embedding = embedding_model.encode(texts)
rf_proba = rf_model.predict_proba(embedding)
berf_feature = np.concatenate([embedding, rf_proba], axis=1)
return berf_feature
```

File này là phần cốt lõi để chứng minh demo bám paper.

---

### 7.3. `src/predictor.py`

File này load model và predict.

Nên có class:

```python
class ReviewHelpfulnessPredictor:
    def __init__(self, model_dir: str):
        pass

    def predict(self, review_text: str) -> dict:
        pass
```

Output của `predict()` nên là:

```python
{
    "label": "Helpful",
    "helpfulness_score": 0.91,
    "confidence_level": "High"
}
```

Logic confidence:

```python
def get_confidence_level(score: float) -> str:
    distance = abs(score - 0.5)
    if distance >= 0.35:
        return "High"
    elif distance >= 0.2:
        return "Medium"
    return "Low"
```

---

### 7.4. `src/gemini_service.py`

File này gọi Gemini API để sinh explanation.

Nên có class:

```python
class GeminiReviewExplainer:
    def __init__(self, api_key: str):
        pass

    def explain(self, product_info: dict, review_text: str, model_result: dict) -> dict:
        pass
```

Prompt mẫu:

```text
Bạn là trợ lý đánh giá chất lượng bình luận thương mại điện tử.

Thông tin sản phẩm:
- Tên sản phẩm: {product_name}
- Danh mục: {category}
- Giá: {price}
- Rating người dùng: {rating}

Bình luận:
{review_text}

Kết quả từ model machine learning:
- Label: {label}
- Helpfulness score: {score}

Hãy trả lời bằng tiếng Việt theo JSON:
{
  "short_assessment": "...",
  "why": ["...", "...", "..."],
  "improvement_suggestion": "...",
  "buyer_value": "..."
}

Yêu cầu:
- Không phủ định kết quả model nếu không có lý do rõ ràng.
- Không bịa thông tin ngoài bình luận.
- Tập trung đánh giá mức độ hữu ích của bình luận, không đánh giá sản phẩm tốt hay xấu.
- Nếu review quá ngắn, hãy nói rõ thiếu thông tin gì.
```

Biến môi trường cần dùng:

```text
GEMINI_API_KEY=your_api_key_here
```

---

### 7.5. `api/main.py`

File này tạo FastAPI backend.

Endpoint chính:

```http
POST /analyze-review
```

Request body:

```json
{
  "product_name": "Tai nghe Bluetooth Xiaomi Redmi Buds 6",
  "category": "electronics",
  "rating": 4,
  "price": 650000,
  "review_text": "Mình dùng được 2 tuần, pin khoảng 5-6 tiếng..."
}
```

Response:

```json
{
  "label": "Helpful",
  "helpfulness_score": 0.91,
  "confidence_level": "High",
  "gemini_explanation": {
    "short_assessment": "Review này có mức độ hữu ích cao.",
    "why": [
      "Có trải nghiệm sử dụng thực tế.",
      "Nêu thông tin cụ thể về pin và mic.",
      "Có cả ưu điểm và nhược điểm."
    ],
    "improvement_suggestion": "Có thể bổ sung ảnh thực tế hoặc độ bền sau thời gian dài sử dụng.",
    "buyer_value": "Giúp người mua hiểu rõ hơn về trải nghiệm sử dụng thực tế."
  }
}
```

Endpoint phụ nên có:

```http
GET /health
POST /rank-reviews
```

`GET /health` dùng để kiểm tra backend có chạy không.

`POST /rank-reviews` dùng để upload nhiều review và xếp hạng theo helpfulness score.

---

### 7.6. `app/streamlit_app.py`

File này tạo giao diện demo.

Nên có 3 màn hình chính:

1. **Single Review Analyzer**
2. **Batch Review Ranking**
3. **Model Evaluation Dashboard**

#### Màn hình 1: Single Review Analyzer

Input:

- Product name
- Category
- Rating
- Price
- Review text

Output:

- Helpfulness score
- Label
- Confidence level
- Gemini explanation
- Improvement suggestion

#### Màn hình 2: Batch Review Ranking

Input:

- Upload CSV chứa nhiều review.

CSV format:

```csv
product_name,category,rating,price,review_text
Tai nghe A,electronics,4,650000,"Pin tốt, mic hơi rè khi gọi ngoài đường."
Tai nghe A,electronics,5,650000,"Ok shop."
```

Output:

- Bảng review được sort theo helpfulness score.
- Top helpful reviews.
- Option gọi Gemini để tóm tắt top reviews.

#### Màn hình 3: Model Evaluation Dashboard

Hiển thị:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Class distribution trước/sau balancing
- So sánh BERT-only vs BERF nếu có

---

## 8. Luồng xử lý trong FastAPI

Pseudo-code cho endpoint `/analyze-review`:

```python
@app.post("/analyze-review")
def analyze_review(request: ReviewRequest):
    product_info = {
        "product_name": request.product_name,
        "category": request.category,
        "rating": request.rating,
        "price": request.price,
    }

    model_result = predictor.predict(request.review_text)

    if request.use_gemini:
        explanation = gemini_explainer.explain(
            product_info=product_info,
            review_text=request.review_text,
            model_result=model_result,
        )
    else:
        explanation = None

    return {
        **model_result,
        "gemini_explanation": explanation,
    }
```

---

## 9. Luồng xử lý trong Streamlit

Pseudo-code:

```python
import streamlit as st
import requests

st.title("ReviewLens AI")

product_name = st.text_input("Product name")
category = st.selectbox("Category", ["electronics", "beauty", "food", "fashion", "other"])
rating = st.slider("Rating", 1, 5, 4)
price = st.number_input("Price", min_value=0)
review_text = st.text_area("Review text")

if st.button("Analyze Review"):
    payload = {
        "product_name": product_name,
        "category": category,
        "rating": rating,
        "price": price,
        "review_text": review_text,
        "use_gemini": True,
    }

    response = requests.post("http://localhost:8000/analyze-review", json=payload)
    result = response.json()

    st.metric("Helpfulness Score", round(result["helpfulness_score"] * 100, 2))
    st.success(result["label"])
    st.write(result["gemini_explanation"])
```

---

## 10. Cài đặt môi trường

### 10.1. Tạo virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 10.2. Cài thư viện

```bash
pip install -r requirements.txt
```

Ví dụ `requirements.txt`:

```text
fastapi
uvicorn
streamlit
requests
pydantic
python-dotenv
numpy
pandas
scikit-learn
lightgbm
joblib
sentence-transformers
google-generativeai
matplotlib
plotly
```

Nếu dùng `transformers` trực tiếp thay vì `sentence-transformers`, thêm:

```text
transformers
torch
```

---

## 11. Chạy project local

### 11.1. Chạy FastAPI backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra API:

```bash
curl http://localhost:8000/health
```

### 11.2. Chạy Streamlit frontend

Mở terminal khác:

```bash
streamlit run app/streamlit_app.py
```

Truy cập:

```text
http://localhost:8501
```

---

## 12. Cấu hình Gemini API

Tạo file `.env` ở root project:

```text
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
API_BASE_URL=http://localhost:8000
```

Trong code, load bằng:

```python
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

Lưu ý:

- Không commit file `.env` lên GitHub.
- Thêm `.env` vào `.gitignore`.
- Khi deploy, cấu hình API key bằng secret/environment variable của nền tảng deploy.

---

## 13. Dockerfile đề xuất

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build image:

```bash
docker build -t reviewlens-api .
```

Run container:

```bash
docker run -p 8000:8000 --env-file .env reviewlens-api
```

Nếu muốn chạy Streamlit bằng Docker riêng, nên tạo Dockerfile khác hoặc dùng `docker-compose`.

---

## 14. Docker Compose đề xuất

Có thể tạo `docker-compose.yml`:

```yaml
version: "3.9"

services:
  api:
    build: .
    container_name: reviewlens-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./models:/app/models

  app:
    build: .
    container_name: reviewlens-streamlit
    command: streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - api
```

Chạy:

```bash
docker compose up --build
```

---

## 15. Deployment options

### Option 1: Local demo

Phù hợp khi thuyết trình trực tiếp trên máy cá nhân.

```text
FastAPI chạy localhost:8000
Streamlit chạy localhost:8501
Gemini API gọi qua internet
```

Ưu điểm:

- Dễ debug.
- Không cần cloud.
- Phù hợp demo môn học.

Nhược điểm:

- Phụ thuộc máy cá nhân.
- Nếu model nặng thì khởi động lâu.

### Option 2: Hugging Face Spaces

Phù hợp nếu muốn public demo nhanh.

Cách làm:

- Đưa Streamlit app lên Hugging Face Spaces.
- Upload model artifact nếu dung lượng cho phép.
- Cấu hình Gemini key trong Secrets.

Ưu điểm:

- Dễ share link.
- Không cần quản lý server.

Nhược điểm:

- Model nặng có thể load chậm.
- Giới hạn tài nguyên.

### Option 3: Render/Railway

Phù hợp nếu muốn deploy FastAPI riêng.

Cách làm:

- Deploy FastAPI backend.
- Deploy Streamlit frontend riêng hoặc chạy cùng service.
- Cấu hình environment variables.

Ưu điểm:

- Demo nhìn chuyên nghiệp hơn.
- Có endpoint API thật.

Nhược điểm:

- Cần cấu hình deploy nhiều hơn.

### Option 4: VPS/Cloud VM

Phù hợp nếu muốn kiểm soát toàn bộ.

Cách làm:

- Cài Docker trên VM.
- Chạy `docker compose up -d`.
- Dùng Nginx reverse proxy nếu cần domain.

Ưu điểm:

- Chủ động tài nguyên.
- Dễ mở rộng.

Nhược điểm:

- Cần biết deploy server, firewall, domain, HTTPS.

---

## 16. Demo script khi thuyết trình

Kịch bản demo 5 phút:

1. Giới thiệu vấn đề: Người mua không thể đọc hết hàng trăm review trên sàn thương mại điện tử.
2. Mở ReviewLens AI.
3. Nhập thông tin sản phẩm và một review chi tiết.
4. Hệ thống trả `Helpful`, score cao và Gemini giải thích vì sao hữu ích.
5. Nhập một review ngắn như `Shop giao nhanh, ok`.
6. Hệ thống trả `Not Helpful`, score thấp và Gemini gợi ý cần bổ sung thông tin gì.
7. Chuyển sang batch mode, upload CSV nhiều review.
8. Hệ thống ranking review theo helpfulness score.
9. Kết luận: Model ML giúp chấm điểm định lượng, Gemini giúp giải thích tự nhiên, từ đó biến nghiên cứu thành product demo có tính ứng dụng.

Câu trả lời khi bị hỏi về Gemini:

> Gemini không thay thế mô hình machine learning. Gemini chỉ nhận kết quả từ model đã train và sinh ra giải thích dễ hiểu cho người dùng. Quyết định Helpful hay Not Helpful vẫn do pipeline BERF-LGBM thực hiện.

---

## 17. Checklist hoàn thành MVP

### Model

- [ ] Đã train embedding model hoặc dùng pretrained Sentence-BERT.
- [ ] Đã train Random Forest trên embedding.
- [ ] Đã lấy `rf.predict_proba()` để tạo probability features.
- [ ] Đã train LGBM trên feature concat.
- [ ] Đã save `rf_model.pkl`.
- [ ] Đã save `lgbm_model.pkl`.
- [ ] Đã lưu model config/threshold.

### Backend

- [ ] Có endpoint `/health`.
- [ ] Có endpoint `/analyze-review`.
- [ ] Load model một lần khi app start.
- [ ] Có xử lý lỗi input rỗng.
- [ ] Có option bật/tắt Gemini.

### Frontend

- [ ] Có form nhập sản phẩm và review.
- [ ] Hiển thị helpfulness score.
- [ ] Hiển thị label.
- [ ] Hiển thị Gemini explanation.
- [ ] Có batch upload CSV nếu kịp.

### Deployment

- [ ] Có `requirements.txt`.
- [ ] Có `Dockerfile`.
- [ ] Có `.env.example`.
- [ ] Có hướng dẫn chạy local trong README.
- [ ] Không commit API key.

---

## 18. Các lỗi dễ gặp

### Lỗi 1: Feature dimension mismatch

Nguyên nhân:

- Lúc train dùng embedding 768 chiều nhưng lúc inference dùng embedding model khác ra 384 chiều.

Cách xử lý:

- Dùng đúng embedding model đã dùng khi train.
- Lưu `embedding_dim` trong `model_config.json`.

### Lỗi 2: Random Forest probability sai thứ tự class

Nguyên nhân:

- `rf_model.classes_` có thể là `[0, 1]`, nhưng cũng cần kiểm tra chắc chắn.

Cách xử lý:

- Log `rf_model.classes_`.
- Lưu label mapping.
- Khi lấy helpfulness score, đảm bảo lấy đúng index class `1`.

### Lỗi 3: Gemini trả về không đúng JSON

Nguyên nhân:

- LLM có thể trả thêm markdown hoặc text ngoài JSON.

Cách xử lý:

- Prompt yêu cầu trả JSON nghiêm ngặt.
- Trong code có fallback parse.
- Nếu parse lỗi, hiển thị raw text.

### Lỗi 4: Model load quá chậm

Nguyên nhân:

- BERT/Sentence-BERT nặng.

Cách xử lý:

- Load model global khi FastAPI start.
- Dùng model nhẹ như `paraphrase-multilingual-MiniLM-L12-v2`.
- Cache embedding cho batch mode nếu cần.

### Lỗi 5: Streamlit không gọi được API

Nguyên nhân:

- Sai URL backend.
- Backend chưa chạy.
- Docker network khác localhost.

Cách xử lý:

- Dùng biến môi trường `API_BASE_URL`.
- Local: `http://localhost:8000`.
- Docker compose: `http://api:8000`.

---

## 19. Mở rộng sau MVP

Sau khi MVP chạy ổn, có thể mở rộng:

1. Thêm batch ranking nhiều review.
2. Thêm tóm tắt top helpful reviews bằng Gemini.
3. Thêm dashboard so sánh BERT-only và BERF.
4. Thêm logging lịch sử prediction.
5. Thêm database PostgreSQL để lưu review và kết quả.
6. Thêm Redis cache cho Gemini response.
7. Thêm authentication nếu muốn thành sản phẩm thật.
8. Thêm model monitoring: phân phối score, số lượng helpful/not helpful theo thời gian.

---

## 20. Kết luận triển khai

Product demo nên được triển khai theo hướng:

```text
Streamlit UI
    ↓
FastAPI backend
    ↓
BERF-LGBM inference pipeline
    ↓
Gemini explanation layer
    ↓
User-friendly helpfulness assessment
```

Trong đó:

- **BERF-LGBM** là lõi dự đoán.
- **Gemini API** là lớp giải thích kết quả.
- **Streamlit** là giao diện demo nhanh.
- **FastAPI** là backend phục vụ inference.
- **Docker** giúp đóng gói và deploy dễ dàng.

Đây là hướng vừa bám paper, vừa có tính sản phẩm, vừa phù hợp yêu cầu đồ án môn học vì có đầy đủ: input/output rõ ràng, thuật toán, data source, evaluation, source code, hướng dẫn chạy và demo trực quan.
