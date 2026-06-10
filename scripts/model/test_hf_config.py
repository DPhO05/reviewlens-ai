from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

DEFAULT_REPO_ID = "DPhO05/tiki-review-helpfulness-phobert-berf-gold-v2"
REQUIRED_FILES = {
    "config.json",
    "final_model/final_ml_model.joblib",
    "final_model/metadata_scaler.joblib",
    "final_model/rf_embedding_probability_model.joblib",
    "final_model/annotation_feature_predictor.joblib",
}


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}
    except requests.JSONDecodeError:
        return {"raw": response.text[:500]}


def check_identity(token: str) -> bool:
    response = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers=auth_headers(token),
        timeout=20,
    )
    if response.status_code != 200:
        error = safe_json(response).get("error", "Không xác định")
        print(f"[FAIL] HF_TOKEN không hợp lệ: HTTP {response.status_code} - {error}")
        return False

    payload = safe_json(response)
    print(f"[PASS] Token hợp lệ cho tài khoản: {payload.get('name', 'unknown')}")
    print(f"       Loại token: {payload.get('auth', {}).get('type', 'unknown')}")
    return True


def check_model_repo(token: str, repo_id: str) -> bool:
    response = requests.get(
        f"https://huggingface.co/api/models/{repo_id}",
        headers=auth_headers(token),
        timeout=20,
    )
    if response.status_code != 200:
        error = safe_json(response).get("error", "Không xác định")
        print(
            f"[FAIL] Không truy cập được model {repo_id}: "
            f"HTTP {response.status_code} - {error}"
        )
        return False

    payload = safe_json(response)
    files = {item["rfilename"] for item in payload.get("siblings", [])}
    missing = sorted(REQUIRED_FILES - files)

    print(f"[PASS] Truy cập được model repo: {payload.get('modelId', repo_id)}")
    print(f"       Private: {payload.get('private', False)}")
    print(f"       Pipeline tag: {payload.get('pipeline_tag', 'not configured')}")
    print(f"       Revision: {payload.get('sha', 'unknown')}")
    if missing:
        print("[FAIL] Thiếu artifact bắt buộc:")
        for filename in missing:
            print(f"       - {filename}")
        return False

    print("[PASS] Đã tìm thấy đầy đủ RF, scaler và final model artifacts")
    provider_mapping = payload.get("inferenceProviderMapping", {})
    if provider_mapping:
        print(f"[PASS] Inference providers: {', '.join(provider_mapping)}")
    else:
        print("[INFO] Repo chưa có serverless inference provider")
    return True


def check_endpoint(token: str, endpoint_url: str) -> bool:
    payload = {
        "inputs": {
            "review_text": (
                "Mình dùng tai nghe được 2 tuần, pin khoảng 6 giờ nhưng mic "
                "hơi rè khi gọi ngoài đường."
            ),
            "rating": 4,
            "verified_purchase": True,
        }
    }
    response = requests.post(
        endpoint_url,
        headers={**auth_headers(token), "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    body = safe_json(response)
    if response.status_code >= 400:
        message = body.get("error") or body.get("detail") or body.get("raw")
        print(
            f"[FAIL] Endpoint trả HTTP {response.status_code}: "
            f"{str(message)[:300]}"
        )
        return False

    print(f"[PASS] Endpoint hoạt động: HTTP {response.status_code}")
    print("       Response:")
    print(json.dumps(body, ensure_ascii=False, indent=2)[:1500])
    return True


def main() -> int:
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = os.getenv("HF_MODEL_ID", DEFAULT_REPO_ID).strip()
    endpoint_url = os.getenv("HF_ENDPOINT_URL", "").strip()

    print("=== Hugging Face configuration check ===")
    print(f"Model repo: {repo_id}")
    print(f"HF_TOKEN: {'configured' if token else 'missing'}")
    print(f"HF_ENDPOINT_URL: {'configured' if endpoint_url else 'not configured'}")

    if not token:
        print("[FAIL] Hãy cấu hình HF_TOKEN trong file .env")
        return 1

    try:
        identity_ok = check_identity(token)
        repo_ok = check_model_repo(token, repo_id) if identity_ok else False
        endpoint_ok = check_endpoint(token, endpoint_url) if endpoint_url else True
        if not endpoint_url:
            print(
                "[SKIP] Chưa kiểm tra inference vì HF_ENDPOINT_URL đang để trống. "
                "Model repo không tự động là một inference endpoint."
            )
        return 0 if identity_ok and repo_ok and endpoint_ok else 1
    except requests.Timeout:
        print("[FAIL] Hết thời gian chờ khi kết nối Hugging Face")
        return 1
    except requests.RequestException as exc:
        print(f"[FAIL] Lỗi kết nối Hugging Face: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
