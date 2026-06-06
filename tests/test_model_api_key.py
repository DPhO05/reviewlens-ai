import os
from google import genai

def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Bạn chưa set GEMINI_API_KEY.\n"
            "Mac/Linux: export GEMINI_API_KEY='your_api_key'\n"
            "Windows PowerShell: $env:GEMINI_API_KEY='your_api_key'"
        )

    client = genai.Client(api_key=api_key)

    print("===== DANH SÁCH MODEL GEMINI API KEY CÓ THỂ THẤY =====\n")

    models = client.models.list()

    callable_models = []

    for model in models:
        name = model.name
        display_name = getattr(model, "display_name", "")
        supported_actions = getattr(model, "supported_actions", [])

        print(f"Model name: {name}")
        print(f"Display name: {display_name}")
        print(f"Supported actions: {supported_actions}")
        print("-" * 60)

        if "generateContent" in supported_actions:
            callable_models.append(name)

    print("\n===== MODEL CÓ THỂ CALL generateContent =====")
    for m in callable_models:
        print(m)


if __name__ == "__main__":
    main()
