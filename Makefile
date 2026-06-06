.PHONY: models api app test

models:
	python3 scripts/train_demo_models.py

api: models
	python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	python3 -m streamlit run app/streamlit_app.py

test: models
	python3 -m pytest -q
