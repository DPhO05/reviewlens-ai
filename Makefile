.PHONY: api app test demo-models

demo-models:
	python3 scripts/model/train_demo_models.py

api:
	python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	python3 -m streamlit run app/streamlit_app.py

test:
	python3 -m pytest -q
