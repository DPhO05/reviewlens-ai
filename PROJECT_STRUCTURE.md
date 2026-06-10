# Project Structure

```text
api/                 FastAPI backend.
app/                 Streamlit frontend.
src/                 Core application code: predictor, feature pipeline, Gemini service.
tests/               Unit and integration tests.
models/              Local model artifacts and Hugging Face artifact cache.
Data/                Source data, gold data, labeling outputs, EDA outputs.
notebooks/           Kaggle/local notebooks.
scripts/
  analysis/          EDA and reporting scripts.
  data/              Data transformation scripts.
  kaggle/            Notebook generator scripts.
  labeling/          LLM-assisted labeling scripts.
  model/             Model utility scripts.
artifacts/           Temporary/generated development artifacts.
```

## Common Commands

```bash
python3 -m pytest -q
python3 scripts/model/test_hf_config.py
python3 scripts/kaggle/create_kaggle_gemini_labeling_notebook.py
python3 scripts/kaggle/create_kaggle_gold_notebook.py
python3 scripts/analysis/analyze_gold_data.py
```

## Main Data Files

```text
Data/batches/                         Raw/manual annotation batches.
Data/gold_data/data_labeling.csv      Dataset prepared for another LLM labeler.
Data/gold_data/data_gold.csv          Gold training dataset.
notebooks/kaggle_gemini_helpfulness_labeling.ipynb
notebooks/notebook_data_gold_phobert_berf.ipynb
```
