# Scripts

```text
analysis/   Build EDA outputs for gold data.
data/       Convert labeling/gold CSV files.
kaggle/     Generate Kaggle notebooks.
labeling/   Gemini/local LLM-assisted labeling workflows.
model/      Model artifact and Hugging Face utility commands.
```

Common commands:

```bash
python3 scripts/model/test_hf_config.py
python3 scripts/kaggle/create_kaggle_gemini_labeling_notebook.py
python3 scripts/kaggle/create_kaggle_gold_notebook.py
python3 scripts/analysis/analyze_gold_data.py
```
