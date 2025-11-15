# Breast Cancer Classifier – Streamlit App

This repository contains a complete workflow for a binary classification task using the Breast Cancer Wisconsin (Diagnostic) dataset and a simple Streamlit app for inference.

## Contents
- `notebooks/classification_workflow.ipynb` – end-to-end data science workflow (EDA, CV, tuning, evaluation, artifacts)
- `models/best_model_logreg.joblib` – saved best model pipeline (created by the notebook)
- `artifacts/test_metrics.json` – test-set metrics
- `streamlit_app.py` – Streamlit app to run predictions
- `requirements.txt` – dependencies for the app and notebook

## Quick start (macOS, zsh)

```bash
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If you haven't created the model yet, open the notebook and run it end-to-end
# It will generate models/best_model_logreg.joblib
code "notebooks/classification_workflow.ipynb"

# Launch the Streamlit app
streamlit run streamlit_app.py
```

Once running, the app offers three input modes:
- Pick a sample from the built-in sklearn dataset
- Manual input of all features
- Upload a CSV with exact feature columns (shown within the app)

## Notes
- The model is a scikit-learn pipeline with StandardScaler + LogisticRegression (selected via cross-validation and tuned with GridSearchCV).
- Feature order is critical; the app ensures that uploaded CSVs are mapped to the expected order.
- To re-train or try other models, modify and re-run the notebook.
