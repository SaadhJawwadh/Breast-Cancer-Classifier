import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets

# If executed directly with `python streamlit_app.py`, spawn `streamlit run`
if __name__ == "__main__" and os.environ.get("LAUNCHED_VIA_PYTHON") != "1":
    import sys
    import subprocess
    os.environ["LAUNCHED_VIA_PYTHON"] = "1"
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
    ])
    raise SystemExit(0)

# Paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model_logreg.joblib"


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_dataset():
    ds = datasets.load_breast_cancer()
    X = pd.DataFrame(ds.data, columns=ds.feature_names)
    y = pd.Series(ds.target, name="target")
    return ds, X, y


def ensure_feature_order(
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Extra columns are ignored in strict order
    return df.loc[:, feature_names]


st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="centered",
)
st.title("🩺 Breast Cancer Classifier (scikit-learn)")
st.write(
    "This app loads the tuned model from the notebook and runs predictions "
    "on the\nBreast Cancer Wisconsin (Diagnostic) dataset. Provide inputs "
    "manually, pick a dataset sample,\n or upload a CSV with the exact "
    "sklearn feature columns."
)

model = load_model(MODEL_PATH)
_, X_full, y_full = load_dataset()
feature_names = list(X_full.columns)

if model is None:
    st.error(
        "Model not found at 'best_model_logreg.joblib'.\n\n"
        "Please open and run the notebook at "
        "'notebooks/classification_workflow.ipynb' to generate it."
    )
    st.stop()

with st.sidebar:
    st.header("Input mode")
    mode = st.radio(
        "Choose input mode",
        ["Pick sample", "Manual input", "Upload CSV"],
        index=0,
    )

    st.markdown("---")
    st.caption("Model: best_model_logreg.joblib")

# Helper to predict and display results
LABELS = {0: "malignant", 1: "benign"}


def predict_df(df: pd.DataFrame):
    df = ensure_feature_order(df, feature_names)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:, 1]
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(df)
    preds = (
        (proba >= 0.5).astype(int)
        if proba is not None and proba.ndim == 1
        else model.predict(df)
    )
    return preds, proba


if mode == "Pick sample":
    st.subheader("Pick a sample from sklearn dataset")
    idx = st.slider(
        "Sample index",
        min_value=0,
        max_value=len(X_full) - 1,
        value=0,
        step=1,
    )
    x = X_full.iloc[[idx]]
    st.dataframe(x.T.rename(columns={x.index[0]: "value"}))
    if st.button("Predict", type="primary"):
        preds, proba = predict_df(x)
        st.success(
            f"Prediction: {LABELS[int(preds[0])]}  •  P(benign)= "
            f"{float(proba[0]):.3f}"
        )
        st.caption(f"True label: {LABELS[int(y_full.iloc[idx])]} (dataset)")

elif mode == "Manual input":
    st.subheader("Enter feature values")
    cols = st.columns(3)
    defaults = X_full.mean()
    inputs = {}
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(
                feat,
                value=float(defaults[feat]),
                format="%.5f",
            )
            inputs[feat] = val
    x = pd.DataFrame([inputs])
    if st.button("Predict", type="primary"):
        preds, proba = predict_df(x)
        st.success(
            f"Prediction: {LABELS[int(preds[0])]}  •  P(benign)= "
            f"{float(proba[0]):.3f}"
        )
        st.dataframe(x.T.rename(columns={0: "value"}))

else:  # Upload CSV
    st.subheader("Upload a CSV with exact feature columns")
    st.caption("Expected columns (case-sensitive):")
    st.code(", ".join(feature_names), language="text")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            # Inform about extra columns (will be ignored)
            extra_cols = [c for c in df.columns if c not in feature_names]
            if extra_cols:
                st.info(f"Ignoring extra columns: {extra_cols}")

            # Enforce required column order and coerce to numeric
            df_ordered = ensure_feature_order(df, feature_names)
            df_ordered = df_ordered.apply(pd.to_numeric, errors="coerce")
            if df_ordered.isna().any().any():
                bad_cols = df_ordered.columns[df_ordered.isna().any()].tolist()
                st.warning(
                    "Some values could not be parsed as numeric and "
                    "became NaN. "
                    f"Check columns: {bad_cols}"
                )

            st.write("Preview:")
            st.dataframe(df_ordered.head())
            if st.button("Predict", type="primary"):
                # Drop rows with any NaNs before prediction
                clean = df_ordered.dropna()
                if clean.empty:
                    st.error("No valid rows to predict after removing NaNs.")
                else:
                    preds, proba = predict_df(clean)
                    out = clean.copy()
                    out["pred"] = preds
                    if proba is not None:
                        out["p_benign"] = proba
                    st.write("Results (first 50 rows):")
                    st.dataframe(out.head(50))
                    st.download_button(
                        "Download predictions CSV",
                        data=out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption(
    "Built with scikit-learn pipelines. Feature scaling is handled "
    "inside the saved model."
)
