import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from model.common import get_models

st.set_page_config(page_title="ML Classification Lab", page_icon="🤖", layout="wide")

FEATURES = list(load_breast_cancer().feature_names)
DISPLAY_NAMES = {
    0: "Benign",
    1: "Malignant"
}

@st.cache_data
def load_reference_data():
    ds = load_breast_cancer(as_frame=True)
    X = ds.data.copy()
    y = (ds.target == 0).astype(int)  # malignant = 1
    return X, y

@st.cache_resource
def train_all_models():
    X, y = load_reference_data()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    fitted = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted

def normalize_uploaded_target(series):
    # Accept either 0/1 or common text labels.
    if pd.api.types.is_numeric_dtype(series):
        vals = set(pd.Series(series).dropna().unique())
        if vals.issubset({0, 1}):
            return series.astype(int)
        if vals.issubset({2, 4}):
            return series.map({2: 0, 4: 1}).astype(int)
    mapping = {
        "benign": 0, "b": 0, "0": 0,
        "malignant": 1, "m": 1, "1": 1
    }
    return series.astype(str).str.strip().str.lower().map(mapping)

def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred

st.title("🤖 Machine Learning Classification Dashboard")
st.caption("M.Tech (AI/ML) • Assignment 2 • Jivitesh Kumar Choudhary • BITSID: 2025AC05786")

with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox(
        "Select classification model",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "SVM"]
    )
    uploaded = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="Use the supplied test_data.csv or another CSV with the same 30 feature columns and a target column."
    )
    st.markdown("**Target convention:** 0 = Benign, 1 = Malignant")

X_ref, y_ref = load_reference_data()
fitted_models = train_all_models()

if uploaded is None:
    test_df = pd.read_csv("test_data.csv")
    source_label = "Bundled test_data.csv"
else:
    test_df = pd.read_csv(uploaded)
    source_label = uploaded.name

if "target" not in test_df.columns:
    st.error("The uploaded CSV must contain a 'target' column for evaluation.")
    st.stop()

missing = [c for c in FEATURES if c not in test_df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X_eval = test_df[FEATURES].apply(pd.to_numeric, errors="coerce")
if X_eval.isna().any().any():
    st.error("Feature columns must contain numeric values without missing/non-numeric entries.")
    st.stop()

y_eval = normalize_uploaded_target(test_df["target"])
if y_eval.isna().any():
    st.error("Target values must be 0/1, 2/4, or labels Benign/Malignant (or B/M).")
    st.stop()
y_eval = y_eval.astype(int)

st.subheader("Test Data")
st.write(f"**Source:** {source_label}  |  **Rows:** {len(test_df)}")
st.dataframe(test_df.head(10), use_container_width=True)

model = fitted_models[selected_model]
metric_values, pred = evaluate(model, X_eval, y_eval)

st.subheader(f"Evaluation — {selected_model}")
c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)
for col, label in zip(
    [c1,c2,c3,c4,c5,c6],
    ["Accuracy","AUC","Precision","Recall","F1","MCC"]
):
    col.metric(label, f"{metric_values[label]:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_eval, pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"]
    )
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_eval, pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(4), use_container_width=True)

st.subheader("All-model comparison on the uploaded test data")
comparison = []
for name, fitted in fitted_models.items():
    vals, _ = evaluate(fitted, X_eval, y_eval)
    comparison.append({"ML Model": name, **vals})
st.dataframe(pd.DataFrame(comparison).set_index("ML Model").round(4), use_container_width=True)

st.info(
    "The reference models are trained on the UCI Breast Cancer Wisconsin (Diagnostic) "
    "dataset available through scikit-learn's bundled copy. The uploaded CSV is used only as test data."
)
