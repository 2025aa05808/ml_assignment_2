import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 – Classification App",
    layout="wide"
)

st.title("📊 ML Assignment 2 – Classification Models")
st.write(
    "This Streamlit application demonstrates multiple machine learning "
    "classification models trained on a UCI dataset."
)

# --------------------------------------------------
# Download Test Data
# --------------------------------------------------
st.subheader("⬇ Download Sample Test Dataset")

with open("data/test_data.csv", "rb") as f:
    st.download_button(
        label="Download Test Data (CSV)",
        data=f,
        file_name="test_data.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Load Models and Scaler
# --------------------------------------------------
MODEL_PATH = "model/"

models = {
    "Logistic Regression": joblib.load(MODEL_PATH + "logistic_regression.pkl"),
    "Decision Tree": joblib.load(MODEL_PATH + "decision_tree.pkl"),
    "KNN": joblib.load(MODEL_PATH + "knn.pkl"),
    "Naive Bayes": joblib.load(MODEL_PATH + "naive_bayes.pkl"),
    "Random Forest": joblib.load(MODEL_PATH + "random_forest.pkl"),
    "XGBoost": joblib.load(MODEL_PATH + "xgboost.pkl"),
}

scaler = joblib.load(MODEL_PATH + "scaler.pkl")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙ Model Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# --------------------------------------------------
# Main Application Logic
# --------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "diagnosis" not in df.columns:
        st.error("❌ Uploaded CSV must contain a 'diagnosis' column.")
        st.stop()

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    model = models[selected_model_name]

    # Scale if required
    if selected_model_name in ["Logistic Regression", "KNN"]:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    metrics = evaluate_model(y, y_pred, y_prob)

    # --------------------------------------------------
    # Display Metrics
    # --------------------------------------------------
    st.subheader(f"📈 Evaluation Metrics – {selected_model_name}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", round(metrics["Accuracy"], 4))
    c2.metric("AUC Score", round(metrics["AUC"], 4))
    c3.metric("Precision", round(metrics["Precision"], 4))

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", round(metrics["Recall"], 4))
    c5.metric("F1 Score", round(metrics["F1 Score"], 4))
    c6.metric("MCC", round(metrics["MCC"], 4))

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.subheader("📄 Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report
