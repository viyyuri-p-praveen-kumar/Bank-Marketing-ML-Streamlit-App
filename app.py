import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Subscription Prediction App")

st.markdown("""
This application predicts whether a client will subscribe to a term deposit 
using multiple Machine Learning models.
""")

# Download sample dataset button
with open("test_data.csv", "rb") as file:
    st.download_button(
        label="📥 Download Sample Test Dataset",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )

# Load scaler
scaler = joblib.load("model_files/scaler.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("model_files/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model_files/decision_tree.pkl"),
    "KNN": joblib.load("model_files/knn.pkl"),
    "Naive Bayes": joblib.load("model_files/naive_bayes.pkl"),
    "Random Forest": joblib.load("model_files/random_forest.pkl"),
    "XGBoost": joblib.load("model_files/xgboost.pkl")
}

st.sidebar.header("Options")

data_option = st.sidebar.radio(
    "Choose Data Source:",
    ("Upload Your Own CSV", "Use Built-in Test Dataset"),
    index=0
)


# Load dataset
if data_option == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=';')
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()
else:
    data = pd.read_csv("test_data.csv", sep=';')
    st.success("Using built-in test dataset.")


st.subheader("Data Preview")
st.dataframe(data.head())

# Ensure target exists
if 'y' not in data.columns:
    st.error("Dataset must contain target column 'y'")
    st.stop()

# Clean and convert target column safely
data['y'] = data['y'].astype(str).str.strip().str.lower()

data['y'] = data['y'].replace({'yes': 1, 'no': 0})

# Drop rows where conversion failed
data = data.dropna(subset=['y'])

y_true = data['y'].astype(int)
X = data.drop('y', axis=1)


# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Load saved feature column names
feature_columns = joblib.load("model_files/feature_columns.pkl")

X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)


# Scale
X_scaled = scaler.transform(X_encoded)

# Model selection with default None
model_options = ["None"] + list(models.keys())

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    model_options,
    index=0  # Default = None
)

if selected_model_name != "None":
    selected_model = models[selected_model_name]

    
    # Predict
    y_pred = selected_model.predict(X_scaled)

    # For AUC, we need probability scores
    y_prob = selected_model.predict_proba(X_scaled)[:, 1]

    st.subheader("📈 Model Evaluation Results")

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")


    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(3, 2.5))  # Smaller figure
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis='both', labelsize=8)

    st.pyplot(fig, use_container_width=False)


    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df)



    st.success("Evaluation completed successfully!")

else:
    st.info("Please select a model from the dropdown to evaluate.")

