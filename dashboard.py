import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="📊 Customer Churn Dashboard", layout="wide")

# -----------------------
# Load Models & Preprocessor
# -----------------------
@st.cache_resource
def load_models():
    log_reg = joblib.load("logistic_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return log_reg, rf_model, preprocessor

log_reg, rf_model, preprocessor = load_models()

# -----------------------
# Header
# -----------------------
st.title("📊 Customer Churn Analysis Dashboard")
st.markdown(
    """
    This dashboard presents **EDA insights**, **model performance**,  
    and a **prediction demo** using trained Logistic Regression & Random Forest models.
    """
)

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")  # replace with your dataset file
    return df

df = load_data()

# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Overview", "EDA", "Model Performance", "Feature Importance", "Recommendations", "Prediction Demo"]
)

# -----------------------
# Overview
# -----------------------
if section == "Overview":
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write(df.head())

    st.subheader("Churn Distribution")
    churn_counts = df["Churn"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

    st.markdown(
        """
        🔎 **Key Insight:** The dataset is imbalanced → majority customers stayed, fewer churned.  
        Accuracy alone is misleading → focus on Recall, F1, and ROC-AUC.
        """
    )

# -----------------------
# EDA
# -----------------------
elif section == "EDA":
    st.subheader("Exploratory Data Analysis")

    # Tenure vs Churn
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, ax=ax)
    st.pyplot(fig)

    # Monthly Charges vs Churn
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", multiple="stack", bins=30, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------
# Model Performance
# -----------------------
elif section == "Model Performance":
    st.subheader("Model Performance (Your Results)")
    st.markdown(
        """
        - **Logistic Regression** → ROC-AUC ≈ 0.83 (best for explainability)  
        - **Random Forest** → ROC-AUC ≈ 0.83 (best for operational predictions)  
        """
    )

    # Example metrics table
    metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [0.80, 0.82],
        "Recall": [0.72, 0.74],
        "F1-Score": [0.75, 0.77],
        "ROC-AUC": [0.83, 0.83]
    })
    st.table(metrics)

# -----------------------
# Feature Importance
# -----------------------
elif section == "Feature Importance":
    st.subheader("Feature Importance")
    importance = {
        "Tenure & TotalCharges": 0.25,
        "MonthlyCharges": 0.20,
        "Contract Type": 0.15,
        "InternetService (Fiber Optic)": 0.12,
        "PaymentMethod (Electronic Check)": 0.10,
        "Add-on Services": 0.10,
        "Demographics": 0.08,
    }
    feat_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
    
    fig, ax = plt.subplots()
    sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax, palette="viridis")
    st.pyplot(fig)

# -----------------------
# Recommendations
# -----------------------
elif section == "Recommendations":
    st.subheader("Business Recommendations")
    st.markdown(
        """
        ✅ **Focus on new customers** (low tenure)  
        ✅ Promote **annual/two-year contracts**  
        ✅ Offer discounts for **high-bill customers**  
        ✅ Investigate **fiber optic dissatisfaction**  
        ✅ Encourage **auto-pay/card payments**  
        ✅ Cross-sell **Tech Support, Security, Backup**  
        ✅ Tailored offers for **senior citizens**  
        """
    )

# -----------------------
# Prediction Demo
# -----------------------
elif section == "Prediction Demo":
    st.subheader("Try a Prediction")
    st.write("Enter customer details to predict churn:")

    # Collect user input
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("Predict"):
        # Convert input to dataframe
        input_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract": contract,
            "PaymentMethod": payment,
            "InternetService": internet
        }])

        # Preprocess
        X_processed = preprocessor.transform(input_data)

        # Predictions
        log_prob = log_reg.predict_proba(X_processed)[0][1]
        rf_prob = rf_model.predict_proba(X_processed)[0][1]

        st.metric("Logistic Regression Churn Probability", f"{log_prob*100:.1f}%")
        st.metric("Random Forest Churn Probability", f"{rf_prob*100:.1f}%")
