# 📊 Customer Churn Analysis - EDA & Predictive Modeling

## 📌 Project Overview
Customer churn is a major issue for subscription-based businesses, especially in the telecom industry. This project performs **Exploratory Data Analysis (EDA)** and applies **Machine Learning models** to identify key factors that influence customer churn and predict which customers are at risk.

---

## 📂 Dataset
- The dataset contains customer demographics, account information, and service subscription details.  
- Target variable: **`Churn`** (Yes/No).

**Key Features:**
- Demographics: gender, senior citizen, partner, dependents  
- Account details: tenure, contract type, payment method, charges  
- Services: phone, internet, streaming, multiple lines, etc.  

---

## ⚙️ Project Workflow

1. **Data Loading & Cleaning**
   - Load dataset
   - Handle missing values & inconsistencies  

2. **Exploratory Data Analysis (EDA)**
   - Distribution of churn vs non-churn customers  
   - Relationship of churn with tenure, contract type, monthly charges, etc.  
   - Visualizations: bar charts, histograms, correlation heatmaps  

3. **Feature Engineering & Preprocessing**
   - Encode categorical variables  
   - Scale numerical features  
   - Train-test split  

4. **Modeling**
   - Logistic Regression, Random Forest, Decision Tree, etc.  
   - Evaluate with accuracy, precision, recall, F1-score, ROC-AUC  

5. **Results & Insights**
   - Identify key factors leading to churn  
   - Generate actionable recommendations for businesses  

---

## 📊 Key Insights
- Customers with **month-to-month contracts** are far more likely to churn.  
- **Higher monthly charges** increase churn probability.  
- **Longer tenure** reduces churn — loyal customers tend to stay.  
- Customers using **automatic payment methods** are less likely to leave.  

---
## 🛠️ Installation & Usage

### Requirements
Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

