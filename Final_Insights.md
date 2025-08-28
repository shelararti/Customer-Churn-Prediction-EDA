## 1. Dataset Distribution
- Churn is **imbalanced** → majority customers stayed (~5100), fewer churned (~1800).
- Accuracy alone is misleading → use **Recall, F1, ROC-AUC**.

---

## 2. Correlation Analysis
- **Tenure vs Churn:** Longer tenure → less churn.
- **MonthlyCharges vs Churn:** Higher charges → slightly more churn.
- **TotalCharges vs Churn:** Low total charges → higher churn (newer customers).
- **Tenure & TotalCharges:** Highly correlated → avoid both in linear models.

---

## 3. Model Performance
- Logistic Regression, Random Forest **ROC-AUC ≈ 0.83**.
- **Logistic Regression** = best for **explainability**.
- **Random Forest** = best for **operational predictions**.

---

## 4. Feature Importance
Top churn drivers:
1. **Tenure & TotalCharges** – short tenure & low total charges → high churn.
2. **MonthlyCharges** – higher bills → more churn.
3. **Contract Type** – yearly/two-year contracts reduce churn.
4. **InternetService (Fiber optic)** – more churn-prone.
5. **PaymentMethod (Electronic check)** – increases churn.
6. **Add-on Services (Tech Support, Security, Backup)** – reduce churn.
7. **Demographics** – Dependents/Partners reduce churn; Senior citizens slightly higher churn.

---

## 6. Business Recommendations
- Focus on **new customers** (low tenure).
- Promote **annual/two-year contracts**.
- Offer discounts/bundles for **high-bill customers**.
- Investigate **fiber optic dissatisfaction**.
- Encourage **auto-pay / card payments** instead of electronic check.
- Cross-sell **value-added services** (Tech Support, Security, Backup).
- Create tailored offers for **senior citizens**.

---
✅ **Summary:**  
Use **Logistic Regression** for insights and business communication.  
Use **Random Forest** for churn prediction and customer targeting.  
Together, they give both *why customers churn* and *who will churn*.

---