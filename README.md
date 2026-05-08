# 🏥 Hospital Readmission Risk Prediction

End-to-end ML system for predicting 30-day hospital readmission using clinical data.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-CatBoost%20%7C%20XGBoost%20%7C%20LightGBM-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🎯 Problem Statement
Hospital readmissions cost the US healthcare system over **$26 billion annually**.
This project builds a binary classifier to identify high-risk diabetic patients
before discharge, enabling targeted follow-up interventions.

## 📊 Dataset
- **Source:** UCI Diabetic Patient Readmission Dataset
- **Size:** 101,766 records, 50 features
- **Target:** 30-day readmission (binary)
- **Class imbalance:** 8:1 (handled with SMOTE)

## 🔬 Methodology
| Step | Detail |
|------|--------|
| Data Cleaning | Missing value imputation, group split by patient ID |
| Class Balancing | SMOTE (inside each CV fold — no leakage) |
| Models | Logistic Regression, Random Forest, XGBoost, LightGBM, **CatBoost** |
| Validation | 5-Fold Stratified Cross-Validation |
| Explainability | SHAP values |
| Business Framing | Threshold analysis + deployment plan |

## 📈 Results
| Model | AUC-ROC | PR-AUC |
|-------|---------|--------|
| **CatBoost** | **0.677** | **0.231** |
| XGBoost | 0.601 | 0.163 |
| LightGBM | 0.597 | 0.158 |
| Random Forest | 0.597 | 0.156 |
| Logistic Regression | 0.573 | 0.162 |
| Majority Baseline | 0.500 | 0.117 |

> AUC ~0.677 is consistent with published literature on this dataset.
> Signal is inherently weak — model is designed for screening, not diagnosis.

## 🔍 Key Findings
- **Medication changes** (insulin, metformin) = strongest predictors (SHAP)
- **Discharge disposition** strongly signals readmission risk
- **Prior inpatient visits** = most important clinical history feature
- At threshold=0.30: catch **36% of readmissions**, flag only **24% of patients**

## 🚀 Deployment
Streamlit app for interactive risk scoring:
```bash
pip install streamlit catboost
streamlit run streamlit_app.py
```

## ⚠️ Limitations
- 40% sample used due to compute constraints
- LabelEncoder used for initial models; CatBoost handles categoricals natively
- Model requires clinical validation before production use

## 🛠️ Tech Stack
`Python` `CatBoost` `XGBoost` `LightGBM` `SHAP` `Streamlit` `Scikit-learn` `SMOTE`
