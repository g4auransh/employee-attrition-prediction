# 📊 Employee Attrition Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

An end-to-end Machine Learning solution to predict the probability of employee turnover. This project uses **Random Forest** classification and **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance, all served through an interactive **Streamlit** web application.

---

## 🎯 Project Overview
Employee attrition costs companies billions in hiring and training. This project provides a data-driven approach to:
1. Identify high-risk employees before they leave.
2. Understand the key drivers (Overtime, Income, etc.) behind turnover.
3. Provide HR managers with an interactive simulator to test "What-If" scenarios.

## 🏗️ Project Structure
```text
employee-attrition/
├── data/               # Raw and processed datasets
├── models/             # Saved .pkl models and evaluation plots
├── src/                # Modular Python scripts
│   ├── preprocess.py   # Data cleaning & pipeline logic
│   ├── train.py        # Model training & SMOTE integration
│   └── evaluate.py     # Metrics & feature importance mapping
├── app.py              # Streamlit Web Application
├── requirements.txt    # Project dependencies
└── README.md           # Documentation