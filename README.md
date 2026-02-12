# 🏦 Credit Risk Prediction System (End-to-End ML Pipeline)
To run the project first run eda file anf then run data ingestion file only ...... and every thing will run automatically with app ui
An end-to-end Machine Learning project for predicting credit risk using customer financial and behavioral data.

This project follows a **production-style modular ML architecture** including:

- Data Ingestion
- Data Transformation
- Model Training & Selection
- Model Evaluation
- Prediction Pipeline
- Streamlit Web App Deployment

The best performing model is automatically selected based on **ROC-AUC score** and saved for inference.

---

## 🚀 Project Overview

The goal of this project is to predict whether a customer is likely to default on credit using historical application data.

The pipeline:

1. Cleaned dataset (after EDA)
2. Train-test split
3. Feature preprocessing (scaling + encoding)
4. Train multiple ML models
5. Select best model automatically
6. Save model + preprocessor
7. Deploy prediction UI using Streamlit

---

## 🧠 Models Evaluated

The system trains and evaluates multiple models automatically:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM (Best Model)

Best model is selected based on:

roc and recall 
and best was LIGHTGBM
