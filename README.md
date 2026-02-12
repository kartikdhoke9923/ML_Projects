# Credit Card Default Prediction ML Pipeline
Run Full ML Pipeline Online (No Installation Required)

https://colab.research.google.com/github/kartikdhoke9923/ML_Projects/blob/main/notebook/EDA_CREDIT_DEFAUL.ipynb

End-to-end Machine Learning pipeline to predict whether a customer will default on credit card payments.  
This project covers the complete ML lifecycle including data ingestion, preprocessing, model training, evaluation, and deployment through a web application.

---

## Problem Statement

Financial institutions need to identify customers who are likely to default on credit card payments.  
Early prediction helps reduce financial risk and improve decision-making.

This project builds a machine learning system that predicts default risk using historical customer data.

---

## Project Structure

ML_Projects/
│
├── artifacts/                   # Saved models and processed data
│
├── notebook/                    # EDA and online runnable notebook
│   └── EDA_CREDIT_DEFAUL.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── train_pipeline.py    # Main training entry point
│
├── app.py                       # Streamlit prediction UI
├── requirements.txt
└── README.md

---

## Machine Learning Pipeline Flow

Raw Dataset  
↓  
Data Ingestion  
↓  
Data Cleaning & Transformation  
↓  
Feature Engineering  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Saved Model (artifacts/model.pkl)  
↓  
Streamlit Prediction Web App  

---

## Model Performance

Best Model: (Update after training)

Accuracy: XX %  
Precision: XX %  
Recall: XX %  
F1 Score: XX  

---

## Run Project Online (No Installation Required)

Run the full ML pipeline directly in Google Colab:

https://colab.research.google.com/github/kartikdhoke9923/ML_Projects/blob/main/notebook/EDA_CREDIT_DEFAUL.ipynb

Steps:
1. Open link
2. Click Runtime → Run All
3. Pipeline executes automatically

---

## Run Project Locally

### Clone repository
git clone https://github.com/kartikdhoke9923/ML_Projects.git  
cd ML_Projects

### Create virtual environment
python -m venv venv  
venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Run training pipeline
python src/pipeline/train_pipeline.py

### Run prediction web app
streamlit run app.py

---

## Live Web App Demo

(Add after Streamlit deployment)

https://your-streamlit-app-link.streamlit.app

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning Pipelines
- Streamlit
- Model Serialization (Pickle)

---

## Key Features

- Modular ML pipeline architecture
- Automated data preprocessing
- Model training and evaluation
- Online runnable notebook (Colab)
- Interactive prediction web interface
- Reproducible environment setup

---

## Author

Kartik Dhoke  
Machine Learning & Data Analytics Enthusiast
