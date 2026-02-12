import streamlit as st
import pickle
from src.pipeline.predict_pipeline import predict_Target

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

# Load preprocessor once
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

raw_columns = preprocessor.feature_names_in_

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("🏦 Credit Risk Prediction")

st.write("Enter applicant details. Missing fields will be auto-filled.")

input_data = {}

# ---- Dynamic Input Fields ----
for col in raw_columns:
    input_data[col] = st.number_input(
        label=col,
        value=0.0
    )

# ---- Predict Button ----
if st.button("Predict Credit Risk"):
    try:
        prediction = predict_Target(
            input_dict=input_data,
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH
        )

        st.success(f"✅ Prediction Output: {prediction}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
