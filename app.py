import streamlit as st
from src.pipeline.predict_pipeline import predict_Target

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("🏦 Credit Risk Prediction")

st.write("Enter applicant details")

# ---------------- BASIC INPUTS ----------------

age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", value=100000)
emp_years = st.number_input("Employment Years", value=5)
credit_amt = st.number_input("Credit Amount", value=300000)
annuity = st.number_input("Loan Annuity", value=15000)
family_members = st.number_input("Family Members", value=2)

# ---------------- CREDIT SCORES ----------------

st.subheader("External Credit Risk Scores (0 to 1)")

ext1 = st.number_input("External Risk Score 1", min_value=0.0, max_value=1.0, value=0.5)
ext2 = st.number_input("External Risk Score 2", min_value=0.0, max_value=1.0, value=0.5)
ext3 = st.number_input("External Risk Score 3", min_value=0.0, max_value=1.0, value=0.5)

# ---------------- SOCIAL RISK ----------------

region_rating = st.selectbox("Region Rating", [1,2,3], index=1)
obs_30 = st.number_input("Observed Defaults in Social Circle (30 Days)", value=0)
phone_change = st.number_input("Days Since Last Phone Change", value=300)

# ---------------- CATEGORICAL ----------------

education = st.selectbox(
    "Education Type",
    ["Secondary", "Higher", "Incomplete Higher"]
)

housing = st.selectbox(
    "Housing Type",
    ["House / apartment", "Rented apartment", "With parents"]
)

own_car = st.selectbox("Own Car", ["Y", "N"])
own_realty = st.selectbox("Own Realty", ["Y", "N"])
gender = st.selectbox("Gender", ["M", "F"])

# ---------------- DERIVED FEATURE ----------------

burden = annuity / income if income > 0 else 0

# ---------------- INPUT DICTIONARY ----------------

input_data = {
    "AGE_YEARS": age,
    "AMT_INCOME_TOTAL": income,
    "EMPLOYED_YEARS": emp_years,
    "AMT_CREDIT": credit_amt,
    "AMT_ANNUITY": annuity,
    "CNT_FAM_MEMBERS": family_members,
    "NAME_EDUCATION_TYPE": education,
    "NAME_HOUSING_TYPE": housing,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "CODE_GENDER": gender,
    "EXT_SOURCE_1": ext1,
    "EXT_SOURCE_2": ext2,
    "EXT_SOURCE_3": ext3,
    "REGION_RATING_CLIENT": region_rating,
    "OBS_30_CNT_SOCIAL_CIRCLE": obs_30,
    "DAYS_LAST_PHONE_CHANGE": phone_change
}

# ---------------- PREDICT ----------------

if st.button("Predict Credit Risk"):
    try:
        prob = predict_Target(
            input_dict=input_data,
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH
        )

        st.metric("Default Probability", f"{round(prob*100,2)}%")

        # -------- Threshold = 0.4 --------

        if prob > 0.4:
            st.error("❌ High Risk - Likely to Default")
        else:
            st.success("✅ Low Risk - Safe Applicant")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
