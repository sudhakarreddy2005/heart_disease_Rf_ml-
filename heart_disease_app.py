import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------------
# 🌟 Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts the **likelihood of heart disease** based on your medical details.  
Please enter your information below carefully 👇
""")

# -------------------------------------------------------------
# ⚙️ Load Model & Scaler
# -------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("heart_disease.joblib")     # trained Random Forest / SVC / Logistic Model
    scaler = joblib.load("scaler.joblib")           # same scaler used on full X dataset
    return model, scaler

model, scaler = load_resources()

# -------------------------------------------------------------
# 🩺 User Inputs
# -------------------------------------------------------------
st.header("🧍‍♂️ Patient Medical Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 10, 100, 45)
    gender = st.selectbox("Gender", ("Male", "Female"))
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, step=0.1)
    triglyceride = st.number_input("Triglyceride Level", 10, 500, 150)
    fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dl)", 50, 300, 100)

with col2:
    crp_level = st.number_input("C-Reactive Protein (CRP)", 0.0, 20.0, 3.0, step=0.1)
    homocysteine = st.number_input("Homocysteine Level", 0.0, 30.0, 10.0, step=0.1)
    smoking = st.selectbox("Smoking Habit", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    family_hd = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    high_bp = st.selectbox("High Blood Pressure", ["No", "Yes"])

# -------------------------------------------------------------
# 🔢 Convert Inputs to Numeric Format
# -------------------------------------------------------------
gender_num = 1 if gender == "Male" else 0
smoking_num = 1 if smoking == "Yes" else 0
diabetes_num = 1 if diabetes == "Yes" else 0
family_hd_num = 1 if family_hd == "Yes" else 0
high_bp_num = 1 if high_bp == "Yes" else 0

# Create DataFrame — must match training columns exactly
input_df = pd.DataFrame([[
    age, gender_num, blood_pressure, cholesterol, bmi, triglyceride,
    fasting_blood_sugar, crp_level, homocysteine,
    smoking_num, diabetes_num, family_hd_num, high_bp_num
]], columns=[
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "BMI", "Triglyceride Level",
    "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
    "Smoking", "Diabetes", "Family Heart Disease", "High Blood Pressure"
])

# -------------------------------------------------------------
# ⚖️ Apply Scaler
# -------------------------------------------------------------
# Quick fix: use .values to avoid feature name mismatch
input_scaled = scaler.transform(input_df.values)

# -------------------------------------------------------------
# 🔮 Predict
# -------------------------------------------------------------
st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(input_scaled)[0]
    try:
        prob = model.predict_proba(input_scaled)[0][1] * 100
    except:
        prob = None

    if prediction == 1:
        st.error("⚠️ **High Risk:** The model predicts that this person may have heart disease.")
    else:
        st.success("💚 **Low Risk:** The model predicts no significant heart disease risk detected.")

    # if prob is not None:
        # st.progress(int(prob))
        # st.write(f"Prediction confidence: **{prob:.2f}%**")

# -------------------------------------------------------------
# 📘 Footer
# -------------------------------------------------------------
st.markdown("---")
st.caption("Developed by Reddy | Tuned Random Forest Model ")
