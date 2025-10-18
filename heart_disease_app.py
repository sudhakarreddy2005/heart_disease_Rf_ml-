import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts the **likelihood of heart disease** based on medical details.  
Please enter your information below carefully 👇
""")

@st.cache_resource
def load_model():
    model = joblib.load("heart_disease.joblib")
    return model

model = load_model()

st.header("🧍‍♂️ Patient Medical Information")

# User inputs for your chosen features
age = st.number_input("Age (years)", min_value=10, max_value=100, value=45)
gender = st.selectbox("Gender", ("Male", "Female"))
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
triglyceride = st.number_input("Triglyceride Level", min_value=10, max_value=500, value=150)
fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50, max_value=300, value=100)
crp_level = st.number_input("CRP Level", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
homocysteine = st.number_input("Homocysteine Level", min_value=0.0, max_value=30.0, value=10.0, step=0.1)

# For categorical yes/no inputs:
smoking = st.selectbox("Smoking", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
family_hd = st.selectbox("Family Heart Disease", ["No", "Yes"])
high_bp = st.selectbox("High Blood Pressure", ["No", "Yes"])

# Convert categorical to numeric
gender_num = 1 if gender == "Male" else 0
smoking_num = 1 if smoking == "Yes" else 0
diabetes_num = 1 if diabetes == "Yes" else 0
family_hd_num = 1 if family_hd == "Yes" else 0
high_bp_num = 1 if high_bp == "Yes" else 0

# Prepare DataFrame in the same order features were used to train
input_data = pd.DataFrame([[
    age, gender_num, blood_pressure, cholesterol, bmi, triglyceride,
    fasting_blood_sugar, crp_level, homocysteine,
    smoking_num, diabetes_num, family_hd_num, high_bp_num
]], columns=[
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "BMI", "Triglyceride Level",
    "Fasting Blood Sugar", "CRP Level", "Homocysteine Level",
    "Smoking", "Diabetes", "Family Heart Disease", "High Blood Pressure"
])

st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(input_data)[0]
    prob = None
    try:
        prob = model.predict_proba(input_data)[0][1] * 100
    except:
        pass

    if prediction == 1:
        st.error("⚠️ **High Risk:** The model predicts this person may have heart disease.")
    else:
        st.success("💚 **Low Risk:** The model predicts no heart disease risk detected.")

    if prob is not None:
        st.progress(int(prob))
        st.write(f"Prediction confidence: **{prob:.2f}%**")

st.markdown("---")
st.caption("Developed with ❤️  |  Machine Learning Model")
