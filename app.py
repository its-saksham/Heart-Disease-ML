# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="💓")
st.title("💓 Heart Attack Risk Prediction App")

preprocess = joblib.load("preprocess.pkl")
pipe_lr = joblib.load("pipe_lr.pkl")
threshold_lr = joblib.load("thresholds.pkl")["lr"]

df = pd.read_csv("heart_2022_no_nans.csv")
num_cols = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "HeightInMeters", "WeightInKilograms", "BMI"]

demo_patients = {
    "Healthy": {
        "Sex": "Female",
        "GeneralHealth": "Excellent",
        "PhysicalHealthDays": 0,
        "MentalHealthDays": 0,
        "SleepHours": 8,
        "PhysicalActivities": "Yes",
        "SmokerStatus": "Never smoker",
        "HadDiabetes": "No",
        "AgeCategory": "35-39",
        "BMI": 22.5,
    },
    "High-Risk": {
        "Sex": "Male",
        "GeneralHealth": "Poor",
        "PhysicalHealthDays": 20,
        "MentalHealthDays": 15,
        "SleepHours": 4,
        "PhysicalActivities": "No",
        "SmokerStatus": "Current smoker",
        "HadDiabetes": "Yes",
        "AgeCategory": "65-69",
        "BMI": 32.0,
    }
}

mode = st.radio("Input Mode", ["Manual Input", "Demo Patient Mode"])

inp = {}
if mode == "Demo Patient Mode":
    selected_demo = st.selectbox("Select Demo Patient", list(demo_patients.keys()))
    inp = demo_patients[selected_demo]
else:
    # Manual Input
    st.subheader("Enter Patient Details")
    template = demo_patients["Healthy"]  # use healthy template
    for col in template:
        if col in num_cols:
            inp[col] = st.number_input(col, value=float(template[col]))
        else:
            options = [str(x).strip() for x in df[col].unique()]
            default_val = str(template[col]).strip()
            default_index = options.index(default_val) if default_val in options else 0
            inp[col] = st.selectbox(col, options, index=default_index)

st.subheader("Patient Input Summary")
input_display_df = pd.DataFrame([inp])
st.table(input_display_df)

input_df = pd.DataFrame([inp])

expected_cols = preprocess.feature_names_in_
for col in expected_cols:
    if col not in input_df.columns:
        if col in num_cols:
            input_df[col] = 0
        else:
            input_df[col] = df[col].mode()[0]  

input_df = input_df[expected_cols]

input_proc = preprocess.transform(input_df)

if st.button("Predict"):
    prob = float(pipe_lr.predict_proba(input_proc)[0, 1])
    pred = int(prob >= threshold_lr)

    if pred == 1:
        st.error(f"⚠️ High Risk (thr={threshold_lr:.2f}): {prob:.2f} probability of Heart Attack")
    else:
        st.success(f"✅ Low Risk (thr={threshold_lr:.2f}): {1 - prob:.2f} probability of No Heart Attack")

    st.caption(f"Debug → P(y=1)={prob:.2f} | Threshold={threshold_lr:.2f} | Model=Logistic Regression")

