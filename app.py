import streamlit as st
import numpy as np
import pickle
import base64

# Load model and scaler
model = pickle.load(open("voting_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below:")

# Sample inputs for diabetic and non-diabetic
sample_inputs = {
    "Diabetic": {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 200,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    },
    "Non-Diabetic": {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 96,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
    }
}

# Initialize session state for reset and sample handling
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 100,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }

# Buttons to auto-fill with sample data
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Use sample: Diabetic"):
        st.session_state.inputs.update(sample_inputs["Diabetic"])
with col2:
    if st.button("Use sample: Non-Diabetic"):
        st.session_state.inputs.update(sample_inputs["Non-Diabetic"])
with col3:
    if st.button("Reset"):
        st.session_state.inputs = {
            "Pregnancies": 1,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 100,
            "BMI": 30.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 30
        }

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=st.session_state.inputs["Pregnancies"], key="preg")
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=st.session_state.inputs["Glucose"], key="glucose")
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=st.session_state.inputs["BloodPressure"], key="bp")
skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=st.session_state.inputs["SkinThickness"], key="skin")
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=st.session_state.inputs["Insulin"], key="insulin")
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=st.session_state.inputs["BMI"], key="bmi")
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=st.session_state.inputs["DiabetesPedigreeFunction"], key="ped")
age = st.number_input("Age", min_value=10, max_value=100, value=st.session_state.inputs["Age"], key="age")

# Prediction
if st.button("Predict"):
    features = np.array([[pregnancies, glucose, bp, skinthickness, insulin, bmi, pedigree, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    st.markdown("---")
    st.subheader("🔎 Prediction Result")
    st.progress(prob, text=f"Probability: {prob:.2%}")

    if prediction == 1:
        st.error(f"⚠️ Diabetic Risk Detected (Probability: {prob:.2%})")
    else:
        st.success(f"✅ No Diabetes Risk (Probability: {prob:.2%})")

    # Additional info block
    st.markdown("""
    ---
    #### ℹ️ Model Info
    This prediction is made using a **Weighted Voting Classifier**, combining:
    - Decision Tree
    - Random Forest
    - Gradient Boosting

    Weights are assigned to models based on individual accuracy. Models were trained on the Pima Indians Diabetes dataset.
    """)
