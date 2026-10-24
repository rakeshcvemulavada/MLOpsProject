import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of travel package purchase by a consumer based on various parameters.
""")

# Customer Details
Age = st.number_input("Age", min_value=18, max_value=100, value=18)
TypeofContact = st.selectbox("Type pf Contact", ["Company Invited", "Self Inquiry"])
CityTier     = st.selectbox("City Tier", [1,2,3])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender    = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonsVisiting  = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star",[1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (per year)", min_value=0, max_value=50, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting",min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation",["Executive", "Manager", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income",min_value=5000, max_value=500000, value=50000)

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score",min_value=1, max_value=5, value=3)

ProductPitched = st.selectbox("Product Pitched",["Basic", "Standard", "Deluxe", "Super Deluxe"])

NumberOfFollowups = st.number_input("Number of Follow-ups",min_value=0, max_value=20, value=2)

DurationOfPitch = st.number_input("Duration of Pitch (minutes)",min_value=1, max_value=120, value=15)

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonsVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# Classification threshold
classification_threshold = 0.5

if st.button("Predict"):
    prediction_probability = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_probability >= classification_threshold).astype(int)

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("The customer is likely to purchase the package.")
    else:
        st.error("The customer is unlikely to purchase the package.")

if st.button("Predict"):
    prediction = model.predict(input_data)[0,1]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
