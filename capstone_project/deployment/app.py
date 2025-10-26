import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="madhukarmehta/tourism-package-prediction", filename="best_tourism_pkg_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction")
st.write("""
This application predicts predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")
st.subheader("Enter the listing details:")

# Collect user input
Age = st.number_input("Age",min_value=18,max_value=61,value=18)
CityTier = st.selectbox("CityTier", ["1", "2", "3"])
DurationOfPitch = st.number_input("DurationOfPitch",min_value=5,max_value=127,value=5)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting",min_value=1,max_value=5,value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups",min_value=1,max_value=6,value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar",min_value=3,max_value=5,value=3)
NumberOfTrips =  st.number_input("NumberOfTrips",min_value=1,max_value=22,value=1)
Passport = st.number_input("Passport",min_value=1,max_value=22,value=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore",min_value=1,max_value=5,value=1)
OwnCar = st.selectbox("OwnCar",["0", "1"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting",min_value=0,max_value=3,value=0)
MonthlyIncome = st.number_input("MonthlyIncome",min_value=1000,max_value=100000,value=1000)
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("MaritalStatus", ["Unmarried", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP" ,"VP"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender' : Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus' : MaritalStatus,
    'NumberOfTrips' : NumberOfTrips,
    'Passport' : Passport,
    'PitchSatisfactionScore' : PitchSatisfactionScore,
    'OwnCar' : OwnCar,
    'NumberOfChildrenVisiting' : NumberOfChildrenVisiting,
    'Designation' : Designation,
    'MonthlyIncome' : MonthlyIncome
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
