import numpy as np
import pandas as pd
import streamlit as st
import requests
import pickle
import joblib
import io



model_url = 'https://raw.githubusercontent.com/ezekielmose/Machine-Learning/refs/heads/main/hd_trained_model.sav'


# Fetch the model file from GitHub
response = requests.get(model_url)
response.raise_for_status()  # Ensure we notice bad responses (404, etc.)

# Load the model using pickle
model = pickle.load(io.BytesIO(response.content))


def main():
#input Variables
    
     # Interface title
    st.title("Heart Disease Prediction Machine Learning Model ")
    
    #getting the input data from the user  
    age = st.text_input("Enter the Patient's Age 15 - 80")
    #sex = st.text_input("Enter the Patient's Gender (0 [F] or 1[M])")

    # DROPDOWN DRROP BOX
    sex = st.selectbox( "What is the Gender", options=["Female", "Male"] )
    # Map the selected gender string to numeric value if needed
    gender_value = 0 if sex == "Female" else 1

    
    Chest_Pain = st.text_input("Chest Pain level (0,1,2 or 3)")
    Blood_Pressure= st.text_input("The Blood Pressure(mm Hg)level (94-200) ")
    cholestoral = st.text_input("Cholestoral Level (mg/dl) (131 -290)")
    Fasting_Blood_Sugar = st.text_input("Patient's Fasting Blood Sugar (0,1)")
    resting_electrocardiographic = st.text_input("Electrocardiographic level (0, 1 or 2)")
    Maximum_Heart_Rate= st.text_input("Maximum Heart Rate (99 - 162)")
    Excersize_Includes = st.text_input("Enter the Patient's Excersize_Includes")
    ST_Depression = st.text_input("Patient's ST Depression [ECG or EKG] (0.0 - 4.4)")
    Slope_of_Excersize	 = st.text_input("Patient's Slope of Excersize (0,1 or 2)")
    Number_of_vessels = st.text_input("Number of vessels (0, 1,2,3 or 4)")
    Thalassemia = st.text_input("Thalassemia (0, 1,2,3 or 4)")


    if st.button("PREDICT"):
        diagnosis = hearth_disease_prediction(])
    st.success(diagnosis)

    
    if st.button('CLICK HERE TO PREDICT'):
        makeprediction = model.predict([[age,gender_value,Chest_Pain,Blood_Pressure,cholestoral,Fasting_Blood_Sugar, resting_electrocardiographic,Maximum_Heart_Rate,Excersize_Includes,ST_Depression,Slope_of_Excersize,Number_of_vessels,Thalassemia]])
        output = round(makeprediction[0])  # Ensure it's an integer (0 or 1)
    
        if output == 0:
            st.success("The Person has Heart Disease")
        else:
            st.warning("The Person Does not have a Heart Disease")        


if __name__ == '__main__':
    main()
