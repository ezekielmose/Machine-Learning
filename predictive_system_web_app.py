# importing the libaries

import numpy as np
import pickle
import streamlit as st
import pandas as pd


### Loading the saved model

loaded_model = pickle.load(open('E:/Ezekiel/Model_Deployment/trained_model.sav', 'rb'))

# creating a function

def diabetes_prediction(input_data):

    
    ## changing input data to numpy array because processing is easier than list 
    input_data_as_numpy_array= np.array(input_data)
    # reshaping the array for predicting 
    
    # Prepare the input data as an array or DataFrame (depending on your model)
    # input_data = [age, sex, Chest_Pain, Blood_Pressure, cholestoral, Fasting_Blood_Sugar, resting_electrocardiographic, Maximum_Heart_Rate, Excersize_Includes, ST_Depression, Slope_of_Excersize, Number_of_vessels, Thalassemia]
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    
    
   #  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    print(prediction)
    
    if prediction [0] == 0:
        return "The Person Does not have a Heart Disease"
    else:
        return "The Person has Heart Disease"
    
    
    
    
#  Streamlit library to craete a user interface

def main():
    
    # Interface title
    st.title(" Heart Disease Prediction ")
    
    #input database  
    age = st.text_input("Enter the Patient's Age")
    sex = st.text_input("Enter the Patient's Gender")
    Chest_Pain = st.text_input("Chest_Pain level")
    Blood_Pressure= st.text_input("Enter the Patient's Blood_Pressure(mm Hg) level")
    cholestoral = st.text_input("Enter the Patient's cholestoral (mg/dl) level")
    Fasting_Blood_Sugar = st.text_input("Enter the Patient's Fasting_Blood_Sugar level")
    resting_electrocardiographic = st.text_input("Enter the Patient's resting electrocardiographic level")
    Maximum_Heart_Rate = st.text_input("Enter the Patient's Maximum_Heart_Rate Value")
    Excersize_Includes = st.text_input("Enter the Patient's Excersize_Includes (0/1)")
    ST_Depression = st.text_input("Enter the Patient's ST Depression Level")
    Slope_of_Excersize = st.text_input("Enter the Patient's Slope_of_Excersize")
    Number_of_vessels = st.text_input("Enter the Patient's Number_of_vessels")
    Thalassemia = st.text_input("Enter the Patient's Thalassemia level")
    
    
    ## Numeric conversion
    
    # Convert inputs to numeric using pd.to_numeric or float conversion
    age = pd.to_numeric(age, errors='coerce')
    sex = pd.to_numeric(sex, errors='coerce')
    Chest_Pain = pd.to_numeric(Chest_Pain, errors='coerce')
    Blood_Pressure = pd.to_numeric(Blood_Pressure, errors='coerce')
    cholestoral = pd.to_numeric(cholestoral, errors='coerce')
    Fasting_Blood_Sugar = pd.to_numeric(Fasting_Blood_Sugar, errors='coerce')
    resting_electrocardiographic = pd.to_numeric(resting_electrocardiographic, errors='coerce')
    Maximum_Heart_Rate = pd.to_numeric(Maximum_Heart_Rate, errors='coerce')
    Excersize_Includes = pd.to_numeric(Excersize_Includes, errors='coerce')
    ST_Depression = pd.to_numeric(ST_Depression, errors='coerce')
    Slope_of_Excersize = pd.to_numeric(Slope_of_Excersize, errors='coerce')
    Number_of_vessels = pd.to_numeric(Number_of_vessels, errors='coerce')
    Thalassemia = pd.to_numeric(Thalassemia, errors='coerce')
    
    # Handle non-numeric values that are coerced to NaN
    if any(pd.isna([age, sex, Chest_Pain, Blood_Pressure, cholestoral, Fasting_Blood_Sugar, resting_electrocardiographic, Maximum_Heart_Rate, Excersize_Includes, ST_Depression, Slope_of_Excersize, Number_of_vessels, Thalassemia])):
        st.error("Invalid input: Please ensure all inputs are numeric.")
    
    
    # code for prediction
    
    diagnosis = ''
    
    # creating  a prediction button
    
    if st.button("Make Your Prediction"):
        diagnosis = diabetes_prediction([age, sex, Chest_Pain, Blood_Pressure, cholestoral, Fasting_Blood_Sugar, resting_electrocardiographic, Maximum_Heart_Rate,Excersize_Includes, ST_Depression, Slope_of_Excersize, Number_of_vessels, Thalassemia])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    

    
    
    
    
    
    
    
    
    
    
    
    