# importing the libaries

import numpy as np
import pickle # to load the model
import streamlit as st
import pandas as pd


# Loading the saved model copy the loaded_model line of code from jupyter notebook
# copy the path to where the loaded model is savel
# change the \ to /
loaded_model = pickle.load(open('E:/Ezekiel/Model_Deployment/trained_model.sav', 'rb'))


# creating a function for prediction
def hearth_disease_prediction(input_data):

    ## Copy from Jupyter, the code for the unstandadized data 
    ## changing input data to numpy array because processing is easier than list 
    input_data_as_numpy_array= np.array(input_data)
    # reshaping the array for predicting 
    
    # Prepare the input data as an array or DataFrame (depending on your model)
    # input_data = [age, sex, Chest_Pain, Blood_Pressure]
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    
   #  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    print(prediction)
    if prediction [0] == 0:
        return "The Person Does not have a Heart Disease" # insted of print change to return
    else:
        return "The Person has Heart Disease" # insted of print change to return  
    
# Streamlit library to craete a user interface
def main():
    
    # Interface title
    st.title(" Heart Disease Prediction ")
    
    #getting the input data from the user  
    age = st.text_input("Enter the Patient's Age")
    sex = st.text_input("Enter the Patient's Gender")
    Chest_Pain = st.text_input("Chest_Pain level")
    Blood_Pressure= st.text_input("Enter the Patient's Blood_Pressure(mm Hg) level")

    
    ## Numeric conversion
    # Convert inputs to numeric using pd.to_numeric or float conversion
    age = pd.to_numeric(age, errors='coerce')
    sex = pd.to_numeric(sex, errors='coerce')
    Chest_Pain = pd.to_numeric(Chest_Pain, errors='coerce')
    Blood_Pressure = pd.to_numeric(Blood_Pressure, errors='coerce')


    # code for prediction
    diagnosis = '' # string tha ontaons null values whose values are stored in the prediction
    
    # creating  a prediction button
    if st.button("Predict"):
        diagnosis = hearth_disease_prediction([age, sex, Chest_Pain, Blood_Pressure])
    st.success(diagnosis)
    
 
# this is to allow our web app to run from anaconda command prompt where the cmd takes the main() only and runs the code
if __name__ == '__main__':
    main()
    

    
    
    
    
    
    
    
    
    
    
    
    