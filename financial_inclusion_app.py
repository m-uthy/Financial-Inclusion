import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Dropdown options
cols = ['education_level', 'cellphone_access', 'country',
       'gender_of_respondent', 'location_type', 'relationship_with_head',
       'job_type']

country = ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']
gender = ["Male", "Female"]
relationship_with_head = ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives']

locations = ["Rural", "Urban"]

job_type = ['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income']

cellphone = ["Yes", "No"]

edu_level = ['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA']

#Streamlit Inputs
st.header("Financial Inclusion") 
with st.form( key = "user_input_data"):
    selected_country = st.selectbox("Enter Country", options = ["Choose an option"] + country)
    selected_gender = st.selectbox("Enter gender", options= ["Choose an option"] + gender)
    selected_location = st.selectbox("Enter location", options = ["Choose an option"] + locations)
    selected_job_type = st.selectbox("Enter job type", options= ["Choose an option"] + job_type)
    selected_cellphone = st.selectbox("Do you have a cellphone", options = ["Choose an option"] + cellphone)
    selected_edu_level = st.selectbox("What's your level of education ", options= ["Choose an option"] + edu_level)
    selected_relationship  = st.selectbox("What's your relationship with the head ", options= ["Choose an option"] + relationship_with_head)
    submit_button = st.form_submit_button(label = "Predict")


#Load the model and encoder
with open("rf_model.pkl", "rb") as file:
    my_model = joblib.load(file)


with open("label_encoders.pkl", "rb") as file:
    label_encoders = joblib.load(file)


# Process inputs and predict
if submit_button:
        # Encode inputs using saved LabelEncoders
        encoded_data = [
        label_encoders["education_level"].transform([selected_edu_level])[0],
        label_encoders["cellphone_access"].transform([selected_cellphone])[0],
        label_encoders["country"].transform([selected_country])[0],
        label_encoders["gender_of_respondent"].transform([selected_gender])[0],
        label_encoders["location_type"].transform([selected_location])[0],
        label_encoders["relationship_with_head"].transform([selected_relationship])[0],
        label_encoders["job_type"].transform([selected_job_type])[0]
        ]

        # Reshape for prediction
        input_data = np.array(encoded_data).reshape(1, -1)
        prediction = my_model.predict(input_data)


        if prediction == 1 :
            st.success("The Person will have a bank account ", icon= "✅")
        else:
            st.error("The person doesn't have a bank account.", icon="❌")





