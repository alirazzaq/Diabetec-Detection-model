
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the trained model and scaler
model = LogisticRegression(random_state=42)
scaler = StandardScaler()

# Define the prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

# Streamlit app
st.title("Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Prediction button
if st.button("Predict"):
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    st.write(f"The person is predicted to be: {result}")
