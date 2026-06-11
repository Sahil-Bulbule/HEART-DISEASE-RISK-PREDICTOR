import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Health AI", page_icon="🫀")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = joblib.load("SVM2_Spam.pkl")
scaler = joblib.load("scaler2.pkl")
columns = joblib.load("columns2.pkl")

st.title("🫀 Heart Disease Risk Predictor")
st.write("Please enter your medical details for a quick health risk analysis.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 100, 19)
        resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 0.0, 0.1)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        cp_type = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
        ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submit = st.form_submit_button("Analyze Heart Risk")

if submit:
    input_data = {col: 0.0 for col in columns}
    
    input_data['Age'] = age
    input_data['RestingBP'] = resting_bp
    input_data['Cholesterol'] = cholesterol
    input_data['FastingBS'] = fasting_bs
    input_data['MaxHR'] = max_hr
    input_data['Oldpeak'] = oldpeak

    input_data[f'Sex_{gender[0]}'] = 1.0 
    input_data[f'ChestPainType_{cp_type}'] = 1.0
    input_data[f'RestingECG_{ecg}'] = 1.0
    input_data[f'ExerciseAngina_{angina}'] = 1.0
    input_data[f'ST_Slope_{slope}'] = 1.0
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns] 
    
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.error("Result: High Risk Detected. Please consult a cardiologist.")
    else:
        st.success("Result: Low Risk. Your parameters are in the healthy range.")