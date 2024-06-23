import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_model.pkl')

# Define the input fields
def get_user_input():
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
    
    # Create a dataframe for user inputs
    user_data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Streamlit app
st.title("Heart Disease Prediction")

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("Patient Details")
st.write(user_input)

# Add a button to rerun the prediction
if st.button("Predict"):
    # Predict using the model
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # Display the prediction
    st.subheader("Prediction")
    heart_disease = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Does the patient likely have heart disease? **{heart_disease}**")

    st.subheader("Prediction Probability")
    st.write(f"Probability of having heart disease: **{prediction_proba[0][1]:.2f}**")
    st.write(f"Probability of not having heart disease: **{prediction_proba[0][0]:.2f}**")
