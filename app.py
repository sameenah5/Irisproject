import pickle
import pandas as pd
import numpy as np
import streamlit as st
import os

# -------------------------------
# Function to predict species
# -------------------------------
def predict_species(sep_len, sep_width, pet_len, pet_width, scaler_path, model_path):
    try:
        # Check if scaler and model files exist
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found: {scaler_path}")
            return None, None
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None

        # Load the scaler
        with open(scaler_path, 'rb') as file1:
            scaler = pickle.load(file1)
        # Load the model
        with open(model_path, 'rb') as file2:
            model = pickle.load(file2)

        # Create a dataframe from user inputs
        data = {
            'SepalLengthCm': [sep_len],
            'SepalWidthCm': [sep_width],
            'PetalLengthCm': [pet_len],
            'PetalWidthCm': [pet_width]
        }
        x_new = pd.DataFrame(data)

        # Scale input
        x_scaled = scaler.transform(x_new)

        # Make predictions
        pred = model.predict(x_scaled)
        prob = model.predict_proba(x_scaled)
        max_prob = np.max(prob)

        return pred, max_prob

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title(" Iris Species Predictor")

st.markdown("Enter the measurements of your Iris flower below:")

sep_len = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=5.1)
sep_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=3.5)
pet_len = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=1.4)
pet_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=0.2)

# Paths to your trained scaler and model
scaler_path = "notebook/scaler.pkl"
model_path = "notebook/model.pkl"

if st.button("Predict"):
    pred, max_prob = predict_species(sep_len, sep_width, pet_len, pet_width, scaler_path, model_path)

    if pred is not None and max_prob is not None:
        st.subheader(f"Predicted species: **{pred[0]}**")
        st.subheader(f"Prediction probability: **{max_prob:.4f}**")
    else:
        st.error("Prediction failed. Check input values and ensure model files exist.")
