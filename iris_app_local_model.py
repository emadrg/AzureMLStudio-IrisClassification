import streamlit as st
import numpy as np
import joblib

try:
    model = joblib.load("v7_iris_model_neural_network.pkl") 
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


def normalize_features(sepal_length, sepal_width, petal_length, petal_width):

    min_vals = np.array([4.3, 2.0, 1.0, 0.1]) 
    max_vals = np.array([7.9, 4.4, 6.9, 2.5]) 
    input_vals = np.array([sepal_length, sepal_width, petal_length, petal_width])
    
    normalized_vals = (input_vals - min_vals) / (max_vals - min_vals)
    return normalized_vals.reshape(1, -1)

st.title("Iris Recognition App")
st.write("Enter the flower's features (as in the original dataset):")


sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)


if st.button("Predict"):

    features = normalize_features(sepal_length, sepal_width, petal_length, petal_width)
    
    try:
        prediction = model.predict(features)
        predicted_species = prediction[0] 
        
        st.success(f"The predicted iris species is: **{predicted_species}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")