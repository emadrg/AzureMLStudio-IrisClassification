import streamlit as st
import numpy as np
import json
import urllib.request
import ssl
import os

def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

st.title("Iris Recognition App 🌸")
st.write("Enter the flower's features (as in the original dataset):")

sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

def normalize_features(sepal_length, sepal_width, petal_length, petal_width):
    min_vals = np.array([4.3, 2.0, 1.0, 0.1]) 
    max_vals = np.array([7.9, 4.4, 6.9, 2.5]) 
    input_vals = np.array([sepal_length, sepal_width, petal_length, petal_width])
    normalized_vals = (input_vals - min_vals) / (max_vals - min_vals)
    return normalized_vals.reshape(1, -1).tolist()

species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

url = 'http://0b792779-e356-4e7a-a29b-d7b2fcbe199c.westeurope.azurecontainer.io/score'
headers = {'Content-Type': 'application/json'}

if st.button("Predict"):
    features = normalize_features(sepal_length, sepal_width, petal_length, petal_width)
    data = {"data": features}  
    body = str.encode(json.dumps(data))

    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode("utf-8"))
        predicted_index = result.get("predictions", [-1])[0]  # Extract first prediction
        predicted_species = species_mapping.get(predicted_index, "Unknown")  # Map to species name
        st.success(f"The predicted iris species is: **{predicted_species}**")
    except urllib.error.HTTPError as error:
        st.error(f"Request failed with status code: {error.code}")
        st.text(error.read().decode("utf8", 'ignore'))
