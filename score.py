import json
import joblib
import numpy as np
import os
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("v7_iris_model_neural_network")
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)["data"]
        predictions = model.predict(np.array(data))
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
