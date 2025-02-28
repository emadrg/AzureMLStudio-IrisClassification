# AzureMLStudio-IrisClassification

# **Iris Flower Classification App**

## **Overview**
This project is an **Iris flower classification application** using **machine learning** and **Streamlit**. A **RandomForestClassifier** was trained on the **normalized Iris dataset** and exported as `iris_model.pkl`. Users input real flower measurements (cm), which are **automatically normalized** before making a prediction. The app then classifies the flower as **Setosa, Versicolor, or Virginica**.
The dataset was imported from Kaggle

The model was trained and **registered in Azure Machine Learning**, ensuring scalability. The app provides an **intuitive UI** with sliders for input and displays predictions dynamically. This project demonstrates **end-to-end ML deployment**, from training to a user-friendly web interface.

## **Dataset**
The dataset used for training the model can be found on Kaggle: [Iris Dataset](https://www.kaggle.com/datasets/saurabh00007/iriscsv)

## **Code Structure**
- **`notebook_1.py`**: Contains the code for training the machine learning model.
- **`app.py`**: Exposes an API and serves as the main application using the deployed model.
- **`iris_app_local_model.py`**: Implements a second approach where the model is exported and saved locally, instead of being registered in Azure.

## **Features**
- **Machine Learning Model**: RandomForestClassifier trained on the normalized Iris dataset.
- **Streamlit UI**: User-friendly interface with sliders for input.
- **Real-time Predictions**: Dynamically updates based on user inputs.
- **Two Deployment Options**:
  - Using an API (`app.py`)
  - Using a locally stored model (`iris_app_local_model.py`)
- **Scalable Deployment**: Model registered in Azure Machine Learning.

## **How to Run the Project**
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the App**
   - Using the API:
     ```bash
     streamlit run app.py
     ```
   - Using the Local Model:
     ```bash
     streamlit run iris_app_local_model.py
     ```

## **Conclusion**
This project showcases **machine learning deployment**, covering model training, API exposure, and an interactive user interface. It provides an accessible way to classify Iris flowers with real-time predictions.

