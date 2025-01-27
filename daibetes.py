import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Title and Description
st.title("Diabetes Prediction App")
st.write("This app predicts whether a person has diabetes based on their health metrics.")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.sidebar.selectbox("Smoking History", ["never", "current", "former", "No Info"])
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    hba1c_level = st.sidebar.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=5.5)
    blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=0, max_value=300, value=120)

    data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Load dataset
data_url = '/mnt/data/diabetes_prediction_dataset.csv'
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Preprocess the dataset
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])

X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess user input
input_df['gender'] = label_encoder.transform(input_df['gender'])
input_df['smoking_history'] = label_encoder.transform(input_df['smoking_history'])

# Model selection
st.sidebar.header("Choose Model")
model_option = st.sidebar.selectbox("Select a machine learning model:",
                                    ["Random Forest", "K-Nearest Neighbors", "AdaBoost", "Decision Tree", "Logistic Regression"])

# Model initialization
if model_option == "Random Forest":
    model = RandomForestClassifier()
elif model_option == "K-Nearest Neighbors":
    model = KNeighborsClassifier()
elif model_option == "AdaBoost":
    model = AdaBoostClassifier()
elif model_option == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_option == "Logistic Regression":
    model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict on user input
prediction = model.predict(input_df)
prediction_prob = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Display prediction
st.subheader("Prediction")
if prediction[0] == 1:
    st.write("The model predicts: **Diabetic**")
else:
    st.write("The model predicts: **Non-Diabetic**")

# Display prediction probabilities (if available)
if prediction_prob is not None:
    st.subheader("Prediction Probabilities")
    st.write(f"Non-Diabetic: {prediction_prob[0][0] * 100:.2f}%")
    st.write(f"Diabetic: {prediction_prob[0][1] * 100:.2f}%")
