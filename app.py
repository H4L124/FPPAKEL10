import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from joblib import load
from sklearn.preprocessing import StandardScaler

# Initialize dummy scaler and model
svm_scaler = StandardScaler()
svm_model = SVC()

# Example scaled data for initialization
svm_scaler.fit([[0, 0, 0], [30000, 86400*365, 365]])

# Dummy model fit for demonstration purposes
svm_model.fit(svm_scaler.transform([[0, 0, 0], [30000, 86400*365, 365]]), [0, 1])

# Initialize session state variables
if 'days' not in st.session_state:
    st.session_state.days = 0.0
if 'second' not in st.session_state:
    st.session_state.second = 0.0

# Function to convert days to seconds
def convert_days_to_seconds(days):
    return days * 86400

# Function to convert seconds to days
def convert_seconds_to_days(seconds):
    return seconds / 86400

# Callback function when days input changes
def update_days():
    st.session_state.second = convert_days_to_seconds(st.session_state.days)

# Callback function when seconds input changes
def update_seconds():
    st.session_state.days = convert_seconds_to_days(st.session_state.second)

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Deskripsi Data", "Prediksi SVM", "Prediksi KMeans SVM", "Perbandingan Model", "Prediksi Baru"])

# New Prediction Page
if page == "Prediksi Baru":
    st.title("Prediksi Menggunakan Model SVM")

    # Input fields for amount, days, and seconds
    amount = st.number_input("Amount", min_value=0.0, max_value=30000.0)
    
    # Input field for days
    days = st.number_input("Days", min_value=0.0, value=st.session_state.days, step=1.0, key='days', on_change=update_days)

    # Input field for seconds
    second = st.number_input("Second", min_value=0.0, value=st.session_state.second, step=1.0, key='second', on_change=update_seconds)

    if st.button("Prediksi"):
        input_data = np.array([[amount, second, days]])
        standardized_input = svm_scaler.transform(input_data)
        prediction = svm_model.predict(standardized_input)
        st.write(f"Hasil Prediksi: {'Transaksi kartu kredit ini adalah Penipuan' if prediction[0] == 1 else 'Transaksi kartu kredit ini adalah Sah'}")

# Run other pages here (Deskripsi Data, Prediksi SVM, Prediksi KMeans SVM, Perbandingan Model)
elif page == "Deskripsi Data":
    st.title("Statistika Deskriptif")
    # Include descriptive statistics, charts, etc.

elif page == "Prediksi SVM":
    st.title("Prediksi Menggunakan SVM")
    # Include SVM predictions, confusion matrix, evaluation metrics, etc.

elif page == "Prediksi KMeans SVM":
    st.title("Prediksi Menggunakan KMeans SVM")
    # Include KMeans SVM predictions, confusion matrix, evaluation metrics, etc.

elif page == "Perbandingan Model":
    st.title("Perbandingan Model SVM dan KMeans SVM")
    # Include comparison of SVM and KMeans SVM, ROC curve, etc.

# Ensure that all plots, figures, and data frames are properly displayed
