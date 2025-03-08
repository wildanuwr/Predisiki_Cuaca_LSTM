import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("model_cuaca.h5")


data = pd.read_csv("datasetCuaca.csv")
features = ["Temperature (C)", "Apparent Temperature (C)", "Humidity"]


data = data.dropna(subset=features)

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data[features])
# Label cuaca yang akan diprediksi
weather_labels = ["snow", "rain"]

# Streamlit UI
st.title("🌦️ Prediksi Cuaca dengan LSTM")
st.markdown("Masukkan suhu, suhu terasa, dan kelembaban untuk memprediksi kondisi cuaca.")

# Input dari pengguna
temp = st.number_input("Masukkan suhu (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
apparent_temp = st.number_input("Masukkan suhu terasa (°C):", min_value=-10.0, max_value=50.0, value=27.0, step=0.1)
humidity = st.number_input("Masukkan kelembaban (%):", min_value=-10.0, max_value=100.0, value=60.0, step=0.1)

# Tombol prediksi
if st.button("Prediksi Cuaca"):
    # Konversi kelembaban ke format 0-1
    humidity = humidity / 100.0

    # Normalisasi input
    user_input = scaler.transform([[temp, apparent_temp, humidity]])

    # Bentuk sequence dengan panjang yang sesuai untuk LSTM
    sequence_length = 10
    input_sequence = np.tile(user_input, (sequence_length, 1)).reshape(1, sequence_length, len(features))

    # Prediksi kondisi cuaca
    predicted_weather = model.predict(input_sequence)
    predicted_class = np.argmax(predicted_weather)
    predicted_condition = weather_labels[predicted_class]

    # Tampilkan hasil prediksi
    st.success(f"🌤️ Prediksi kondisi cuaca: **{predicted_condition}**")
