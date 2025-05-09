import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from db import connect_db, fetch_crop_data, insert_crop_data

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🌾 Smart Agriculture App", page_icon="🌱", layout="wide")

# ----------------------------
# CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to right, #e0f7fa, #fffde7);
        background-attachment: fixed;
    }
    h1 {
        color: #2e7d32;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
    }
    .css-1v0mbdj {
        border: 2px solid #81c784;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Models and Encoder
# ----------------------------
model = pickle.load(open('crop_model.pkl', 'rb'))  # Classification Model
le = pickle.load(open('label_encoder.pkl', 'rb'))  # LabelEncoder
kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))  # KMeans Model

# ----------------------------
# Title
# ----------------------------
st.title("🌾 Smart Agriculture Crop Prediction")

# ----------------------------
# Input Section
# ----------------------------
st.markdown("### Enter Soil and Weather Conditions:")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input('🌱 Nitrogen (N)', min_value=0, max_value=150, value=50)
        P = st.number_input('🌱 Phosphorus (P)', min_value=0, max_value=150, value=50)
        K = st.number_input('🌱 Potassium (K)', min_value=0, max_value=150, value=50)
        ph = st.number_input('🧪 pH value', min_value=0.0, max_value=14.0, value=6.5)

    with col2:
        temperature = st.number_input('🌡️ Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input('💧 Humidity (%)', min_value=0.0, max_value=100.0, value=60.0)
        rainfall = st.number_input('☔ Rainfall (mm)', min_value=0.0, max_value=400.0, value=100.0)

    submitted = st.form_submit_button('🌿 Predict, Cluster, and Save Crop')

if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Classification: Predict the crop using the trained model
    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]

    # Clustering: Apply KMeans clustering to group similar data points
    # Fetch all crop data for clustering
    data = fetch_crop_data()
    df = pd.DataFrame(data, columns=['ID', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop', 'Cluster_Label'])

    # Ensure column names are consistent (match training column names)
    df['temperature'] = df['temperature'].astype(float)
    df['humidity'] = df['humidity'].astype(float)
    df['rainfall'] = df['rainfall'].astype(float)

    # Ensure input data column names match the model's expectations
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features for clustering

    # Fit KMeans on data if needed and update the cluster labels
    df['Cluster_Label'] = kmeans.predict(X)

    # Predict cluster label for the new input data
    cluster_label = kmeans.predict(input_data)[0]

    # Display prediction result
    st.success(f"✅ Recommended Crop: **{crop_name}**")
    st.success(f"🌱 Assigned to Cluster: **{cluster_label}**")

    # Corrected part for saving data with 'cluster_label'
    try:
        insert_crop_data(
            None,  # ID is auto-generated
            int(N), 
            int(P), 
            int(K), 
            float(temperature), 
            float(humidity), 
            float(ph), 
            float(rainfall), 
            crop_name, 
            int(cluster_label)  # Ensure cluster_label is an integer
        )
        st.info("📦 Crop data saved to the database successfully!")
    except Exception as e:
        st.error(f"❌ Failed to save crop data: {e}")


# ----------------------------
# Show Crops from Database
# ----------------------------
st.markdown("---")
st.markdown("### 📋 View Available Crops:")

if st.button('📄 Show Available Crops'):
    with st.spinner('🔄 Loading crops...'):
        try:
            data = fetch_crop_data()
            if data:
                columns = ['ID', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop', 'Cluster_Label']
                df = pd.DataFrame(data, columns=columns)
                st.success('✅ Crops loaded successfully!')
                st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))

                # Download crops as Excel
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Crops Data as CSV",
                    data=csv,
                    file_name='crops_data.csv',
                    mime='text/csv',
                )
            else:
                st.warning('⚠️ No crops found in database.')
        except Exception as e:
            st.error(f"❌ Failed to fetch crops: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<center>Made with Student of Sanjivani College for Farmers by Smart Agriculture Team</center>", unsafe_allow_html=True)
