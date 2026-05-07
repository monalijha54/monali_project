import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from model import train_model, detect_anomaly, anomaly_kmeans, risk_score

scaler_path='scaler.pkl'
model_path='model.pkl'

with open(scaler_path,'rb') as file1:
 scaler = pkl.load(file1)
 with open(model_path,'rb')as file2:
  model = pkl.load(file2)

st.set_page_config(page_title="Cybersecurity SIEM", layout="wide")

st.title("🔐 AI-Powered Cybersecurity Log Anomaly Detection")


file = st.file_uploader("Upload Log Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    
    if st.button("Train Model"):
        acc = train_model(df)
        st.success(f"Model trained successfully! Accuracy: {acc:.2f}")

    
    if st.button("Detect Anomalies"):
        result = detect_anomaly(df)

        st.subheader("🚨 Predictions")
        st.dataframe(result.head())

        
        result['Risk Score'] = result.apply(risk_score, axis=1)

        st.subheader("⚠️ Risk Scores")
        st.dataframe(result[['Risk Score']].head())

        st.subheader("📈 Visualization")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Prediction Count")
            sns.countplot(x='Prediction', data=result)
            st.pyplot(plt)

        with col2:
            st.write("Risk Score Distribution")
            sns.histplot(result['Risk Score'], kde=True)
            st.pyplot(plt)

        
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", csv, "datasetp.csv", "text/csv")

    
    if st.button("Run KMeans Clustering"):
        km = anomaly_kmeans(df)
        st.dataframe(km.head())

        st.write("Cluster Distribution")
        sns.countplot(x='Cluster', data=km)
        st.pyplot(plt)

    scaler_path = 'anomaly/scaler.pkl'
    model_path = 'anomaly/model.pkl'