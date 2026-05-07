# model.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Global objects
scaler = StandardScaler()
model = RandomForestClassifier(n_estimators=100, random_state=42)

def preprocess_data(df):
    df = df.copy()

    # Handle null values
    df.fillna(0, inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

def train_model(df):
    df = preprocess_data(df)

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return acc

def detect_anomaly(df):
    df = preprocess_data(df)

    X = scaler.transform(df.iloc[:, :-1])

    preds = model.predict(X)

    df['Prediction'] = preds

    return df

def anomaly_kmeans(df):
    df = preprocess_data(df)

    X = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    return df

def risk_score(row):
    # Simple logic
    return np.sum(row) % 100