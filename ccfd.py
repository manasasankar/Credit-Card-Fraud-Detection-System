import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

st.title("Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    credit_card_data = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Overview")
    st.write(credit_card_data.head())
    st.write(credit_card_data.tail())
    st.write(credit_card_data.info())
    st.write(credit_card_data.isnull().sum())
    st.write(credit_card_data['IsFraud'].value_counts())

    legit = credit_card_data[credit_card_data.IsFraud == 0]
    fraud = credit_card_data[credit_card_data.IsFraud == 1]
    
    st.subheader("Legitimate vs Fraudulent Transactions")
    st.write(f"Legit shape: {legit.shape}")
    st.write(f"Fraud shape: {fraud.shape}")
    st.write("Legit Transaction Amount Stats:")
    st.write(legit.Amount.describe())
    st.write("Fraud Transaction Amount Stats:")
    st.write(fraud.Amount.describe())
    st.write(credit_card_data.groupby('IsFraud').mean(numeric_only=True))
    
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    
    st.subheader("Balanced Dataset")
    st.write(new_dataset.head())
    st.write(new_dataset.tail())
    st.write(new_dataset['IsFraud'].value_counts())
    st.write(new_dataset.groupby('IsFraud').mean(numeric_only=True))
    
    X = new_dataset.drop(columns='IsFraud', axis=1)
    Y = new_dataset['IsFraud']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    for col in X_train.columns:
        if X_train[col].dtype == 'object':  
            X_train[col] = encoder.fit_transform(X_train[[col]])
            X_test[col] = encoder.transform(X_test[[col]])  
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    st.subheader("Model Accuracy")
    st.write(f'Accuracy: {accuracy}')
