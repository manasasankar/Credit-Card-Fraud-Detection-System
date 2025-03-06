import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")


st.title("🔍 Credit Card Fraud Detection ")


dataset_path = "dataset.csv"
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.lower()

if "isfraud" not in df.columns:
    st.error("⚠️ Dataset must contain an 'isfraud' column!")
    st.stop()

X = df.drop(columns=["isfraud"])
Y = df["isfraud"]

categorical_cols = X.select_dtypes(include=['object']).columns
if not categorical_cols.empty:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

st.subheader("📝 Enter Transaction Details")

input_data = {}
for col in df.drop(columns=["isfraud"]).columns:
    if df[col].dtype == "object":
        unique_vals = df[col].dropna().unique()
        input_data[col] = st.selectbox(f"📌 {col}", unique_vals)
    else:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        default = float(df[col].median())
        input_data[col] = st.number_input(f"📊 {col}", min_value=min_val, max_value=max_val, value=default)

input_df = pd.DataFrame([input_data])
if not categorical_cols.empty:
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
input_scaled = scaler.transform(input_df)

if st.button("🚀 Detect Fraud"):
    pred = model.predict(input_scaled)[0]
    
    accuracy = accuracy_score(Y_test, model.predict(X_test))

    if pred == 1:
        st.markdown(
            '<h2 style="text-align:center; color:red;">⚠️ Fraudulent Transaction Detected! ❌</h2>',
            unsafe_allow_html=True
        )
        st.error("🔴 Warning! This transaction is likely fraudulent.")
    else:
        st.markdown(
            '<h2 style="text-align:center; color:green;">✅ Legitimate Transaction ✔️</h2>',
            unsafe_allow_html=True
        )
        st.success("🟢 This transaction appears to be safe.")
    
    st.subheader("📊 Model Performance")
    st.progress(accuracy)
    st.markdown(
        f'<h3 style="text-align:center; color:blue;">📈 Model Accuracy: {accuracy:.2%}</h3>',
        unsafe_allow_html=True
    )
