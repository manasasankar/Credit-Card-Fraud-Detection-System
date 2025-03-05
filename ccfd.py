import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score

st.title("🔍 Credit Card Fraud Detection")

dataset_path = "dataset.csv" 
df = pd.read_csv(dataset_path)

df.columns = df.columns.str.lower()  # Ensure all columns are lowercase

st.subheader("📊 Dataset Preview")
st.write(df.head())

# Check if 'isfraud' exists
if "isfraud" not in df.columns:
    st.error("⚠️ The dataset must contain an 'isfraud' column.")
    st.stop()

# Class Distribution
st.subheader("🧐 Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x=df["isfraud"], palette="coolwarm", ax=ax)
ax.set_title("Legit vs Fraud Transactions")
st.pyplot(fig)
plt.close(fig)  # Close figure to prevent Streamlit caching issues

# Preprocessing
X = df.drop(columns=["isfraud"])
Y = df["isfraud"]

categorical_cols = X.select_dtypes(include=['object']).columns
if not categorical_cols.empty:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model Training
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Model Accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

st.subheader("🎯 Model Accuracy")
st.success(f"✅ Accuracy: {accuracy:.2%}")

if accuracy >= 0.9:
    st.balloons()

# Fraud Trend Over Time (Fixed Column Names)
if "transactiondate" in df.columns:
    st.subheader("⏳ Fraud Trend Over Time")
    
    # Convert to datetime properly
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")

    # Ensure conversion worked
    if df["transactiondate"].isna().all():
        st.error("⚠️ All transaction dates are invalid. Check dataset format.")
    else:
        df.dropna(subset=["transactiondate"], inplace=True)  # Remove invalid dates

        # Group by date for fraud cases
        fraud_trend = df[df["isfraud"] == 1].groupby(df["transactiondate"].dt.date).size()

        fig, ax = plt.subplots()
        fraud_trend.plot(ax=ax, marker="o", linestyle="-", color="red")
        ax.set_title("Fraudulent Transactions Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Fraud Cases")
        st.pyplot(fig)
        plt.close(fig)

# Feature Importance (Fix)
try:
    feature_names = df.drop(columns=["isfraud"]).columns
    feature_importance = pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)

    st.subheader("🔍 Feature Importance")
    
    if feature_importance.empty:
        st.error("⚠️ Feature importance calculation failed. Check model training.")
    else:
        st.write("Feature Importance Values:", feature_importance.head(10))  # Debugging print
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis", ax=ax)
        ax.set_title("Feature Importance in Fraud Detection")
        st.pyplot(fig)
        plt.close(fig)

except Exception as e:
    st.error(f"⚠️ Error in Feature Importance: {e}")

st.write("---")  # Force layout break



