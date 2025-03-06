# 🔍 Credit Card Fraud Detection System  
**Live Demo:** [Credit Card Fraud Detection App](https://creditcardfrauddetectionprediction.streamlit.app/)  

---

## 📌 Overview  
This **Credit Card Fraud Detection System** uses **Logistic Regression** to classify transactions as **fraudulent or legitimate** based on historical data. The app is built using **Streamlit** and **scikit-learn**, providing an intuitive interface for users to input transaction details and detect fraud in real-time.  

---

## 🚀 Features  
✔️ **Machine Learning Model** – Trained using Logistic Regression  
✔️ **User-Friendly Interface** – Built with Streamlit  
✔️ **Real-Time Fraud Prediction** – Detect fraud based on transaction inputs  
✔️ **Dynamic Input Handling** – Supports categorical & numerical fields  
✔️ **Visual Feedback** – Displays results with warning messages & model accuracy  

---

## 📂 Dataset  
The model is trained on a dataset (`dataset.csv`) with transaction details, including:  
- **Numerical Features**: (e.g., Amount, Transaction Time)  
- **Categorical Features**: (e.g., Payment Method, Merchant)  
- **Target Column**: `isfraud` (1 = Fraud, 0 = Legitimate)  

---

## 🛠️ Installation & Setup  
1️⃣ Clone the repository:  
```sh
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
