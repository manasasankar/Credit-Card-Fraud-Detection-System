
# 💳 Credit Card Fraud Detection System

A Streamlit web app that detects fraudulent credit card transactions using Machine Learning.
**Live Demo:** [Credit Card Fraud Detection App](https://creditcardfrauddetecting.streamlit.app/)  
---

## 🚀 Features

* Upload & explore transaction dataset (EDA with charts)
* Preprocess data (encoding + scaling)
* Train models (Random Forest & XGBoost)
* Evaluate models with metrics & ROC curves
* Predict fraud for:

  * Single transaction (manual input)
  * Bulk transactions (CSV upload)
* Automatically saves & loads trained models

---

## ⚙️ Tech Stack

* Python, Streamlit
* scikit-learn, XGBoost, pandas, numpy, plotly, joblib

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection-System/
│
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

---

## 🖥️ Run Locally

```bash
git clone <your-repo-url>
cd Credit-Card-Fraud-Detection-System
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push to GitHub (without `.venv`)
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → select repo & `app.py`

---

## 📌 Notes

* Include `dataset.csv` or allow user upload
* `.joblib` model files are generated after training
* Use relative paths to avoid file-not-found issues

---

If you want, I can also **add badges (like Python version, Streamlit, License)** to make it look more polished.
Want me to do that?

