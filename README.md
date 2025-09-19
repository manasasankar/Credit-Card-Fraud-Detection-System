
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

**ML algorithms Algorithms**:

* **Random Forest Classifier**

  * Ensemble of many decision trees
  * Votes from all trees decide the final class (fraud / not fraud)

* **XGBoost Classifier**

  * Gradient boosting algorithm
  * Builds trees sequentially, each correcting errors of the previous

It compares both and selects the one with **higher ROC-AUC score** as the best model.


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
git clone <repo-url>
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

