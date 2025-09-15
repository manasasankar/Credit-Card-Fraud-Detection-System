# app.py
import os
import joblib
import shap
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import xgboost as xgb

# ---------------------- Config ----------------------
st.set_page_config(page_title="Credit Card Fraud Detection (Resume-ready)", layout="wide")
DATA_PATH = "dataset.csv"
RF_FILE = "rf_model.joblib"
XGB_FILE = "xgb_model.joblib"
SCALER_FILE = "scaler.joblib"
ENC_FILE = "encoder.joblib"
# ----------------------------------------------------

st.title("💳 Fraud Detection — Interactive Dashboard (Resume-ready)")

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df

def preprocess(df, encoder=None, scaler=None, fit=False):
    df = df.copy()
    # parse dates if present
    if "transactiondate" in df.columns:
        df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
        df["txn_day"] = df["transactiondate"].dt.day
        df["txn_hour"] = df["transactiondate"].dt.hour
        df = df.drop(columns=["transactiondate"])
    # identify target
    if "isfraud" in df.columns:
        y = df["isfraud"]
        X = df.drop(columns=["isfraud"])
    else:
        y = None
        X = df
    # categorical
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if fit:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if cat_cols:
            encoder.fit(X[cat_cols].fillna("NA"))
    if encoder is not None and cat_cols:
        X[cat_cols] = encoder.transform(X[cat_cols].fillna("NA"))
    # scaling
    if fit:
        scaler = StandardScaler()
        scaler.fit(X[num_cols])
    if scaler is not None and num_cols:
        X[num_cols] = scaler.transform(X[num_cols])
    return X, y, encoder, scaler, cat_cols, num_cols

def train_and_save_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=2, n_jobs=-1)
    rf.fit(X_train, y_train)
    # XGBoost
    xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=2, n_jobs=-1)
    xg.fit(X_train, y_train)
    # Save models
    joblib.dump(rf, RF_FILE)
    joblib.dump(xg, XGB_FILE)
    return rf, xg, X_test, y_test

def metrics_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, roc=roc, cm=cm, fpr=fpr, tpr=tpr)

# ---------- Load data ----------
st.sidebar.header("Data & Model")
if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
    st.sidebar.success(f"Loaded dataset: {DATA_PATH} ({df.shape[0]} rows, {df.shape[1]} cols)")
else:
    st.sidebar.error(f"{DATA_PATH} not found. Upload below or place dataset.csv in app folder.")
    df = None

# ---------- Model load or train ----------
models_present = os.path.exists(RF_FILE) and os.path.exists(XGB_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(ENC_FILE)
st.sidebar.markdown(f"**Pretrained models present:** {'✅' if models_present else '❌'}")

if df is not None:
    if models_present:
        encoder = joblib.load(ENC_FILE)
        scaler = joblib.load(SCALER_FILE)
        rf = joblib.load(RF_FILE)
        xg = joblib.load(XGB_FILE)
        X_all, y_all, _, _, cat_cols, num_cols = preprocess(df, encoder=encoder, scaler=scaler, fit=False)
        st.sidebar.success("Loaded models & preprocessors from disk.")
    else:
        st.sidebar.info("No pretrained models found. You can train models (1 button) and save them.")
        if st.sidebar.button("Train models & save (may take a while)"):
            with st.spinner("Training models..."):
                X_all, y_all, encoder, scaler, cat_cols, num_cols = preprocess(df, encoder=None, scaler=None, fit=True)
                rf, xg, X_test, y_test = train_and_save_models(X_all, y_all)
                joblib.dump(encoder, ENC_FILE)
                joblib.dump(scaler, SCALER_FILE)
                st.success("Models and preprocessors trained & saved.")
                models_present = True
        else:
            # allow continuing without models
            X_all, y_all, _, _, cat_cols, num_cols = preprocess(df, encoder=None, scaler=None, fit=True)

# ---------------- EDA ----------------
st.header("📊 Exploratory Data Analysis")
if df is None:
    st.info("Upload dataset to enable EDA and model features.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.lower()
        X_all, y_all, encoder, scaler, cat_cols, num_cols = preprocess(df, encoder=None, scaler=None, fit=True)
else:
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        st.subheader("Data sample")
        st.dataframe(df.head(100))
    with c2:
        st.subheader("Quick stats")
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        if "isfraud" in df.columns:
            fraud_rate = df["isfraud"].mean()
            st.metric("Fraud rate", f"{fraud_rate:.2%}")
    with c3:
        st.subheader("Missing values")
        mv = df.isna().sum()
        mv = mv[mv>0]
        if mv.empty:
            st.write("No missing values")
        else:
            st.dataframe(mv)

    # interactive plots
    if "amount" in df.columns:
        fig = px.histogram(df, x="amount", nbins=50, title="Transaction Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)
    if "location" in df.columns and "isfraud" in df.columns:
        cnt = df.groupby(["location","isfraud"]).size().reset_index(name="count")
        fig2 = px.bar(cnt, x="location", y="count", color="isfraud", title="Transactions by Location & Fraud")
        st.plotly_chart(fig2, use_container_width=True)

# --------------- Model Comparison & Metrics ---------------
st.header("🤖 Model Comparison")
if models_present:
    # prepare test set
    X_full, y_full, _, _, cat_cols, num_cols = preprocess(df, encoder=encoder, scaler=scaler, fit=False)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=2)

    models = {"RandomForest": rf, "XGBoost": xg}
    results = {}
    for name, m in models.items():
        results[name] = metrics_report(m, X_test, y_test)

    cols = st.columns(len(models))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.subheader(name)
            st.metric("Accuracy", f"{res['acc']:.3f}")
            st.metric("Precision", f"{res['prec']:.3f}")
            st.metric("Recall", f"{res['rec']:.3f}")
            st.metric("F1", f"{res['f1']:.3f}")
            st.metric("ROC-AUC", f"{res['roc']:.3f}")

    # ROC curves combined
    st.subheader("ROC Curves")
    roc_df = []
    for name, res in results.items():
        roc_df.append(pd.DataFrame({"fpr": res["fpr"], "tpr": res["tpr"], "model": name}))
    roc_df = pd.concat(roc_df)
    fig = px.line(roc_df, x="fpr", y="tpr", color="model", title="ROC Curves")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for best model (by ROC)
    best_name = max(results.items(), key=lambda x: x[1]["roc"])[0]
    st.markdown(f"**Best model:** {best_name}")
    cm = results[best_name]["cm"]
    cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
    st.dataframe(cm_df)

    
else:
    st.info("No pretrained models loaded. Train models from sidebar or place model files in app folder.")

# ----------------- Prediction UI -----------------
st.header("🔎 Predict Transaction")

with st.expander("Single transaction input"):
    if df is None:
        st.info("Upload dataset first so app knows columns & preprocessing.")
    else:
        input_vals = {}
        for col in df.drop(columns=["isfraud"]).columns:
            if df[col].dtype == "object":
                vals = df[col].dropna().unique().tolist()
                default = vals[0] if vals else ""
                input_vals[col] = st.selectbox(f"{col}", vals, index=0)
            else:
                mn, mx = float(df[col].min()), float(df[col].max())
                default = float(df[col].median())
                input_vals[col] = st.number_input(f"{col}", min_value=mn, max_value=mx, value=default)

        if st.button("Predict single transaction"):
            input_df = pd.DataFrame([input_vals])
            if models_present:
                # preprocess using saved encoder/scaler
                input_proc, _, _, _, _, _ = preprocess(input_df, encoder=encoder, scaler=scaler, fit=False)
                preds = {}
                for name, m in models.items():
                    p = m.predict(input_proc)[0]
                    prob = m.predict_proba(input_proc)[:,1][0] if hasattr(m, "predict_proba") else None
                    preds[name] = (p, prob)
                st.write(preds)
                chosen = max(preds.items(), key=lambda x: (x[1][1] if x[1][1] is not None else x[1][0]))
                label, probability = chosen[1]
                if label == 1:
                    st.error(f"⚠️ Detected Fraud (model {chosen[0]}), prob={probability:.3f}")
                else:
                    st.success(f"✅ Legitimate (model {chosen[0]}), prob={probability:.3f}")
            else:
                st.warning("Models not available. Train or load models first.")

with st.expander("Batch predict via CSV"):
    uploaded = st.file_uploader("Upload CSV for batch prediction (must match columns)", type=["csv"], key="batch")
    if uploaded and models_present:
        batch = pd.read_csv(uploaded)
        batch.columns = batch.columns.str.lower()
        X_batch, _, _, _, _, _ = preprocess(batch, encoder=encoder, scaler=scaler, fit=False)
        preds = xg.predict_proba(X_batch)[:,1]
        out = batch.copy()
        out["fraud_prob"] = preds
        st.dataframe(out.head(200))
        st.download_button("Download predictions CSV", out.to_csv(index=False).encode(), "preds.csv")
    elif uploaded:
        st.warning("Models not available to predict. Train or load models first.")

st.markdown("---")
st.write("Made for portfolio: Model comparison, SHAP explainability, CSV upload, and pre-trained fast loading.")
