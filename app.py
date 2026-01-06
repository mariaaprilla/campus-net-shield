import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Campus-Net Shield", layout="centered")
st.title("🔒 Campus-Net Shield")
st.write("Deteksi URL berbahaya menggunakan AI (Random Forest)")

@st.cache_resource
def train_model():
    # LOAD DATASET KECIL
    df = pd.read_csv("malicious_urls_small.csv")
    df.columns = ["url", "label"]

    # LABEL ENCODING
    df["label"] = df["label"].map({
        "benign": 0,
        "phishing": 1,
        "malware": 1,
        "defacement": 1
    })
    df.dropna(inplace=True)

    # FEATURE ENGINEERING
    def extract_features(url):
        return {
            "url_length": len(url),
            "has_https": 1 if "https" in url else 0,
            "count_digits": sum(c.isdigit() for c in url),
            "count_special": len(re.findall(r"[^\w]", url)),
            "has_gambling_keyword": 1 if any(
                k in url.lower() for k in ["bet", "slot", "casino", "togel"]
            ) else 0
        }

    X = pd.DataFrame(df["url"].apply(extract_features).tolist())
    y = df["label"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# TRAIN MODEL SEKALI
model = train_model()

# ================= UI =================
url_input = st.text_input("Masukkan URL:")

if st.button("Cek URL"):
    if url_input:
        features = pd.DataFrame([{
            "url_length": len(url_input),
            "has_https": 1 if "https" in url_input else 0,
            "count_digits": sum(c.isdigit() for c in url_input),
            "count_special": len(re.findall(r"[^\w]", url_input)),
            "has_gambling_keyword": 1 if any(
                k in url_input.lower() for k in ["bet", "slot", "casino", "togel"]
            ) else 0
        }])

        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("⚠️ URL BERBAHAYA")
        else:
            st.success("✅ URL AMAN")
    else:
        st.warning("Masukkan URL terlebih dahulu")
