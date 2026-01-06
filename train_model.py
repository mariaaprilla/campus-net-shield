import pandas as pd
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("dataset/malicious_urls.csv")
df.columns = ["url", "label"]

# 2. Mapping label
df["label"] = df["label"].map({
    "benign": 0,
    "phishing": 1,
    "malware": 1,
    "defacement": 1
})
df.dropna(inplace=True)

# 3. Feature engineering
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

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/url_detector.pkl")
print("\nModel saved.")
