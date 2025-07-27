import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Paths
DATA_PATH = "data/processed/loan_data_cleaned.csv"
MODEL_PATH = "models/model.pkl"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please clean data first.")
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess(df):
    # Encode categorical features
    df = df.copy()
    df = df.drop(columns=["Applicant_ID", "Notes"])  # Drop unneeded/unstructured fields

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["Job_Type", "Loan_Type", "Existing_Loan"], drop_first=True)

    # Encode label
    df["Loan_Status"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("📥 Loading data...")
    df = load_data()

    print("🧹 Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess(df)

    print("🧠 Training model...")
    model = train_model(X_train, y_train)

    print("📈 Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(f"💾 Saving model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Save the trained model
    joblib.dump(model, "models/model.pkl")
    print("✅ Model saved to models/model.pkl")

    # Save the feature columns used in training
    joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")
    print("✅ Feature columns saved to models/feature_columns.pkl")


    print("✅ Training complete.")

