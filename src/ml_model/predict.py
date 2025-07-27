import pandas as pd
import joblib
import os

MODEL_PATH = "models/model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"

def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Trained model not found. Run train_model.py first.")
    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError("❌ Feature column file not found. Ensure you saved it during training.")
    
    model = joblib.load(MODEL_PATH)
    expected_cols = joblib.load(FEATURE_PATH)
    return model, expected_cols

def preprocess_input(data: dict, expected_cols: list) -> pd.DataFrame:
    """
    Accepts raw input dict and returns a padded/preprocessed DataFrame with all expected columns.
    """
    df = pd.DataFrame([data])

    # One-hot encode categorical columns
    df = pd.get_dummies(df)

    # Add missing columns as 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order matches training
    df = df[expected_cols]
    return df

def predict(input_data: dict):
    model, expected_cols = load_model_and_features()
    X = preprocess_input(input_data, expected_cols)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # Probability of class "Approved" (1)

    return {
        "prediction": "Approved" if pred == 1 else "Rejected",
        "confidence": round(prob * 100, 2)
    }

if __name__ == "__main__":
    # Sample test case
    sample_input = {
        "Age": 34,
        "Income": 60000,
        "Credit_Score": 720,
        "Loan_Amount": 25000,
        "Job_Type": "Self-Employed",
        "Loan_Type": "Personal",
        "Existing_Loan": "No"
    }

    result = predict(sample_input)
    print(f"✅ Prediction: {result['prediction']} (Confidence: {result['confidence']}%)")
