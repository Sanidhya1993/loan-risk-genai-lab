import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/loan_data_dirty.csv"
PROCESSED_PATH = "data/processed/loan_data_cleaned.csv"

def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the loan dataset.
    """
    # Fix job title typos
    job_map = {
        "salried": "Salaried",
        "Self Emp": "Self-Employed",
        "unemployd": "Unemployed",
        None: "Unknown"
    }
    df["Job_Type"] = df["Job_Type"].replace(job_map)
    df["Job_Type"] = df["Job_Type"].fillna("Unknown")

    # Fix loan amount (convert strings like "Fifty Thousand" to NaN)
    def parse_amount(x):
        try:
            return float(x)
        except:
            return np.nan

    df["Loan_Amount"] = df["Loan_Amount"].apply(parse_amount)

    # Fill missing numerical values with median
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df["Loan_Amount"] = df["Loan_Amount"].fillna(df["Loan_Amount"].median())

    # Clean and standardize 'Notes'
    df["Notes"] = df["Notes"].fillna("").str.strip()
    df["Notes"] = df["Notes"].replace(r"^\s*$", "No notes provided", regex=True)

    return df

if __name__ == "__main__":
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"{RAW_PATH} not found. Please add the raw CSV file first.")

    print(f"📥 Loading raw data from {RAW_PATH}...")
    df_raw = pd.read_csv(RAW_PATH)

    print("🧹 Cleaning data...")
    df_cleaned = clean_loan_data(df_raw)

    print(f"💾 Saving cleaned data to {PROCESSED_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_cleaned.to_csv(PROCESSED_PATH, index=False)

    print("✅ Data cleaning complete.")


