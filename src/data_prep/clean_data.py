import pandas as pd
import numpy as np

def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the loan data.
    """
    # Fix Job_Type typos
    job_map = {
        "salried": "Salaried",
        "Self Emp": "Self-Employed",
        "unemployd": "Unemployed",
        None: "Unknown"
    }
    df["Job_Type"] = df["Job_Type"].replace(job_map)

    # Clean Loan_Amount: handle string values like "Fifty Thousand"
    def parse_amount(x):
        if isinstance(x, str) and x.replace(",", "").isdigit():
            return float(x.replace(",", ""))
        try:
            return float(x)
        except:
            return np.nan

    df["Loan_Amount"] = df["Loan_Amount"].apply(parse_amount)

    # Fill missing income with median
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # Clean Notes field
    df["Notes"] = df["Notes"].fillna("").str.strip()

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/loan_data_dirty.csv")
    df_cleaned = clean_loan_data(df)
    df_cleaned.to_csv("data/processed/loan_data_cleaned.csv", index=False)
    print("✅ Cleaned data saved.")

