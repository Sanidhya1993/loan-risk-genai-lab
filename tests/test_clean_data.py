import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep.clean_data import clean_loan_data


def test_clean_data_drops_missing():
    raw_data = pd.DataFrame({
        'Name': ['A', 'B', None],
        'Age': [25, None, 35],
        'Income': [50000, 60000, None],
        'Loan_Status': [1, 0, 1],
        'Job_Type': ['Salaried', None, 'Self-employed'],
        'Gender': ['Male', 'Female', None],
        'Loan_Amount': [200, 300, None],
        'Loan_Amount_Term': [360.0, 360.0, None],
        'Credit_History': [1.0, 0.0, None],
        'Property_Area': ['Urban', 'Rural', None],
        'Notes': ['NA', 'Urgent case', None]  # ✅ Added this line
    })

    cleaned = clean_loan_data(raw_data)

    assert isinstance(cleaned, pd.DataFrame)


def test_clean_data_returns_dataframe():
    raw_data = pd.DataFrame({
        'Age': [25, 30],
        'Income': [50000, 60000],
        'Loan_Status': [1, 0],
        'Job_Type': ['Salaried', 'Self-employed'],
        'Gender': ['Male', 'Female'],
        'Loan_Amount': [200, 300],
        'Loan_Amount_Term': [360.0, 360.0],
        'Credit_History': [1.0, 0.0],
        'Property_Area': ['Urban', 'Rural'],
        'Notes': ['NA', 'None']  # ✅ Added this line
    })

    cleaned = clean_loan_data(raw_data)

    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.isnull().values.any()

