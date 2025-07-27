import pandas as pd
import sys
import os


from src.ml_model.train_model import train_model


def test_train_model_creates_model_file():
    # Create a mini dataset
    df = pd.DataFrame({
        'Age': [25, 35, 45, 55],
        'Income': [50000, 60000, 55000, 70000],
        'Loan_Status': [1, 0, 1, 0]
    })

    X = df[['Age', 'Income']]
    y = df['Loan_Status']

    model = train_model(X, y)

    # Check model object
    assert hasattr(model, "predict")

    # Check if model and feature file are saved
    assert os.path.exists("models/model.pkl")
    assert os.path.exists("models/feature_columns.pkl")

