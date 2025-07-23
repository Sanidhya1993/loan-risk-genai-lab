# Loan Risk Assessment with GenAI

This project is a hands-on implementation of an end-to-end AI/ML/GenAI pipeline in the Financial Services domain. It predicts loan eligibility and risk, explains decisions using SHAP/LIME, and generates personalized GenAI-based messages.

## 🔍 Use Case
- Predict loan approval status from user financial data
- Explain decision using SHAP (Explainable AI)
- Generate human-readable reason letters using Generative AI

## 📁 Project Structure
- `data/`: Raw and cleaned datasets
- `src/`: Modular Python code
- `models/`: Trained models and artifacts
- `notebooks/`: Exploratory notebooks
- `pipelines/`: MLOps-ready pipelines
- `tests/`: Unit and integration tests
- `.github/workflows/`: CI/CD workflows

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python pipelines/train_pipeline.py

