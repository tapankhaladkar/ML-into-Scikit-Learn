# Intro to Machine Learning with scikit-learn

This repo is my hands-on introduction to scikit-learn, where I implement and evaluate traditional machine learning models from scratch.

## 1. Linear Regression – Medical Insurance Charges

- Dataset: Medical charges (age, BMI, children, smoker, region, etc.)
- Goal: Predict insurance charges using linear regression
- Highlights:
  - Exploratory data analysis and correlation analysis
  - Manual feature engineering (binary encoding, one-hot encoding)
  - Train/test split
  - Model training with `LinearRegression`
  - Evaluation with RMSE

Notebook: `notebooks/01-linear-regression-medical-charges.ipynb`

## 2. Logistic Regression – Rain Prediction

- Dataset: Australian weather dataset (`weatherAUS.csv`)
- Goal: Predict whether it will rain tomorrow (classification)
- Highlights:
  - Train/validation/test split
  - Handling missing values with `SimpleImputer`
  - Scaling numeric features with `MinMaxScaler`
  - One-hot encoding categorical features with `OneHotEncoder`
  - Training `LogisticRegression`
  - Model evaluation with accuracy and confusion matrix

Notebook: `notebooks/02-logistic-regression-weather.ipynb`

## Setup

```bash
git clone https://github.com/<your-username>/ml-intro-scikit-learn.git
cd ml-intro-scikit-learn
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
