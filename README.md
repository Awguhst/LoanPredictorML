# Loan Management App - Proof of Concept

## Overview

The **Loan Management App** is a proof of concept that uses machine learning models to assess loan applications. By predicting the **risk score** and **approval status** of loan applicants, this app provides a quick and effective way to manage loan data. It allows users to upload loan data, view predictions, and store the results in an SQLite database for further analysis.

This app is built using **XGBoost** and **Optuna** for hyperparameter optimization, providing powerful predictive capabilities for loan risk and approval prediction.

## Features

- **Loan Risk Prediction:** The app predicts two key values:
  - **Risk Score (Regression Model):** A predicted score representing the likelihood of loan repayment.
  - **Loan Approval (Classification Model):** A predicted approval status (approved or rejected).

- **CSV Upload & Prediction:**
  - Upload a CSV file containing loan application data.
  - The app predicts the **Risk Score** and **Loan Approval** status for each loan based on the input features.

- **Database Management:** 
  - Store filtered loan application data in an SQLite database.
  - Filter and sort loan applications based on various parameters such as loan amount, approval status, risk score, and employment status.

- **Risk Threshold & Approval Filtering:**
  - Set a **Risk Threshold** to filter which data is stored in the database.
  - Choose whether to store **approved**, **rejected**, or **both** loan applications.

## Machine Learning Models

### 1. **Loan Risk Prediction (Regression Model)**
The **Loan Risk Prediction** model is a **regression model** trained to predict the **Risk Score** of loan applicants. The model uses financial and personal details from the applicant (such as age, income, credit score, and more) to estimate their risk of defaulting on the loan.

- **Model Type:** XGBoost Regressor
- **Evaluation Metric:**
  - **R² Score:** 0.769
  - **Mean Absolute Error (MAE):** 2.801
  - **Mean Squared Error (MSE):** 14.315
  - **Root Mean Squared Error (RMSE):** 3.784

### 2. **Loan Approval Prediction (Classification Model)**
The **Loan Approval Prediction** model is a **binary classification model** trained to predict whether a loan will be **approved (1)** or **rejected (0)** based on an applicant's data.

- **Model Type:** XGBoost Classifier
- **Evaluation Metrics:**
  - **F1 Score:** 0.81 (macro average)
  - **Precision (Approved = 1):** 0.85
  - **Recall (Approved = 1):** 0.77
  - **Accuracy:** 0.91
  - **Macro Average F1 Score:** 0.87
  - **Weighted Average F1 Score:** 0.91

Both models were trained on synthetic data derived from the [Financial Risk for Loan Approval dataset on Kaggle](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval).

### Hyperparameter Tuning with Optuna
Both models underwent hyperparameter optimization using **Optuna**, an automatic hyperparameter optimization framework. Optuna helps to fine-tune model parameters and improve performance through an efficient search process, ensuring that the models achieve optimal results in terms of predictive accuracy and risk assessment.

## Installation

To run the app locally, follow these steps:

### 1. Install Dependencies

Make sure you have Python 3.7+ installed. Then, install the required Python libraries:

```bash
pip install streamlit pandas scikit-learn xgboost sqlalchemy optuna sqlite3
```

### 2. Download the Pre-trained Models

The app requires two pre-trained models (`risk_predictor.pkl` for regression and `loan_predictor.pkl` for classification). Place these files in the `./models` directory.

- **`risk_predictor.pkl`:** The regression model for predicting loan risk scores.
- **`loan_predictor.pkl`:** The classification model for predicting loan approval.

### 3. Run the App

After installing dependencies and placing the models in the correct directory, start the app by running the following command:

```bash
streamlit run app.py
```
