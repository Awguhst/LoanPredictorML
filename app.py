import streamlit as st
import pandas as pd
import pickle
import sqlite3
import os
from sqlalchemy import create_engine

st.markdown("""
    <style>
        .stButton > button {
            background-color: #99aab5;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stSlider > div {
            margin-top: 10px;
        }
        .stTextInput input {
            font-size: 14px;
        }
        .stDataFrame {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to predict RiskScore and LoanApproved
def predict_risk_approval(dataframe):
    features = dataframe[['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'LoanAmount', 
                          'LoanDuration', 'EducationLevel', 'NumberOfDependents', 'MonthlyDebtPayments', 
                          'CreditCardUtilizationRate', 'DebtToIncomeRatio', 'BankruptcyHistory', 
                          'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory', 'LoanPurpose']]
    predicted_risk = risk_model.predict(features)
    predicted_approval = approval_model.predict(features)
    dataframe['PredictedRiskScore'] = predicted_risk
    dataframe['PredictedLoanApproved'] = predicted_approval
    return dataframe

# Function to save the data to the database with filtering options
def save_to_db(dataframe, risk_threshold, approval_status):
    if approval_status == "approved":
        filtered_df = dataframe[(dataframe['PredictedLoanApproved'] == 1) & 
                                (dataframe['PredictedRiskScore'] <= risk_threshold)]
    elif approval_status == "rejected":
        filtered_df = dataframe[(dataframe['PredictedLoanApproved'] == 0) & 
                                (dataframe['PredictedRiskScore'] <= risk_threshold)]
    else:  # "both" approval status
        filtered_df = dataframe[dataframe['PredictedRiskScore'] <= risk_threshold]

    if not filtered_df.empty:
        conn = sqlite3.connect('loan_data.db')  
        filtered_df.to_sql('loan_data', conn, if_exists='append', index=False)
        conn.close()
        st.success(f"Filtered data has been saved to the database with risk threshold {risk_threshold}.")
    else:
        st.warning("No loans met the criteria based on the selected filters.")

# Load the pre-trained models using pickle
with open('./models/risk_predictor.pkl', 'rb') as f:
    risk_model = pickle.load(f)  

with open('./models/loan_predictor.pkl', 'rb') as f:
    approval_model = pickle.load(f)  
        
# Streamlit interface configuration
st.set_page_config(page_title="Loan Data Prediction and Management", layout="wide", page_icon="")

st.markdown("""
<h1 style='text-align: center; color: #99aab5;'>Loan Approval & Risk Predictor</h1>
<p style='text-align: center; font-size: 18px; color: #BDC3C7;'>
    Discover loan eligibility and receive a quick risk assessment based on financial information.
</p>
""", unsafe_allow_html=True)

# Tabs for CSV preview and SQlite database 
st.sidebar.header("Navigation")

# Create radio buttons for selection 
tabs = st.sidebar.radio(
    "Choose an option",  
    options=["Upload CSV", "SQlite Database"],  
    index=0,  
    help="Select a tab to navigate between" 
)

if tabs == "Upload CSV":

    # Define the path to the example CSV file
    file_path = "./data/example_loan_data.csv"

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the contents of the CSV file
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Display the download button
        st.sidebar.subheader("Example CSV", help="You can download an example CSV to get started")
        st.sidebar.download_button(
            label="Download Example CSV",  
            data=file_data,               
            file_name="example_loan_data.csv",  
            mime="text/csv"               
        )
    else:
        st.sidebar.error(f"File '{file_path}' not found. Please make sure the file is in the correct directory.")
        
        
    uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Ensure the necessary columns are there
        required_columns = ['Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'LoanAmount', 
                            'LoanDuration', 'EducationLevel', 'NumberOfDependents', 'MonthlyDebtPayments', 
                            'CreditCardUtilizationRate', 'DebtToIncomeRatio', 'BankruptcyHistory', 
                            'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory', 'LoanPurpose']
        
        if all(col in df.columns for col in required_columns):
            # Display the CSV preview
            st.subheader("CSV Preview")
            st.dataframe(df)  # Show the dataframe as a table

            # Predict risk score and loan approval
            st.markdown("**Predicting Loan Risk and Approval...**")
            with st.spinner("Processing predictions..."):
                df_with_predictions = predict_risk_approval(df)

            # Allow the user to select the approval status they want to save
            approval_status = st.radio(
                "Choose loan approval status to save:",
                options=["both", "approved", "rejected"]
            )

            # Save the data to the database with an option for threshold adjustment
            risk_threshold = st.slider("Set Risk Threshold for saving to DB", 0.0, 100.0, 50.0)
            if st.button("Save Data to Database"):
                save_to_db(df_with_predictions, risk_threshold, approval_status)

        else:
            st.error("CSV is missing required columns.")

elif tabs == "SQlite Database":
    # Add a section for filtering and sorting data from the database
    st.header("Filter and Sort SQlite Database")

    # Set up SQLAlchemy connection to fetch data
    engine = create_engine('sqlite:///loan_data.db')

    # Initialize filters
    filter_conditions = []

    # Filter by Loan Amount
    min_loan_amount = st.sidebar.number_input("Minimum Loan Amount", min_value=0, value=0)
    max_loan_amount = st.sidebar.number_input("Maximum Loan Amount", min_value=0, value=1000000)
    filter_conditions.append(f"LoanAmount BETWEEN {min_loan_amount} AND {max_loan_amount}")

    # Filter by Approval Status 
    approval_status = st.sidebar.selectbox("Approval Status", options=["Both", "Approved", "Rejected"])
    if approval_status != "Both":
        approval_value = 1 if approval_status == "Approved" else 0
        filter_conditions.append(f"PredictedLoanApproved = {approval_value}")

    # Filter by Risk Score Range
    min_risk_score = st.sidebar.slider("Minimum Risk Score", 0.0, 100.0, 0.0)
    max_risk_score = st.sidebar.slider("Maximum Risk Score", 0.0, 100.0, 100.0)
    filter_conditions.append(f"PredictedRiskScore BETWEEN {min_risk_score} AND {max_risk_score}")

    # Filter by Employment Status
    employment_status = st.sidebar.selectbox("Employment Status", options=["Both", "Employed", "Self-Employed", "Unemployed"])
    if employment_status != "Both":
        filter_conditions.append(f"EmploymentStatus = '{employment_status}'")

    # Filter by Loan Purpose 
    loan_purpose = st.sidebar.selectbox("Loan Purpose", options=[
        "Both", "Debt Consolidation", "Auto", "Home", "Education", "Other"
    ])
    if loan_purpose != "Both":
        filter_conditions.append(f"LoanPurpose = '{loan_purpose}'")

    # Create the final query with filters
    base_query = "SELECT * FROM loan_data"
    if filter_conditions:
        base_query += " WHERE " + " AND ".join(filter_conditions)

    # Execute the query and handle empty tables 
    try:
        df_db = pd.read_sql(base_query, engine)
        if df_db.empty:
            st.warning("No records found matching your filters.")
        else:
            # Display the filtered and sorted data
            st.write(f"Displaying {len(df_db)} records that match your filters:")
            st.dataframe(df_db)
    except Exception as e:
        st.error(f"An error occurred while querying the database: {e}")

    if st.button('Clear All Rows from Table'):
        try:
            # Perform delete operation
            with sqlite3.connect("loan_data.db") as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM loan_data;")
                conn.commit()
            st.success("All rows have been cleared from the table.")
            
        except Exception as e:
            st.error(f"An error occurred while clearing the table: {e}")
    
    
