import streamlit as st
import pandas as pd
import joblib

# Load the trained model/pipeline
model = joblib.load("fraud_detection_pipeline.pkl")

st.title("🚨 Fraud Detection Prediction App")

st.markdown("Enter transaction details below to check if it is fraudulent.")

st.divider()

# Input Fields
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)  # FIXED

# Prediction Button
if st.button("🔍 Predict"):
    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.subheader(f"Prediction: {int(prediction)}")

        if prediction == 1:
            st.error("🚫 The transaction is predicted to be **FRAUDULENT**.")
        else:
            st.success("✅ The transaction is predicted to be **NOT FRAUDULENT**.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        st.info("Make sure the model expects these input columns.")
