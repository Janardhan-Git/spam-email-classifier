 import streamlit as st
from src.predict import predict_email

st.set_page_config(page_title="Spam Email Classifier")
st.title("📧 Spam Email Classifier")
text = st.text_area("Enter email content:")

if st.button("Predict"):
    if text.strip():
        result = predict_email(text)
        st.success(f"Result: **{result}**")
    else:
        st.warning("Please enter some text.")
