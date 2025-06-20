import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict import predict_email, predict_proba

st.set_page_config(page_title="Spam Email Classifier", layout="centered")

st.title("📧 Spam Email Classifier")
st.markdown("Enter email content below to predict if it's spam or not.")

# Predefined sample emails
sample_emails = {
    "🔒 Bank Alert": "Dear user, your account has been suspended. Click here to verify your account immediately.",
    "🎉 Lottery Winner": "Congratulations! You've won $1,000,000. Click here to claim now!",
    "📦 Order Confirmation": "Your Amazon order has been shipped. Track it here.",
    "🧾 Invoice": "Your invoice for the last month is attached. Please review and process payment.",
}

# Select sample email
option = st.selectbox("Or choose a sample email:", ["(None)"] + list(sample_emails.keys()))
selected_text = sample_emails.get(option, "") if option != "(None)" else ""

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []

if "text_input" not in st.session_state:
    st.session_state.text_input = selected_text

# Text area input
st.session_state.text_input = st.text_area("✉️ Enter Email Content:",
                                           value=st.session_state.text_input,
                                           height=200)

# Predict button
if st.button("🔍 Predict"):
    text = st.session_state.text_input
    if not text.strip():
        st.warning("Please enter or select an email message.")
    else:
        result = predict_email(text)
        prob = predict_proba(text)

        # Store result
        st.session_state.predictions.append(result)

        # Show result
        if result == "Spam":
            st.error(f"🛑 This is SPAM (Confidence: {prob:.2f}%)")
        else:
            st.success(f"✅ This is NOT Spam (Confidence: {prob:.2f}%)")

# Show charts and download if predictions exist
if st.session_state.predictions:
    df = pd.DataFrame(st.session_state.predictions, columns=["Prediction"])
    st.subheader("📊 Prediction History")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bar Chart")
        count_df = df["Prediction"].value_counts().reset_index()
        count_df.columns = ["Prediction", "Count"]
        bar_fig = px.bar(count_df, x="Prediction", y="Count",
                         color="Prediction",
                         color_discrete_map={"Spam": "crimson", "Not Spam": "green"})
        st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        st.markdown("#### Pie Chart")
        pie_fig = px.pie(df, names="Prediction",
                         color="Prediction",
                         color_discrete_map={"Spam": "crimson", "Not Spam": "green"})
        st.plotly_chart(pie_fig, use_container_width=True)

    # Clear button
    if st.button("🧹 Clear Prediction History"):
        st.session_state.predictions = []
        st.session_state.text_input = ""

    # CSV download
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction History as CSV",
        data=csv_bytes,
        file_name='prediction_history.csv',
        mime='text/csv',
    )