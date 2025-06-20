import joblib

# Load model and vectorizer
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
model = joblib.load('model/spam_classifier.pkl')

def predict_email(text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)[0]
    return "Spam" if prediction == 1 else "Not Spam"

def predict_proba(text):
    text_transformed = vectorizer.transform([text])
    proba = model.predict_proba(text_transformed)[0]
    return max(proba) * 100  # Return highest class confidence as a percentage
