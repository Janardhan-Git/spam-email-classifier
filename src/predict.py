import joblib

def predict_email(text):
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    model = joblib.load('model/spam_classifier.pkl')
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    user_input = input("Enter email text: ")
    print("Prediction:", predict_email(user_input))
