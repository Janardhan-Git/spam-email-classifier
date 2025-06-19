import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

# Sample dataset
data = pd.read_csv("https://raw.githubusercontent.com/Janardhan-Git/spam-email-classifier/main/data/spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']

# Encode labels: ham -> 0, spam -> 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Train TF-IDF vectorizer and Naive Bayes model
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Create folder if not exists
os.makedirs("model", exist_ok=True)

# Save vectorizer and model
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(model, "model/spam_classifier_model.pkl")

print("✅ Model and vectorizer saved in 'model/' folder.")
