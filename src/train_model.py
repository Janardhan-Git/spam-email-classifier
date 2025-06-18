import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv('data/spam.csv')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(model, 'model/spam_classifier.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
