import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

X = data["message"]
y = data["label"].map({"ham": 0, "spam": 1})

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

joblib.dump(model, "app/spam_model.pkl")
joblib.dump(vectorizer, "app/vectorizer.pkl")
