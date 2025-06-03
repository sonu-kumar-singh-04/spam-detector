import joblib

model = joblib.load("app/spam_model.pkl")
vectorizer = joblib.load("app/vectorizer.pkl")

def predict_spam(text: str) -> str:
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return "Spam" if prediction == 1 else "Ham"
