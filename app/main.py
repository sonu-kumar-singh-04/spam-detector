from fastapi import FastAPI
from app.schema import Message
from app.utils import predict_spam

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "Spam Classifier API is running"}

@app.post("/predict")
def classify_message(msg: Message):
    result = predict_spam(msg.text)
    return {"prediction": result}
