# 📩 SMS Spam Classifier API (FastAPI)

This project is a simple machine learning-powered API to classify SMS messages as `Spam` or `Ham` using `FastAPI`, `CountVectorizer`, and `Multinomial Naive Bayes`.

## 🧠 How it Works

- Preprocesses SMS text using `CountVectorizer` (converts text to numerical features).
- Trained on UCI SMS Spam Collection dataset.
- Uses `MultinomialNB`, a probabilistic classifier suitable for text data.

## 🔧 Setup Instructions

### Clone the repo & install dependencies

```bash
git clone https://github.com/sonu-kumar-singh-04/spam-detector.git
cd spam-detector
pip install -r requirements.txt
```
### Download dataset:
- Download tadaset from https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
- Place `SMSSpamCollection` file inside the `data/` folder.

### Train model:
```
python app/train_model.py
```

### Start FastAPI server:
```
uvicorn app.main:app --reload
```

### Example Usage:
```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text": "Congratulations! You've won a free ticket"}'
```
#### Response:
```
{"prediction": "Spam"}
```

### ✨ Technologies Used:
 - FastAPI
 - Scikit-learn
 - CountVectorizer
 - MultinomialNB
 - Joblib

 ### 🙋 Why CountVectorizer?
 Converts text messages to a matrix of token counts — this helps the model understand frequency-based features (e.g., how many times words like "free", "win", "cash" appear).

 ### 📈 Why MultinomialNB?
 Best suited for discrete features like word counts. It calculates probabilities of each class (spam/ham) using Bayes’ theorem.

 👤 Author
 
 Sonu Kumar Singh