# 📩 SMS Spam Classifier API (FastAPI)

This project is a simple machine learning-powered API to classify SMS messages as `Spam` or `Ham` using `FastAPI`, `CountVectorizer`, and `Multinomial Naive Bayes`.

## 🧠 How it Works

- Preprocesses SMS text using `CountVectorizer` (converts text to numerical features).
- Trained on UCI SMS Spam Collection dataset.
- Uses `MultinomialNB`, a probabilistic classifier suitable for text data.

## 🔧 Setup Instructions

### Clone the repo & install dependencies

```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
pip install -r requirements.txt
