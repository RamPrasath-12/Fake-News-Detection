# src/features.py
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data/processed/cleaned.csv")
texts = df['clean_text'].fillna("").tolist()

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = tfidf.fit_transform(texts)

joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
print("Saved TF-IDF vectorizer. Shape:", X.shape)
