# src/preprocess.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# first-time: download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r"http\S+", " ", text)     # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove punctuation
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/combined.csv")
    df['clean_text'] = df['text'].astype(str).apply(clean_text)
    df.to_csv("data/processed/cleaned.csv", index=False)
    print("Saved cleaned.csv, rows:", len(df))
    print(df['label'].value_counts())
