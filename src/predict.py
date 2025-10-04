# src/predict.py
import joblib
from preprocess import clean_text

tfidf = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/logistic_tfidf.joblib")

def predict_text(text):
    ct = clean_text(text)
    v = tfidf.transform([ct])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(v)[0].max()
    label = model.predict(v)[0]
    label_name = "Real" if label==1 else "Fake"
    return label_name, float(prob) if prob is not None else None

if __name__ == "__main__":
    sample = input("Enter a news headline/text: ")
    label, conf = predict_text(sample)
    print("Prediction:", label, "Confidence:", conf)
