# src/train.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/processed/cleaned.csv")
X_text = df['clean_text'].fillna("")
y = df['label'].values

tfidf = joblib.load("models/tfidf_vectorizer.joblib")
X = tfidf.transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

# save model
joblib.dump(model, "models/logistic_tfidf.joblib")
print("Saved model at models/logistic_tfidf.joblib")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic TFIDF")
plt.savefig("models/confusion_matrix.png", bbox_inches='tight')
plt.close()
