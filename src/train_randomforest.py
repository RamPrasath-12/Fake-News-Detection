import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/cleaned.csv")
X_text = df['clean_text'].fillna("")
y = df['label'].values

tfidf = joblib.load("models/tfidf_vectorizer.joblib")
X = tfidf.transform(X_text)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Ensure directory exists
os.makedirs("models/random_forest", exist_ok=True)

# Confusion matrix plot
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest TFIDF")
plt.savefig("models/random_forest/confusion_matrix.png", bbox_inches='tight')
plt.close()

joblib.dump(model, "models/randomforest_tfidf.joblib")