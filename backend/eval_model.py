from datasets import load_dataset
import joblib
from sklearn.metrics import (
    classification_report, hamming_loss, f1_score
)

model = joblib.load("models/sentiment_model.joblib")
mlb   = joblib.load("models/label_binarizer.joblib")

id2name = load_dataset("go_emotions", split="train")\
            .features["labels"].feature.names          
target_names = [id2name[int(i)] for i in mlb.classes_]

ds_test = load_dataset("go_emotions", split="test")
X_test = ds_test["text"]
y_test = mlb.transform(ds_test["labels"])           

y_pred = model.predict(X_test)

print("Micro-F1 :", f1_score(y_test, y_pred, average="micro"))
print("Macro-F1 :", f1_score(y_test, y_pred, average="macro"))
print("Hamming  :", hamming_loss(y_test, y_pred))
print("\nDetailed per-label report ↓")
print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0, digits=3))
