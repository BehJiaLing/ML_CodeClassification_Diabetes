import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================
# 1. Load preprocessed dataset
# ===============================
df = pd.read_csv("datasets/datasets_processed.csv")
print(f"Dataset loaded: {df.shape}")

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 2. Load Random Forest & DNN
# ===============================
rf_model = joblib.load("models/rf_diabetes_tuned.pkl")
dnn_model = keras.models.load_model("models/dnn_diabetes_tuned.keras")

print("\n✅ Tuned models loaded successfully!")

# ===============================
# 3. Hybrid Predictions
# ===============================
# RF probability
rf_prob = rf_model.predict_proba(X_test)[:,1]

# DNN probability
dnn_prob = dnn_model.predict(X_test).flatten()

# Average probabilities (simple ensemble)
hybrid_prob = (rf_prob + dnn_prob) / 2
hybrid_pred = (hybrid_prob >= 0.5).astype(int)

# ===============================
# 4. Evaluate Hybrid Model
# ===============================
accuracy = accuracy_score(y_test, hybrid_pred)
precision = precision_score(y_test, hybrid_pred)
recall = recall_score(y_test, hybrid_pred)
f1 = f1_score(y_test, hybrid_pred)
cm = confusion_matrix(y_test, hybrid_pred)

print("\n📊 Hybrid Model Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hybrid RF+DNN")
plt.show()
