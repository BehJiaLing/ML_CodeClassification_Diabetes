import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Load preprocessed dataset
# ===============================
df = pd.read_csv("datasets/datasets_processed.csv")
print(f"Dataset loaded: {df.shape}")

# ===============================
# 2. Split features & target
# ===============================
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ===============================
# 3. Build Baseline DNN model
# ===============================
def create_baseline_dnn(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # sigmoid for binary classification (output = 0/1)
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',   # use BCE for classification
        metrics=['accuracy']
    )
    return model

baseline_model = create_baseline_dnn(X_train.shape[1])

# ===============================
# 4. Train Baseline DNN
# ===============================
print("\n🏗️ Training baseline DNN model...")
history = baseline_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ===============================
# 5. Evaluate Baseline DNN
# ===============================
y_pred_prob = baseline_model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)  # threshold 0.5

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Baseline DNN Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Baseline DNN")
plt.show()

# ===============================
# 6. Save Baseline DNN model
# ===============================
os.makedirs("models", exist_ok=True)
baseline_model.save("models/dnn_diabetes_baseline.keras")
print("\n💾 Baseline DNN model saved as 'models/dnn_diabetes_baseline.keras'")
