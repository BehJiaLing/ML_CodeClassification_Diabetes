import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# 1. Load preprocessed dataset
# ===============================
df = pd.read_csv("datasets_processed.csv")
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
# 3. Train Random Forest Classifier
# ===============================
rf_model = RandomForestClassifier(
    n_estimators=200, # 200 trees in the forest
    max_depth=None,
    min_samples_split=2, # minimum samples to split a node
    min_samples_leaf=1, # minimum samples required at a leaf node
    random_state=42 # reproducible results
)

print("\n🌲 Training Random Forest Classifier...")
rf_model.fit(X_train, y_train)

# ===============================
# 4. Make Predictions
# ===============================
y_pred = rf_model.predict(X_test)

# ===============================
# 5. Evaluate Model
# ===============================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred) # how many predicted positives are actually positive
recall = recall_score(y_test, y_pred) # how many actual positives were correctly predicted
f1 = f1_score(y_test, y_pred) # harmonic mean of precision and recall
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ===============================
# 6. Save Trained Model
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/rf_diabetes_model.pkl")
print("\n💾 Random Forest model saved as 'models/rf_diabetes_model.pkl'")
