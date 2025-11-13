import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

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
# 3. Define baseline Random Forest
# ===============================
rf_base = RandomForestClassifier(random_state=42)

# ===============================
# 4. Define hyperparameter search space
# ===============================
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# ===============================
# 5. Randomized Search with 3-fold CV
# ===============================
# for hyperparameter tuning with random sampling and cross-validation
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,           # Test 20 random combinations
    cv=3,                # 3-fold cross-validation; splits the dataset into k equal-sized folds, trained on k-n folds and tested on the remaining fold
    scoring='f1',        # Optimize for F1-score
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\n🔍 Starting hyperparameter tuning...")
rf_random.fit(X_train, y_train)

best_rf = rf_random.best_estimator_
print("\n✅ Tuning complete!")
print("Best Hyperparameters:", rf_random.best_params_)
print("Best CV F1-score:", rf_random.best_score_)

# ===============================
# 6. Evaluate Tuned Model
# ===============================
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Tuned Random Forest Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned RF")
plt.show()

# ===============================
# 7. Save Tuned Model
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf, "models/rf_diabetes_tuned.pkl")
print("\n💾 Tuned Random Forest model saved as 'models/rf_diabetes_tuned.pkl'")
