import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import keras_tuner as kt
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ===============================
# 2. Define model builder for KerasTuner
# ===============================
# random hyperparameters (layers, units, learning rate)
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Tune number of hidden layers (1-3)
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
            activation='relu'
        ))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Tune learning rate
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===============================
# 3. Initialize KerasTuner
# ===============================
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',  # maximize validation accuracy
    max_trials=10,              # 10 different hyperparameter combinations
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_logs',
    project_name='dnn_diabetes'
)

# ===============================
# 4. Run tuner
# ===============================
print("\n🔍 Starting hyperparameter tuning...")
tuner.search(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\n✅ Best Hyperparameters:")
for key, value in best_hp.values.items():
    print(f"{key}: {value}")

# ===============================
# 5. Train best tuned model
# ===============================
best_model = tuner.hypermodel.build(best_hp)
history = best_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ===============================
# 6. Evaluate tuned DNN
# ===============================
y_pred_prob = best_model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Tuned DNN Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion matrix plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned DNN")
plt.show()

# ===============================
# 7. Save tuned model
# ===============================
os.makedirs("models", exist_ok=True)
best_model.save("models/dnn_diabetes_tuned.keras")
print("\n💾 Tuned DNN model saved as 'models/dnn_diabetes_tuned.keras'")
