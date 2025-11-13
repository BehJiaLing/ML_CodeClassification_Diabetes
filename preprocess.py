import pandas as pd
from sklearn.preprocessing import StandardScaler # scale numerical features
from sklearn.impute import SimpleImputer # fill missing values.
from imblearn.over_sampling import SMOTE # (Synthetic Minority Oversampling Technique) to handle class imbalance

# ====================================================
# Step 1: Load dataset
# ====================================================
df = pd.read_csv("datasets.csv")
print("Original Shape:", df.shape)

# ====================================================
# Step 2: Remove duplicates
# ====================================================
before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed duplicates: {before - df.shape[0]} rows")

# ====================================================
# Step 3: Encode Categorical Columns
# ====================================================
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})

smoking_map = {
    'never': 0,
    'former': 1,
    'current': 2,
    'ever': 3,
    'not current': 4
}
df['smoking_history'] = df['smoking_history'].map(smoking_map)

# ====================================================
# Step 4: Feature Engineering - Age Group
# ====================================================
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]
)

# ====================================================
# Step 5: Handle Missing Values
# ====================================================
from sklearn.impute import SimpleImputer

num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'age_group']

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("✅ Missing values handled")


# ====================================================
# Step 6: Scale numerical features
# ====================================================
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nScaled numerical columns:")
print(df[num_cols].head())

# ====================================================
# Step 7: Handle Imbalance Using SMOTE
# ====================================================
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X, y)

print("After SMOTE Shape:", X_bal.shape)

# ====================================================
# Step 8: Export cleaned dataset
# ====================================================
processed_df = pd.concat([X_bal, y_bal], axis=1)
processed_df.to_csv("datasets/datasets_processed.csv", index=False)

print("✅ Preprocessing complete. Saved as 'datasets_processed.csv'")
print("Final shape:", processed_df.shape)
