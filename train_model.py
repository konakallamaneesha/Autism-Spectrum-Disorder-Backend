import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("Toddler Autism dataset July 2018.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ---------------- TARGET ----------------
df['Class/ASD Traits'] = df['Class/ASD Traits'].astype(str).str.strip()
df['Class/ASD Traits'] = df['Class/ASD Traits'].map({'Yes': 1, 'No': 0})

# ---------------- FEATURES ----------------
df['Sex'] = df['Sex'].astype(str).str.lower().str.strip().map({'m': 1, 'f': 0})
df['Jaundice'] = df['Jaundice'].astype(str).str.lower().str.strip().map({'yes': 1, 'no': 0})
df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].astype(str).str.lower().str.strip().map({'yes': 1, 'no': 0})

# Fill missing
df.fillna(0, inplace=True)

# Features
features = [
    'A1','A2','A3','A4','A5','A6','A7',
    'Age_Mons','Sex','Jaundice','Family_mem_with_ASD'
]

X = df[features]
y = df['Class/ASD Traits']

# ---------------- K-FOLD ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

rf = RandomForestClassifier(n_estimators=100, random_state=42)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))

# Average results
print("Accuracy:", np.mean(accuracy_list))
print("Precision:", np.mean(precision_list))
print("Recall:", np.mean(recall_list))
print("F1 Score:", np.mean(f1_list))

# ---------------- FINAL MODEL (IMPORTANT) ----------------
rf.fit(X, y)  # Train on full data

# Save model
joblib.dump(rf, "model/rf_model.pkl")

print("✅ Final model trained on full dataset & saved")