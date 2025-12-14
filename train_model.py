import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Toddler Autism dataset July 2018.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ---------------- TARGET ----------------
df['Class/ASD Traits'] = df['Class/ASD Traits'].astype(str).str.strip()
df['Class/ASD Traits'] = df['Class/ASD Traits'].map({'Yes': 1, 'No': 0})

# ---------------- FEATURES ----------------
# Normalize Sex
df['Sex'] = df['Sex'].astype(str).str.lower().str.strip()
df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})

# Normalize Jaundice
df['Jaundice'] = df['Jaundice'].astype(str).str.lower().str.strip()
df['Jaundice'] = df['Jaundice'].map({'yes': 1, 'no': 0})

# Normalize Family history
df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].astype(str).str.lower().str.strip()
df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].map({'yes': 1, 'no': 0})

# ---------------- FILL MISSING ----------------
df['Sex'].fillna(0, inplace=True)
df['Jaundice'].fillna(0, inplace=True)
df['Family_mem_with_ASD'].fillna(0, inplace=True)

# Select features
features = [
    'A1','A2','A3','A4','A5','A6','A7',
    'Age_Mons','Sex','Jaundice','Family_mem_with_ASD'
]

X = df[features]
y = df['Class/ASD Traits']

# Final safety: remove any remaining NaNs
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model (LOCAL COMPATIBLE)
joblib.dump(rf, "model/rf_model.pkl")

print("✅ Model trained & saved locally")
