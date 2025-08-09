import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("water_potability.csv")  # Replace with your dataset file name

# Optional: drop rows with missing values
df = df.dropna()

# Feature selection
features = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']  # Modify based on your data
X = df[features]
y = df['Potability']  # Target variable (0 = Not safe, 1 = Safe)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
