# 📦 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
os.makedirs("model", exist_ok=True)

# 🔹 Load Dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(r'C:\coding\MLA\Wine_Quality_Prediction\winequality-red.csv')

# 🔹 Convert Quality Score into Categories
def quality_category(q):
    if q <= 4:
        return 0  # Low
    elif q <= 6:
        return 1  # Medium
    else:
        return 2  # High

df['quality_label'] = df['quality'].apply(quality_category)

# 🔹 Features and Target
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Train Logistic Regression Model with Class Weights
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 🔹 Predictions
y_pred = model.predict(X_test_scaled)

# 🔹 Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔹 Show which classes were predicted
print("\nPredicted Class Labels:", np.unique(y_pred))

# 💾 Save the Model and Scaler
joblib.dump(model, 'model/logistic_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("\n✅ Model and Scaler saved successfully.")
