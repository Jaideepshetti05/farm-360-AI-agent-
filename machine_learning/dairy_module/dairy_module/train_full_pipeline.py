import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================
# 1️⃣ LOAD DATA
# ==============================

df = pd.read_csv("data/dairy/Mik_Pro.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nChecking Missing Values:")
print(df.isnull().sum())

# ==============================
# 2️⃣ DATA CLEANING
# ==============================

# Drop missing rows if any
df = df.dropna()

# ==============================
# 3️⃣ FEATURE & TARGET SELECTION
# ==============================

X = df[['Year']]
y = df['Milk Production']

# ==============================
# 4️⃣ TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5️⃣ FEATURE SCALING
# ==============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 6️⃣ TRAIN MODELS
# ==============================

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# ==============================
# 7️⃣ EVALUATION
# ==============================

def evaluate(model, X_test_data, model_name):
    predictions = model.predict(X_test_data)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"\n--- {model_name} ---")
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)
    
    return r2

lr_r2 = evaluate(lr_model, X_test_scaled, "Linear Regression")
rf_r2 = evaluate(rf_model, X_test, "Random Forest")

# ==============================
# 8️⃣ CROSS VALIDATION
# ==============================

cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nRandom Forest Cross Validation Score:", np.mean(cv_scores))

# ==============================
# 9️⃣ FEATURE IMPORTANCE
# ==============================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

print("\nFeature Importance:")
print(feature_importance)

# ==============================
# 🔟 VISUALIZATION
# ==============================

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=rf_model.predict(X_test))
plt.xlabel("Actual Milk Production")
plt.ylabel("Predicted Milk Production")
plt.title("Actual vs Predicted")
plt.show()

# ==============================
# 1️⃣1️⃣ SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

pickle.dump(rf_model, open("models/dairy_regression_model.pkl", "wb"))
pickle.dump(scaler, open("models/dairy_scaler.pkl", "wb"))

print("\nModel and scaler saved successfully!")
