import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from config import DATA_PATH, MODEL_DIR, TEST_SIZE, RANDOM_STATE

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Drop leakage column
df = df.drop(columns=["Yield"])

# Log transform target (Production)
y = np.log1p(df["Production"])
X = df.drop(columns=["Production"])

# Identify categorical and numerical columns
categorical_cols = ["Crop", "Season", "State"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print("Training model...")
pipeline.fit(X_train, y_train)

# Predictions (log scale)
y_pred_log = pipeline.predict(X_test)

# Convert predictions back to original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Metrics (on original scale)
r2 = r2_score(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))

print("\n=== Regression Performance (Log Transformed Target) ===")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save model
joblib.dump(pipeline, os.path.join(MODEL_DIR, "production_model_log.pkl"))
print("Model saved successfully.")