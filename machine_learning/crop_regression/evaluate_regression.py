import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_PATH, MODEL_DIR, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading trained model...")
pipeline = joblib.load(os.path.join(MODEL_DIR, "production_model_log.pkl"))

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Yield"])

X = df.drop(columns=["Production"])

# Extract feature names after preprocessing
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# Get categorical feature names
categorical_cols = ["Crop", "Season", "State"]
encoder = preprocessor.named_transformers_["cat"]
encoded_cat_features = encoder.get_feature_names_out(categorical_cols)

# Numerical columns
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Combine all feature names
feature_names = list(encoded_cat_features) + numerical_cols

# Extract feature importances
importances = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n=== Top 20 Most Important Features ===")
print(importance_df.head(20))

# Save results
importance_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(
    importance_df.head(20)["Feature"][::-1],
    importance_df.head(20)["Importance"][::-1]
)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
plt.close()

print("\nFeature importance saved in results folder.")