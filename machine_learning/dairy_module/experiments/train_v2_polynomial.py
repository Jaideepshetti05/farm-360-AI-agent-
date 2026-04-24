import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================
# 1️⃣ LOAD DATA
# ==============================

df = pd.read_csv("data/dairy/Mik_Pro.csv")

X = df[['Year']]
y = df['Milk Production']

# ==============================
# 2️⃣ TIME SERIES SPLIT
# ==============================

tscv = TimeSeriesSplit(n_splits=5)

# ==============================
# 3️⃣ MODEL 1 — Polynomial Regression
# ==============================

poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

poly_scores = cross_val_score(poly_pipeline, X, y, cv=tscv, scoring="r2")
print("\nPolynomial Regression CV R2:", np.mean(poly_scores))

# ==============================
# 4️⃣ MODEL 2 — Gradient Boosting
# ==============================

gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_scores = cross_val_score(gb_model, X, y, cv=tscv, scoring="r2")
print("Gradient Boosting CV R2:", np.mean(gb_scores))

# ==============================
# 5️⃣ TRAIN BEST MODEL ON FULL DATA
# ==============================

if np.mean(gb_scores) > np.mean(poly_scores):
    best_model = gb_model
    model_name = "Gradient Boosting"
else:
    best_model = poly_pipeline
    model_name = "Polynomial Regression"

best_model.fit(X, y)

# ==============================
# 6️⃣ FINAL EVALUATION
# ==============================

predictions = best_model.predict(X)

r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
