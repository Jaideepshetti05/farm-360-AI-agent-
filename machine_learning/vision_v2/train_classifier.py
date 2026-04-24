import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from config import EMBEDDINGS_DIR, TEST_SIZE, RANDOM_SEED, CLASSIFIER_TYPE

# Load embeddings
X = np.load(os.path.join(EMBEDDINGS_DIR, "X_embeddings.npy"))
y = np.load(os.path.join(EMBEDDINGS_DIR, "y_labels.npy"))

print("Loaded embeddings:", X.shape)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# Choose classifier
if CLASSIFIER_TYPE == "logistic":
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
elif CLASSIFIER_TYPE == "svm":
    model = SVC(kernel="linear")
else:
    raise ValueError("Unsupported classifier type")

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
print("Cross-validation accuracy:", cv_scores.mean())