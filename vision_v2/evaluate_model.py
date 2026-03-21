import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from config import EMBEDDINGS_DIR, RESULTS_DIR, TEST_SIZE, RANDOM_SEED

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load embeddings
X = np.load(os.path.join(EMBEDDINGS_DIR, "X_embeddings.npy"))
y = np.load(os.path.join(EMBEDDINGS_DIR, "y_labels.npy"))

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred)
print(report)

# Save report
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("Evaluation complete. Results saved.")