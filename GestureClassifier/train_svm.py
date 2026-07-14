import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================
# Config
# =====================================

DATASET_DIR = "Gesture_Detect/GestureClassifier/dataset"
MODEL_DIR = "Gesture_Detect/GestureClassifier/models"

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

# =====================================
# Read Dataset
# =====================================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# =====================================
# Split Feature / Label
# =====================================

X_train = train_df.drop(columns=["label"]).to_numpy(dtype="float32")
y_train = train_df["label"].to_numpy()

X_val = val_df.drop(columns=["label"]).to_numpy(dtype="float32")
y_val = val_df["label"].to_numpy()

X_test = test_df.drop(columns=["label"]).to_numpy(dtype="float32")
y_test = test_df["label"].to_numpy()

print(f"Train : {len(X_train)}")
print(f"Val   : {len(X_val)}")
print(f"Test  : {len(X_test)}")

# =====================================
# Build Model
# =====================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True 
    ))
])

print("\nTraining SVM...")

model.fit(X_train, y_train)

# =====================================
# Validation
# =====================================

print("\n========== Validation ==========")

val_pred = model.predict(X_val)

val_acc = accuracy_score(y_val, val_pred)

print(f"Accuracy : {val_acc:.4f}")

# =====================================
# Test
# =====================================

print("\n========== Test ==========")

test_pred = model.predict(X_test)

test_acc = accuracy_score(y_test, test_pred)

print(f"Accuracy : {test_acc:.4f}")

print("\nClassification Report")

print(classification_report(
    y_test,
    test_pred,
    digits=4
))

print("\nConfusion Matrix")

print(confusion_matrix(
    y_test,
    test_pred
))

# =====================================
# Save Model
# =====================================

joblib.dump(model, MODEL_PATH)

print("\n====================================")
print("Training Finished")
print("====================================")
print("Model Saved:")
print(MODEL_PATH)