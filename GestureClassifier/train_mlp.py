import os
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================
# Config
# =====================================

DATASET_DIR = "Gesture_Detect/GestureClassifier/dataset"
MODEL_DIR = "Gesture_Detect/GestureClassifier/models"

TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler.pkl")

BATCH_SIZE = 64
EPOCHS = 95
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# Read Dataset
# =====================================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y_train = train_df["label"].to_numpy()

X_val = val_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y_val = val_df["label"].to_numpy()

X_test = test_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
y_test = test_df["label"].to_numpy()

print(f"Train : {len(X_train)}")
print(f"Val   : {len(X_val)}")
print(f"Test  : {len(X_test)}")

# =====================================
# StandardScaler
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, SCALER_PATH)

# =====================================
# DataLoader
# =====================================

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_x = torch.tensor(X_val).to(DEVICE)
test_x = torch.tensor(X_test).to(DEVICE)

# =====================================
# Model
# =====================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(79,128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,10)

        )

    def forward(self,x):
        return self.net(x)

model = MLP().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# =====================================
# Train
# =====================================

best_acc = 0

print("\nTraining MLP...\n")

for epoch in range(EPOCHS):

    model.train()

    loss_sum = 0

    for x,y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out,y)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

    model.eval()

    with torch.no_grad():

        pred = model(val_x)

        pred = pred.argmax(1).cpu().numpy()

    acc = accuracy_score(y_val,pred)

    print(
        f"Epoch {epoch+1:03d} "
        f"Loss {loss_sum:.4f} "
        f"Val {acc:.4f}"
    )

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

# =====================================
# Test
# =====================================

print("\nLoading Best Model...\n")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

with torch.no_grad():

    pred = model(test_x)

    pred = pred.argmax(1).cpu().numpy()

acc = accuracy_score(y_test,pred)

print("========== Test ==========")

print(f"Accuracy : {acc:.4f}")

print()

print(classification_report(
    y_test,
    pred,
    digits=4
))

print()

print(confusion_matrix(
    y_test,
    pred
))

print()

print("Saved Model:")
print(MODEL_PATH)

print("Saved Scaler:")
print(SCALER_PATH)