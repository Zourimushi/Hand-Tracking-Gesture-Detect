from .GestureCollector.preprocess import GesturePreprocessor
import os
import joblib
import numpy as np
import torch
import torch.nn as nn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_DIR = os.path.join(
    CURRENT_DIR,
    "models"
)


class MLP(nn.Module):
    """Network structure used by ``train_mlp.py``."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(79, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
    def forward(self,x):
        return self.net(x)



class GesturePredictor:
    """
    Real-time gesture predictor.

    Input:
        MediaPipe hand landmarks

    Output:
        gesture id
        confidence
    """

    def __init__(self,
                 model_type="svm",
                 model_path=None,
                 scaler_path=None,
                 model_name=None):
        """Load an SVM or MLP gesture model.

        Args:
            model_type: ``"svm"`` (default) or ``"mlp"``.
            model_path: Optional path overriding the model's default path.
            scaler_path: Optional MLP scaler path. Ignored for SVM models.
            model_name: Backward-compatible SVM model name, without ``.pkl``.
        """
        # ``GesturePredictor("custom_svm")`` was valid before model_type was
        # introduced, so continue to interpret that positional value as an SVM
        # model name.
        if model_type not in {"svm", "mlp"} and model_name is None:
            model_name = model_type
            model_type = "svm"
        if model_name is not None:
            model_type = "svm"

        self.preprocessor = GesturePreprocessor()
        self.model_type = model_type.lower()

        if self.model_type not in {"svm", "mlp"}:
            raise ValueError("model_type must be 'svm' or 'mlp'")

        if self.model_type == "svm":
            if model_path is None:
                model_path = os.path.join(
                    DEFAULT_MODEL_DIR,
                    f"{model_name or 'svm_model'}.pkl",
                )
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            self.model = joblib.load(model_path)
            self.scaler = None
        else:
            model_path = model_path or os.path.join(DEFAULT_MODEL_DIR, "mlp_model.pth")
            scaler_path = scaler_path or os.path.join(DEFAULT_MODEL_DIR, "mlp_scaler.pkl")

            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(scaler_path)

            state_dict = torch.load(model_path, map_location="cpu")
            self.model = MLP()
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.scaler = joblib.load(scaler_path)

    # -----------------------------------------------------

    def predict(self, hand_info):
        """
        Parameters
        ----------
        hand_info : dict

            {
                "landmarks": [[x,y,z], ...],   # 21x3
                "handedness": "Left"/"Right"
            }

        Returns
        -------
        gesture : int

        confidence : float
        """

        landmarks = np.asarray(
            hand_info["landmarks"],
            dtype=np.float32
        )

        handedness = np.asarray(
            [hand_info["handedness"]]
        )

        landmarks = landmarks.reshape(1, 21, 3)

        feature = self.preprocessor.build_feature(
            landmarks,
            handedness
        )

        if self.model_type == "svm":
            prediction = self.model.predict(feature)[0]
            confidence = 1.0

            # SVM(probability=True)
            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(feature)[0]
                confidence = float(np.max(probability))
        else:
            scaled_feature = self.scaler.transform(feature).astype(np.float32)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(scaled_feature))
                probability = torch.softmax(logits, dim=1)[0]
                confidence, prediction = torch.max(probability, dim=0)

            prediction = prediction.item()
            confidence = confidence.item()

        return int(prediction), confidence
