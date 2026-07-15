"""Gesture dataset preprocessing for both training and real-time prediction."""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_DIR = os.path.dirname(CURRENT_DIR)
DEFAULT_DATASET_DIR = os.path.join(CLASSIFIER_DIR, "dataset_gesture")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42


class GesturePreprocessor:
    """Build model features and split the collected gesture dataset."""

    landmark_columns = [
        coordinate
        for index in range(21)
        for coordinate in (f"x{index}", f"y{index}", f"z{index}")
    ]
    angle_columns = [
        f"angle{i}" for i in range(15)
    ]
    feature_columns = landmark_columns + angle_columns

    def __init__(self, input_csv=None, output_dir=None):
        self.input_csv = input_csv or os.path.join(DEFAULT_DATASET_DIR, "raw.csv")
        self.output_dir = output_dir or DEFAULT_DATASET_DIR

    @staticmethod
    def build_feature(landmarks: np.ndarray, hands: np.ndarray) -> np.ndarray:
        """Convert hand landmarks into the same features used for training.

        Args:
            landmarks: Landmark matrix with shape ``(N, 21, 3)``.
            hands: Hand side values with shape ``(N,)``. Values may be
                ``"Left"``/``"Right"`` or already encoded as 0/1.

        Returns:
            A matrix with shape ``(N, 1+63+15)``: 
            1.1 hand value (0 for left, 1 for right),
            2.63 normalized landmark coordinates, 
            3.15 additional angle features.

        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        hands = np.asarray(hands)

        if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
            raise ValueError("landmarks must have shape (N, 21, 3)")
        if hands.ndim != 1 or hands.shape[0] != landmarks.shape[0]:
            raise ValueError("hands must have shape (N,) and match landmarks")

        try:
            hand_values = np.asarray(
                [0 if hand == "Left" else 1 if hand == "Right" else float(hand) for hand in hands],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("hands must contain Left/Right or numeric 0/1 values") from error

        # Keep the original preprocessing: wrist as origin, then normalize by
        # the wrist-to-middle-MCP (landmark 9) distance.
        normalized = landmarks - landmarks[:, 0:1, :]
        scale = np.linalg.norm(normalized[:, 9, :], axis=1)
        scale[scale < 1e-6] = 1.0
        normalized = normalized / scale[:, np.newaxis, np.newaxis]

        # Compute additional angle features (example: angles between fingers)
        angles = []

        finger_angles = [
            (0, 1, 2),
            (1, 2, 3),
            (2, 3, 4),

            (0, 5, 6),
            (5, 6, 7),
            (6, 7, 8),

            (0, 9, 10),
            (9, 10, 11),
            (10, 11, 12),

            (0, 13, 14),
            (13, 14, 15),
            (14, 15, 16),

            (0, 17, 18),
            (17, 18, 19),
            (18, 19, 20),
        ]
        for a, b, c in finger_angles:
            angles.append(
                GesturePreprocessor.calculate_angle(
                    normalized[:, a],
                    normalized[:, b],
                    normalized[:, c],
                )
            )

        angles = np.stack(angles, axis=1)
        return np.concatenate(
            (hand_values[:, np.newaxis], normalized.reshape(landmarks.shape[0], -1), angles),
            axis=1,
        )

    def preprocess_dataset(self):
        """Read raw.csv, build features, split it, and save train/val/test CSVs."""
        print("Loading dataset...")
        raw_df = pd.read_csv(self.input_csv)

        # Explicitly split the raw CSV fields. Score is intentionally excluded
        # from model features, exactly as in the previous preprocessing script.
        labels = raw_df.pop("label")
        scores = raw_df.pop("score")
        hands = raw_df.pop("hand").to_numpy()
        landmarks = raw_df[self.landmark_columns].to_numpy(dtype=np.float32)
        landmarks = landmarks.reshape(-1, 21, 3)

        print("Normalizing landmarks...")
        features = self.build_feature(landmarks, hands)
        feature_df = pd.DataFrame(features[:, 1:], columns=self.feature_columns)
        feature_df.insert(0, "hand", features[:, 0].astype(np.int8))
        feature_df.insert(0, "label", labels.to_numpy())

        print("Splitting dataset...")
        train_df, temp_df = train_test_split(
            feature_df,
            test_size=VAL_RATIO + TEST_RATIO,
            random_state=RANDOM_STATE,
            stratify=feature_df["label"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_STATE,
            stratify=temp_df["label"],
        )

        os.makedirs(self.output_dir, exist_ok=True)
        paths = {
            "train": os.path.join(self.output_dir, "train.csv"),
            "val": os.path.join(self.output_dir, "val.csv"),
            "test": os.path.join(self.output_dir, "test.csv"),
        }
        train_df.to_csv(paths["train"], index=False)
        val_df.to_csv(paths["val"], index=False)
        test_df.to_csv(paths["test"], index=False)

        return train_df, val_df, test_df, paths


    @staticmethod
    def calculate_angle(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> np.ndarray:
        """
        Calculate angle ABC for one or more samples.

        Args:
            a: Shape (N, 3)
            b: Shape (N, 3)
            c: Shape (N, 3)

        Returns:
            Angles in radians with shape (N,).
        """

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        c = np.asarray(c, dtype=np.float32)

        if a.shape != b.shape or b.shape != c.shape:
            raise ValueError("a, b and c must have the same shape.")

        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError("Input arrays must have shape (N, 3).")

        ba = a - b
        bc = c - b

        ba_norm = np.linalg.norm(ba, axis=1)
        bc_norm = np.linalg.norm(bc, axis=1)

        denominator = ba_norm * bc_norm

        # 防止除0
        denominator[denominator < 1e-6] = 1.0

        cosine = np.sum(ba * bc, axis=1) / denominator
        cosine = np.clip(cosine, -1.0, 1.0)

        return np.arccos(cosine)
def main():
    """Allow this module to remain directly executable for dataset preparation."""
    preprocessor = GesturePreprocessor()
    train_df, val_df, test_df, paths = preprocessor.preprocess_dataset()

    print("\n=================================")
    print("Preprocess Finished")
    print("=================================")
    print(f"Train : {len(train_df)}")
    print(f"Val   : {len(val_df)}")
    print(f"Test  : {len(test_df)}")
    print(f"\nFeature Dimension : {train_df.shape[1] - 2}")
    print("Classes :", train_df["label"].nunique())
    print("Saved to:")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
