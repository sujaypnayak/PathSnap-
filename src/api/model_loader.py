
"""Load the trained PathSnap candidate classifier."""

from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_model.joblib"


def load_model():
    """Load the trained model when its artifact is available."""
    if not MODEL_PATH.is_file():
        return None

    return joblib.load(MODEL_PATH)
