
"""PathSnap candidate classification and route decoding pipeline."""

from pathlib import Path

import yaml

from src.api.model_loader import load_model
from src.features.candidates import build_candidates
from src.hmm.road_graph import build_road_graph
from src.hmm.viterbi import decode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_feature_columns():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["feature_cols"]


def predict_trip(trip_path: str | Path):
    """Generate a route using the trained candidate classifier."""
    model = load_model()

    if model is None:
        raise RuntimeError("Trained model artifact is not available.")

    trip_path = Path(trip_path)
    trip_id = trip_path.name

    candidates = build_candidates(trip_path, include_labels=False)

    feature_columns = load_feature_columns()
    missing = set(feature_columns) - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    X = candidates[feature_columns]

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model must provide binary class probabilities.")

    # Positive class probability is the candidate emission score.
    positive_index = list(model.classes_).index(1)
    probs = probabilities[:, positive_index]

    arcs_file = trip_path / f"{trip_id}.arcs"
    road_graph = build_road_graph(arcs_file)

    predicted_arc_ids = decode(candidates, probs, road_graph)

    return {
        "predicted_arc_ids": predicted_arc_ids,
        "candidate_count": len(candidates),
        "observation_count": candidates["obs_idx"].nunique(),
    }
