
"""PathSnap candidate classification and route decoding pipeline."""

from pathlib import Path

import yaml

from src.api.model_loader import load_model
from src.features.candidates import build_candidates, read_trip
from src.hmm.road_graph import build_road_graph
from src.hmm.viterbi import decode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_feature_columns():
    """Load the feature columns expected by the trained model."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["feature_cols"]


def build_route_geometry(trip_path, predicted_arc_ids):
    """Convert predicted arc IDs into [latitude, longitude] coordinates."""
    trip = read_trip(trip_path)
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    coordinates = []

    for arc_id in predicted_arc_ids:
        arc_id = int(arc_id)

        if arc_id < 0 or arc_id >= len(arcs):
            raise ValueError(f"Invalid predicted arc ID: {arc_id}")

        arc = arcs.iloc[arc_id]
        start_node = int(arc["from_node"])
        end_node = int(arc["to_node"])

        for node_id in (start_node, end_node):
            if node_id < 0 or node_id >= len(nodes):
                raise ValueError(f"Invalid node ID: {node_id}")

            node = nodes.iloc[node_id]
            point = [float(node["lat"]), float(node["lon"])]

            if not coordinates or coordinates[-1] != point:
                coordinates.append(point)

    return coordinates


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

    if 1 not in model.classes_:
        raise ValueError("Model does not contain the expected positive class label 1.")

    positive_index = list(model.classes_).index(1)
    probs = probabilities[:, positive_index]

    arcs_file = trip_path / f"{trip_id}.arcs"
    road_graph = build_road_graph(arcs_file)

    predicted_arc_ids = decode(candidates, probs, road_graph)
    route_geometry = build_route_geometry(trip_path, predicted_arc_ids)

    return {
        "predicted_arc_ids": predicted_arc_ids,
        "route_geometry": route_geometry,
        "candidate_count": len(candidates),
        "observation_count": candidates["obs_idx"].nunique(),
    }
