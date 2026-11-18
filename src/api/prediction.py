
"""PathSnap: adaptive candidates, classifier, HMM, route geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.api.model_loader import load_model
from src.features.candidates import build_candidates, read_trip
from src.hmm.road_graph import build_road_graph
from src.hmm.viterbi import decode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = (
    PROJECT_ROOT / "src" / "models" / "best_model_metadata.json"
)


def load_feature_columns():
    """Use the exact feature order recorded during final training."""
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    features = metadata["feature_columns"]

    if not features:
        raise ValueError("Model feature list is empty.")

    return features


def load_candidate_settings():
    """Read candidate settings from final model metadata."""
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    return {
        "search_radius_m": float(
            metadata["search_radius_m"]
        ),
        "min_candidates": int(
            metadata["min_candidates"]
        ),
        "max_radius_m": float(
            metadata["max_radius_m"]
        ),
    }


def build_route_geometry(trip_path, predicted_arc_ids):
    """Convert predicted directed road arcs into map coordinates."""
    trip = read_trip(trip_path)

    nodes = trip["nodes"]
    arcs = trip["arcs"]

    coordinates = []

    for arc_id in predicted_arc_ids:
        arc_id = int(arc_id)

        if arc_id < 0 or arc_id >= len(arcs):
            raise ValueError(
                f"Invalid predicted arc ID: {arc_id}"
            )

        arc = arcs.iloc[arc_id]

        start_node = int(arc["from_node"])
        end_node = int(arc["to_node"])

        for node_id in (start_node, end_node):
            if node_id < 0 or node_id >= len(nodes):
                raise ValueError(
                    f"Invalid node ID: {node_id}"
                )

            node = nodes.iloc[node_id]

            point = [
                float(node["lat"]),
                float(node["lon"]),
            ]

            if not coordinates or coordinates[-1] != point:
                coordinates.append(point)

    return coordinates


def predict_trip(trip_path: str | Path):
    """Predict the map-matched road sequence for one trip."""
    trip_path = Path(trip_path)
    trip_id = trip_path.name

    model = load_model()

    if model is None:
        raise RuntimeError(
            "Trained model artifact is not available."
        )

    feature_columns = load_feature_columns()
    candidate_settings = load_candidate_settings()

    print(
        f"Generating adaptive candidates for {trip_id}...",
        flush=True,
    )

    candidates = build_candidates(
        trip_path,
        include_labels=False,
        fallback_nearest=True,
        **candidate_settings,
    )

    if candidates.empty:
        raise RuntimeError(
            f"No candidates generated for {trip_id}"
        )

    missing = set(feature_columns) - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    X = candidates[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if X.shape[1] != model.n_features_in_:
        raise ValueError(
            "Model feature count does not match "
            "candidate feature count."
        )

    print(
        f"Scoring {len(candidates):,} candidates...",
        flush=True,
    )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise ValueError(
            "Model returned invalid probabilities."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Model does not contain positive class 1."
        )

    positive_index = classes.index(1)
    probs = probabilities[:, positive_index]

    if not np.isfinite(probs).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    # Keep emissions strictly positive so Viterbi
    # can normalize them safely.
    probs = np.clip(probs, 1e-9, 1.0)

    arcs_file = trip_path / f"{trip_id}.arcs"

    print("Building road graph...", flush=True)
    road_graph = build_road_graph(arcs_file)

    print("Running HMM/Viterbi decoding...", flush=True)

    predicted_arc_ids = decode(
        candidates,
        probs,
        road_graph,
    )

    route_geometry = build_route_geometry(
        trip_path,
        predicted_arc_ids,
    )

    observation_count = int(
        candidates["obs_idx"].nunique()
    )

    if len(predicted_arc_ids) != observation_count:
        raise RuntimeError(
            "Decoded arc count does not match "
            "observation count."
        )

    return {
        "predicted_arc_ids": [
            int(arc_id)
            for arc_id in predicted_arc_ids
        ],
        "route_geometry": route_geometry,
        "candidate_count": int(len(candidates)),
        "observation_count": observation_count,
    }
