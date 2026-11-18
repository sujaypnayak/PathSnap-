import numpy as np
import pandas as pd

from src.features.candidates import (
    FEATURE_COLUMNS,
    build_candidates,
)


def create_sample_trip(path):
    trip_id = path.name

    # Three nearby GPS observations
    pd.DataFrame([
        [13.00000, 77.00000, 1000],
        [13.00010, 77.00000, 1010],
        [13.00020, 77.00000, 1020],
    ]).to_csv(path / f"{trip_id}.track", sep=" ", header=False, index=False)

    # Road network nodes
    pd.DataFrame([
        [13.00000, 77.00000],
        [13.00010, 77.00000],
        [13.00020, 77.00000],
    ]).to_csv(path / f"{trip_id}.nodes", sep=" ", header=False, index=False)

    # Directed road arcs
    pd.DataFrame([
        [0, 1],
        [1, 2],
    ]).to_csv(path / f"{trip_id}.arcs", sep=" ", header=False, index=False)

    # Reference route for training
    pd.DataFrame([[0], [1]]).to_csv(
        path / f"{trip_id}.route",
        sep=" ",
        header=False,
        index=False,
    )


def test_candidate_generation(tmp_path):
    trip = tmp_path / "sample_trip"
    trip.mkdir()
    create_sample_trip(trip)

    df = build_candidates(
        trip,
        search_radius_m=50,
        include_labels=True,
    )

    assert not df.empty
    assert all(col in df.columns for col in FEATURE_COLUMNS)
    assert df["distance_m"].ge(0).all()
    assert set(df["label"].unique()).issubset({0, 1})


def test_labels_hidden_when_disabled(tmp_path):
    trip = tmp_path / "sample_trip"
    trip.mkdir()
    create_sample_trip(trip)

    df = build_candidates(
        trip,
        search_radius_m=50,
        include_labels=False,
    )

    assert not df.empty
    assert df["label"].isna().all()


def test_candidate_ids_and_features(tmp_path):
    trip = tmp_path / "sample_trip"
    trip.mkdir()
    create_sample_trip(trip)

    df = build_candidates(trip)

    assert df["trip_id"].eq("sample_trip").all()
    assert df["obs_idx"].ge(0).all()
    assert df["arc_id"].ge(0).all()
    assert np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all()