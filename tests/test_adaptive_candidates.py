
import numpy as np
import pytest
import shapely
from shapely import STRtree

from src.features.candidates import select_candidate_indices


class FakeGeometry:
    def __init__(self):
        self.lines = shapely.linestrings([
            [[0, 0], [0, 10]],
            [[10, 0], [10, 10]],
            [[20, 0], [20, 10]],
            [[80, 0], [80, 10]],
        ])
        self.tree = STRtree(self.lines)


def test_fixed_radius_baseline():
    g = FakeGeometry()
    point = shapely.Point(0, 5)

    candidates = select_candidate_indices(
        g,
        point,
        search_radius_m=15,
        min_candidates=0,
        max_radius_m=100,
    )

    assert set(candidates.tolist()) == {0, 1}


def test_adaptive_expands_radius():
    g = FakeGeometry()
    point = shapely.Point(0, 5)

    candidates = select_candidate_indices(
        g,
        point,
        search_radius_m=15,
        min_candidates=3,
        max_radius_m=100,
    )

    assert set(candidates.tolist()) == {0, 1, 2}


def test_fallback_to_nearest():
    g = FakeGeometry()
    point = shapely.Point(200, 5)

    candidates = select_candidate_indices(
        g,
        point,
        search_radius_m=10,
        min_candidates=0,
        max_radius_m=100,
        fallback_nearest=True,
    )

    assert candidates.tolist() == [3]


def test_no_fallback_returns_empty():
    g = FakeGeometry()
    point = shapely.Point(200, 5)

    candidates = select_candidate_indices(
        g,
        point,
        search_radius_m=10,
        min_candidates=0,
        max_radius_m=100,
        fallback_nearest=False,
    )

    assert len(candidates) == 0


def test_no_duplicate_candidates():
    g = FakeGeometry()
    point = shapely.Point(0, 5)

    candidates = select_candidate_indices(
        g,
        point,
        search_radius_m=15,
        min_candidates=10,
        max_radius_m=100,
    )

    assert len(candidates) == len(np.unique(candidates))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_radius_m": 0},
        {"search_radius_m": -1},
        {"min_candidates": -1},
        {"max_radius_m": 10, "search_radius_m": 50},
    ],
)
def test_invalid_parameters(kwargs):
    g = FakeGeometry()
    point = shapely.Point(0, 5)

    with pytest.raises(ValueError):
        select_candidate_indices(g, point, **kwargs)
