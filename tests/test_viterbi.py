
import networkx as nx
import pandas as pd
import pytest
import numpy as np

from src.hmm.viterbi import decode, DISCONNECTED_PENALTY


def create_toy_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("A", "B"),
        ("B", "C"),
    ])
    return graph


def test_viterbi_finds_connected_route():
    graph = create_toy_graph()

    candidates = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1, 2, 2],
        "arc_id": ["A", "X", "B", "Y", "C", "Z"],
    })

    probabilities = [
        0.90, 0.10,
        0.85, 0.15,
        0.95, 0.05,
    ]

    result = decode(candidates, probabilities, graph)

    assert result == ["A", "B", "C"]


def test_viterbi_prefers_connected_path_over_disconnected_path():
    graph = nx.DiGraph()
    graph.add_edge("A", "B")
    graph.add_node("C")

    candidates = pd.DataFrame({
        "obs_idx": [0, 1, 1],
        "arc_id": ["A", "B", "C"],
    })

    probabilities = [1.0, 0.2, 0.8]

    result = decode(candidates, probabilities, graph)

    # B is connected to A, while C is disconnected.
    assert result == ["A", "B"]

    # Compare the two possible transition scores.
    connected_score = 0.0 + np.log(0.2)
    disconnected_score = DISCONNECTED_PENALTY + np.log(0.8)

    assert disconnected_score < connected_score


def test_viterbi_allows_staying_on_same_arc():
    graph = nx.DiGraph()

    candidates = pd.DataFrame({
        "obs_idx": [0, 1, 2],
        "arc_id": ["A", "A", "A"],
    })

    probabilities = [1.0, 1.0, 1.0]

    result = decode(candidates, probabilities, graph)

    assert result == ["A", "A", "A"]


def test_viterbi_rejects_wrong_probability_count():
    graph = create_toy_graph()

    candidates = pd.DataFrame({
        "obs_idx": [0, 0],
        "arc_id": ["A", "X"],
    })

    probabilities = [0.9]

    with pytest.raises(ValueError, match="exactly one probability"):
        decode(candidates, probabilities, graph)
