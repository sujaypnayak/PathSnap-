
"""Utilities for constructing an arc-transition road graph."""

import networkx as nx
import pandas as pd


def build_arc_transition_graph(arcs_df):
    """
    Build a directed graph in which each node is a road arc.

    An edge A -> B exists when the ending node of arc A
    equals the starting node of arc B.

    Expected columns:
        arc_id, start_node, end_node
    """
    required = {"arc_id", "start_node", "end_node"}

    if not required.issubset(arcs_df.columns):
        raise ValueError(f"Required columns: {required}")

    if arcs_df.empty:
        raise ValueError("The arcs dataframe is empty.")

    if arcs_df["arc_id"].duplicated().any():
        raise ValueError("Duplicate arc IDs found.")

    graph = nx.DiGraph()

    # Every road segment is a state in the graph.
    for row in arcs_df.itertuples(index=False):
        graph.add_node(
            row.arc_id,
            start_node=row.start_node,
            end_node=row.end_node,
        )

    # Index arcs by their starting node.
    arcs_by_start = {}

    for row in arcs_df.itertuples(index=False):
        arcs_by_start.setdefault(row.start_node, []).append(row.arc_id)

    # Connect arcs when their endpoints match.
    for row in arcs_df.itertuples(index=False):
        next_arcs = arcs_by_start.get(row.end_node, [])

        for next_arc in next_arcs:
            graph.add_edge(row.arc_id, next_arc)

    return graph


def validate_ground_truth_route(route, arcs_df):
    """Check that every route arc exists and consecutive arcs connect."""

    arc_lookup = arcs_df.set_index("arc_id")

    for arc_id in route:
        if arc_id not in arc_lookup.index:
            raise ValueError(f"Unknown arc ID in route: {arc_id}")

    for current_arc, next_arc in zip(route, route[1:]):
        current_end = arc_lookup.loc[current_arc, "end_node"]
        next_start = arc_lookup.loc[next_arc, "start_node"]

        if current_end != next_start:
            raise ValueError(
                f"Disconnected route transition: "
                f"{current_arc} -> {next_arc}"
            )

    return True
