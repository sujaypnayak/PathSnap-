
"""Build a directed graph of connected road arcs."""

from pathlib import Path

import networkx as nx


def build_road_graph(arcs_file: str | Path) -> nx.DiGraph:
    """Build a directed arc-transition graph."""

    arcs = []

    with Path(arcs_file).open(
        "r", encoding="utf-8"
    ) as file:
        for line_number, line in enumerate(file):
            values = line.split()

            if not values:
                continue

            if len(values) < 2:
                raise ValueError(
                    f"Invalid arc at line {line_number + 1}"
                )

            arcs.append((
                int(values[0]),
                int(values[1]),
            ))

    graph = nx.DiGraph()

    for arc_id, (from_node, to_node) in enumerate(arcs):
        graph.add_node(
            arc_id,
            from_node=from_node,
            to_node=to_node,
        )

    outgoing_arcs = {}

    for arc_id, (from_node, _) in enumerate(arcs):
        outgoing_arcs.setdefault(
            from_node, []
        ).append(arc_id)

    for arc_id, (_, to_node) in enumerate(arcs):
        for next_arc_id in outgoing_arcs.get(
            to_node, []
        ):
            if arc_id != next_arc_id:
                graph.add_edge(
                    arc_id, next_arc_id
                )

    return graph
