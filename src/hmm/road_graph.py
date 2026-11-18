
"""Build a directed graph of connected road arcs."""

from pathlib import Path

import networkx as nx


def build_road_graph(arcs_file: str | Path) -> nx.DiGraph:
    """
    Build a directed graph where each node is a road arc ID.

    Two arc IDs are connected when the first arc's ending
    road node matches the second arc's starting road node.
    Arc IDs are zero-based line indices in the .arcs file.
    """
    arcs = []

    with Path(arcs_file).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            values = line.split()

            if not values:
                continue

            if len(values) < 2:
                raise ValueError(
                    f"Invalid arc at line {line_number + 1}"
                )

            from_node = int(values[0])
            to_node = int(values[1])

            arcs.append((from_node, to_node))

    graph = nx.DiGraph()

    # Preserve the original zero-based arc IDs.
    graph.add_nodes_from(range(len(arcs)))

    # Group outgoing arcs by their starting road node.
    outgoing_arcs = {}

    for arc_id, (from_node, _) in enumerate(arcs):
        outgoing_arcs.setdefault(from_node, []).append(arc_id)

    # Connect arcs when the first arc ends where the next begins.
    for arc_id, (_, to_node) in enumerate(arcs):
        for next_arc_id in outgoing_arcs.get(to_node, []):
            if arc_id != next_arc_id:
                graph.add_edge(arc_id, next_arc_id)

    return graph
