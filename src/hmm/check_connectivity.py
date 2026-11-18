
import csv
from pathlib import Path

import networkx as nx

DATA_ROOT = Path("data/raw/Student folder")
REPORT_PATH = Path("experiments/connectivity_report.csv")


def check_trip(trip_dir):
    trip_id = trip_dir.name

    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    # Each line in .arcs represents a directed edge:
    # start_node -> end_node
    arcs = []

    with arcs_file.open("r") as file:
        for line in file:
            values = line.split()
            if len(values) >= 2:
                arcs.append((int(values[0]), int(values[1])))

    with route_file.open("r") as file:
        route = [
            int(line.strip())
            for line in file
            if line.strip()
        ]

    # Build a directed graph to measure the shortest
    # number of road-network hops between two nodes.
    graph = nx.DiGraph()
    graph.add_edges_from(arcs)

    mismatches = []

    for i in range(len(route) - 1):
        current_arc = route[i]
        next_arc = route[i + 1]

        # Consecutive repetitions of the same arc are valid.
        if current_arc == next_arc:
            continue

        # Check that both arc IDs are valid line indices.
        if (
            current_arc < 0
            or next_arc < 0
            or current_arc >= len(arcs)
            or next_arc >= len(arcs)
        ):
            mismatches.append({
                "trip_id": trip_id,
                "route_position": i,
                "current_arc": current_arc,
                "current_end_node": None,
                "next_arc": next_arc,
                "next_start_node": None,
                "hop_distance": None,
                "reason": "Invalid arc ID",
            })
            continue

        current_end = arcs[current_arc][1]
        next_start = arcs[next_arc][0]

        if current_end == next_start:
            continue

        try:
            hop_distance = nx.shortest_path_length(
                graph,
                source=current_end,
                target=next_start,
            )
        except nx.NetworkXNoPath:
            hop_distance = None
        except nx.NodeNotFound:
            hop_distance = None

        mismatches.append({
            "trip_id": trip_id,
            "route_position": i,
            "current_arc": current_arc,
            "current_end_node": current_end,
            "next_arc": next_arc,
            "next_start_node": next_start,
            "hop_distance": hop_distance,
            "reason": "Node mismatch",
        })

    return mismatches


def main():
    all_mismatches = []
    total_trips = 0

    for split in ["train", "student_test"]:
        split_dir = DATA_ROOT / split

        for trip_dir in sorted(split_dir.iterdir()):
            if not trip_dir.is_dir():
                continue

            total_trips += 1
            mismatches = check_trip(trip_dir)
            all_mismatches.extend(mismatches)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trip_id",
        "route_position",
        "current_arc",
        "current_end_node",
        "next_arc",
        "next_start_node",
        "hop_distance",
        "reason",
    ]

    with REPORT_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_mismatches)

    print(f"Total trips checked: {total_trips}")
    print(f"Total mismatches: {len(all_mismatches)}")

    for item in all_mismatches:
        print(
            f"Trip {item['trip_id']} | "
            f"Position {item['route_position']} | "
            f"Arc {item['current_arc']} ends at "
            f"{item['current_end_node']} -> "
            f"Arc {item['next_arc']} starts at "
            f"{item['next_start_node']} | "
            f"Hop distance: {item['hop_distance']} | "
            f"{item['reason']}"
        )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
