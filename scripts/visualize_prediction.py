
"""Visualize PathSnap GPS observations and predicted road arcs."""

from __future__ import annotations

import argparse
from pathlib import Path

import folium

from src.api.config import STUDENT_TEST_DIR
from src.api.main import load_gps_points
from src.api.prediction import predict_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-id", default="00000052")
    args = parser.parse_args()

    trip_id = args.trip_id
    trip_path = STUDENT_TEST_DIR / trip_id
    track_file = trip_path / f"{trip_id}.track"

    if not track_file.is_file():
        raise FileNotFoundError(track_file)

    gps_points = load_gps_points(track_file)
    result = predict_trip(trip_path)

    gps_coordinates = [
        [point["lat"], point["lng"]]
        for point in gps_points
    ]

    if not gps_coordinates:
        raise RuntimeError("No GPS observations found.")

    # No external map tiles are required.
    # This avoids the OpenStreetMap HTTP 403 error.
    map_view = folium.Map(
        location=gps_coordinates[0],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    # Original GPS track.
    folium.PolyLine(
        gps_coordinates,
        color="#ef4444",
        weight=3,
        opacity=0.75,
        tooltip="Original GPS track",
    ).add_to(map_view)

    # Draw each predicted road arc separately.
    # This avoids visually inventing straight connections
    # between unrelated arcs.
    from src.features.candidates import read_trip

    trip = read_trip(trip_path)
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    predicted_arc_ids = result["predicted_arc_ids"]

    disconnected_transitions = 0
    previous_end_node = None

    for arc_id in predicted_arc_ids:
        arc = arcs.iloc[int(arc_id)]

        start_node = int(arc["from_node"])
        end_node = int(arc["to_node"])

        start = nodes.iloc[start_node]
        end = nodes.iloc[end_node]

        arc_coordinates = [
            [float(start["lat"]), float(start["lon"])],
            [float(end["lat"]), float(end["lon"])],
        ]

        folium.PolyLine(
            arc_coordinates,
            color="#2563eb",
            weight=4,
            opacity=0.8,
            tooltip=f"Predicted arc {arc_id}",
        ).add_to(map_view)

        if (
            previous_end_node is not None
            and previous_end_node != start_node
        ):
            disconnected_transitions += 1

        previous_end_node = end_node

    folium.CircleMarker(
        gps_coordinates[0],
        radius=7,
        color="green",
        fill=True,
        tooltip="GPS start",
    ).add_to(map_view)

    folium.CircleMarker(
        gps_coordinates[-1],
        radius=7,
        color="black",
        fill=True,
        tooltip="GPS end",
    ).add_to(map_view)

    map_view.fit_bounds(gps_coordinates)

    output_dir = Path("reports/maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{trip_id}_prediction.html"
    map_view.save(str(output_file))

    print("Visualization saved:", output_file.resolve())
    print("GPS observations:", len(gps_coordinates))
    print("Predicted arcs:", len(predicted_arc_ids))
    print(
        "Disconnected consecutive arc transitions:",
        disconnected_transitions,
    )


if __name__ == "__main__":
    main()
