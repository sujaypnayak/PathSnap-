
"""Candidate generation and feature engineering for PathSnap trips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely import STRtree


FEATURE_COLUMNS = [
    "distance_m",
    "gps_bearing_deg",
    "road_bearing_deg",
    "bearing_diff_deg",
    "vehicle_speed_mps",
    "time_diff_s",
    "stopped_flag",
    "prev_obs_distance_m",
    "next_obs_distance_m",
    "prev_time_diff_s",
    "next_time_diff_s",
    "prev_bearing_deg",
    "next_bearing_deg",
    "segment_length_m",
    "candidate_within_radius",
]


def read_trip(
    trip_path: str | Path,
    include_reference_route: bool = False,
) -> dict[str, Any]:
    """Read the files for one trip."""
    p = Path(trip_path)
    trip_id = p.name

    track = pd.read_csv(
        p / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype="float64",
    )

    nodes = pd.read_csv(
        p / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype="float64",
    )

    arcs = pd.read_csv(
        p / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype="int64",
    )

    route = None

    if include_reference_route:
        route = pd.read_csv(
            p / f"{trip_id}.route",
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            dtype="int64",
        )["arc_id"].to_numpy()

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "path": p,
    }


def _bearing(
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    """Clockwise degrees from north, in [0, 360)."""
    return (
        np.degrees(np.arctan2(dx, dy)) + 360.0
    ) % 360.0


def _route_labels(
    points: np.ndarray,
    route_lines: np.ndarray,
    route_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign GPS observations to reference arcs in route order.

    Dynamic programming permits staying on the current
    reference arc or advancing to a later route position.

    Reference labels are inferred from the supplied route.
    They are not independently verified ground truth.
    """
    if len(route_ids) == 0:
        raise ValueError(
            "Reference route contains no valid arc IDs"
        )

    n = len(points)
    m = len(route_ids)

    costs = np.empty((n, m), dtype=np.float32)

    for start in range(0, n, 128):
        end = min(start + 128, n)

        distances = shapely.distance(
            points[start:end, None],
            route_lines[None, :],
        )

        costs[start:end] = (
            np.asarray(distances, dtype=np.float32) ** 2
        )

    dp = np.empty_like(costs)
    back = np.zeros((n, m), dtype=np.int32)

    dp[0] = costs[0]

    for i in range(1, n):
        previous = dp[i - 1]

        prefix_cost = np.minimum.accumulate(previous)

        best_idx = np.empty(m, dtype=np.int32)
        best = 0

        for j in range(m):
            if previous[j] < previous[best]:
                best = j

            best_idx[j] = best

        dp[i] = costs[i] + prefix_cost
        back[i] = best_idx

    idx = int(np.argmin(dp[-1]))

    path_idx = np.empty(n, dtype=np.int32)
    path_idx[-1] = idx

    for i in range(n - 1, 0, -1):
        idx = int(back[i, idx])
        path_idx[i - 1] = idx

    label_distances = np.sqrt(
        costs[np.arange(n), path_idx]
    )

    return route_ids[path_idx], label_distances


@dataclass
class TripGeometry:
    trip: dict[str, Any]
    transformer: pyproj.Transformer
    gps: pd.DataFrame
    nodes_xy: np.ndarray
    arcs: pd.DataFrame
    lines: np.ndarray
    tree: STRtree
    points: np.ndarray
    labels: np.ndarray | None
    label_distance: np.ndarray | None
    road_bearing: np.ndarray
    segment_length: np.ndarray


def prepare_trip(
    trip_path: str | Path,
    include_reference_route: bool = False,
) -> TripGeometry:
    """Load and project one trip into a local metric CRS."""
    trip = read_trip(
        trip_path,
        include_reference_route=include_reference_route,
    )

    gps0 = trip["track"].copy()
    gps0["obs_idx"] = np.arange(
        len(gps0),
        dtype=np.int32,
    )

    good = (
        np.isfinite(
            gps0[["lon", "lat", "timestamp"]]
        ).all(axis=1)
        & gps0.lon.between(-180, 180)
        & gps0.lat.between(-90, 90)
    )

    gps = gps0.loc[good].copy().reset_index(drop=True)

    if gps.empty:
        raise ValueError(
            f"{trip['trip_id']}: no valid GPS observations"
        )

    lon0 = float(gps.lon.median())
    lat0 = float(gps.lat.median())

    zone = int(np.floor((lon0 + 180) / 6) + 1)
    zone = max(1, min(60, zone))

    epsg = (
        32600 if lat0 >= 0 else 32700
    ) + zone

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    gx, gy = transformer.transform(
        gps.lon.to_numpy(),
        gps.lat.to_numpy(),
    )

    points = shapely.points(
        np.asarray(gx),
        np.asarray(gy),
    )

    nodes = trip["nodes"]

    nx, ny = transformer.transform(
        nodes.lon.to_numpy(),
        nodes.lat.to_numpy(),
    )

    nodes_xy = np.column_stack([nx, ny])

    arcs = trip["arcs"].copy()

    endpoints_ok = (
        arcs.from_node.between(0, len(nodes_xy) - 1)
        & arcs.to_node.between(0, len(nodes_xy) - 1)
    )

    arcs = arcs.loc[endpoints_ok].copy()

    if arcs.empty:
        raise ValueError(
            f"{trip['trip_id']}: no valid road arcs"
        )

    arc_idx = arcs.index.to_numpy(dtype=np.int64)

    start = nodes_xy[
        arcs.from_node.to_numpy()
    ]

    end = nodes_xy[
        arcs.to_node.to_numpy()
    ]

    lines = shapely.linestrings(
        np.stack([start, end], axis=1)
    )

    road_bearing = _bearing(
        end[:, 0] - start[:, 0],
        end[:, 1] - start[:, 1],
    )

    segment_length = np.hypot(
        end[:, 0] - start[:, 0],
        end[:, 1] - start[:, 1],
    )

    # Preserve original directed arc IDs.
    arcs["arc_id"] = arc_idx

    tree = STRtree(lines)

    labels = None
    label_distance = None

    if include_reference_route:
        route = trip["route"]

        valid_route = (
            (route >= 0)
            & (route < len(trip["arcs"]))
        )

        route_ids = route[valid_route]

        line_lookup = np.full(
            len(trip["arcs"]),
            -1,
            dtype=np.int64,
        )

        line_lookup[arc_idx] = np.arange(
            len(arc_idx)
        )

        route_ids = route_ids[
            line_lookup[route_ids] >= 0
        ]

        route_lines = lines[
            line_lookup[route_ids]
        ]

        labels, label_distance = _route_labels(
            points,
            route_lines,
            route_ids,
        )

    return TripGeometry(
        trip=trip,
        transformer=transformer,
        gps=gps,
        nodes_xy=nodes_xy,
        arcs=arcs,
        lines=lines,
        tree=tree,
        points=points,
        labels=labels,
        label_distance=label_distance,
        road_bearing=road_bearing,
        segment_length=segment_length,
    )


def _obs_features(
    g: TripGeometry,
) -> pd.DataFrame:
    """Calculate observation-level GPS features."""
    gps = g.gps

    xy = np.asarray(
        shapely.get_coordinates(g.points),
        dtype=float,
    )

    timestamps = gps.timestamp.to_numpy()

    dt_prev = np.r_[
        np.nan,
        np.diff(timestamps),
    ]

    dt_next = np.r_[
        np.diff(timestamps),
        np.nan,
    ]

    dx_prev = np.r_[
        np.nan,
        np.diff(xy[:, 0]),
    ]

    dy_prev = np.r_[
        np.nan,
        np.diff(xy[:, 1]),
    ]

    d_prev = np.hypot(
        dx_prev,
        dy_prev,
    )

    d_next = np.r_[
        d_prev[1:],
        np.nan,
    ]

    b_prev = _bearing(
        dx_prev,
        dy_prev,
    )

    b_prev[~np.isfinite(dx_prev)] = np.nan

    b_next = np.r_[
        b_prev[1:],
        np.nan,
    ]

    dx_next = np.r_[
        np.diff(xy[:, 0]),
        np.nan,
    ]

    dy_next = np.r_[
        np.diff(xy[:, 1]),
        np.nan,
    ]

    dx_center = (
        np.nan_to_num(dx_prev)
        + np.nan_to_num(dx_next)
    )

    dy_center = (
        np.nan_to_num(dy_prev)
        + np.nan_to_num(dy_next)
    )

    bearing = _bearing(
        dx_center,
        dy_center,
    )

    bearing[
        np.hypot(dx_center, dy_center) < 0.5
    ] = np.nan

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        speed = d_next / dt_next

    speed[
        (dt_next <= 0)
        | ~np.isfinite(speed)
    ] = np.nan

    speed_arr = np.asarray(
        speed,
        dtype=float,
    )

    stopped = (
        np.isfinite(speed_arr)
        & (speed_arr < 1.0)
    ).astype(np.int8)

    features = pd.DataFrame({
        "gps_bearing_deg": bearing,
        "vehicle_speed_mps": speed_arr,
        "time_diff_s": dt_prev,
        "stopped_flag": stopped,
        "prev_obs_distance_m": d_prev,
        "next_obs_distance_m": d_next,
        "prev_time_diff_s": dt_prev,
        "next_time_diff_s": dt_next,
        "prev_bearing_deg": b_prev,
        "next_bearing_deg": b_next,
    })

    return (
        features
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def select_candidate_indices(
    g: TripGeometry,
    point,
    search_radius_m: float = 50.0,
    min_candidates: int = 0,
    max_radius_m: float = 150.0,
    fallback_nearest: bool = True,
) -> np.ndarray:
    """Find candidate road arcs with optional adaptive expansion.

    min_candidates=0:
        Original fixed-radius behavior.

    min_candidates>0:
        Expand the radius until enough candidates are found
        or max_radius_m is reached.

    This function never accesses reference labels.
    """
    if not np.isfinite(search_radius_m) or search_radius_m <= 0:
        raise ValueError(
            "search_radius_m must be positive and finite"
        )

    if not np.isfinite(max_radius_m) or max_radius_m < search_radius_m:
        raise ValueError(
            "max_radius_m must be finite and >= search_radius_m"
        )

    if not isinstance(min_candidates, (int, np.integer)) or min_candidates < 0:
        raise ValueError(
            "min_candidates must be a nonnegative integer"
        )

    radius = float(search_radius_m)

    while True:
        candidates = np.asarray(
            g.tree.query(
                point,
                predicate="dwithin",
                distance=radius,
            ),
            dtype=np.int64,
        )

        if (
            len(candidates) >= min_candidates
            or radius >= max_radius_m
        ):
            break

        radius = min(
            radius * 2.0,
            max_radius_m,
        )

    if len(candidates) == 0 and fallback_nearest:
        candidates = np.asarray(
            [g.tree.nearest(point)],
            dtype=np.int64,
        )

    return np.unique(candidates)


def build_candidates(
    trip_path: str | Path,
    search_radius_m: float = 50.0,
    fallback_nearest: bool = True,
    include_labels: bool = False,
    min_candidates: int = 0,
    max_radius_m: float = 150.0,
) -> pd.DataFrame:
    """Build candidate rows for every valid GPS observation.

    Default arguments preserve the existing 50 m pipeline.

    To enable adaptive search, use for example:
        min_candidates=5
        max_radius_m=150.0

    Reference labels are used only when include_labels=True.
    They never influence candidate selection.
    """
    g = prepare_trip(
        trip_path,
        include_reference_route=include_labels,
    )

    obs = _obs_features(g)

    rows: list[dict[str, Any]] = []

    arc_ids_all = g.arcs.arc_id.to_numpy(
        dtype=np.int64
    )

    for i, point in enumerate(g.points):
        candidates = select_candidate_indices(
            g,
            point,
            search_radius_m=search_radius_m,
            min_candidates=min_candidates,
            max_radius_m=max_radius_m,
            fallback_nearest=fallback_nearest,
        )

        if candidates.size == 0:
            continue

        distances = np.asarray(
            shapely.distance(
                g.lines[candidates],
                point,
            ),
            dtype=float,
        )

        arc_ids = arc_ids_all[candidates]

        bearing_diff = np.abs(
            (
                obs.gps_bearing_deg.iloc[i]
                - g.road_bearing[candidates]
                + 180.0
            ) % 360.0 - 180.0
        )

        prev_point = (
            g.points[i - 1]
            if i > 0
            else None
        )

        next_point = (
            g.points[i + 1]
            if i + 1 < len(g.points)
            else None
        )

        prev_distances = (
            np.asarray(
                shapely.distance(
                    g.lines[candidates],
                    prev_point,
                ),
                dtype=float,
            )
            if prev_point is not None
            else np.zeros(len(candidates))
        )

        next_distances = (
            np.asarray(
                shapely.distance(
                    g.lines[candidates],
                    next_point,
                ),
                dtype=float,
            )
            if next_point is not None
            else np.zeros(len(candidates))
        )

        for j, arc_id in enumerate(arc_ids):
            row = {
                "trip_id": g.trip["trip_id"],
                "obs_idx": int(
                    g.gps.obs_idx.iloc[i]
                ),
                "arc_id": int(arc_id),
                "distance_m": float(
                    distances[j]
                ),
                "road_bearing_deg": float(
                    g.road_bearing[candidates[j]]
                ),
                "bearing_diff_deg": float(
                    bearing_diff[j]
                ) if np.isfinite(bearing_diff[j]) else 180.0,
                "segment_length_m": float(
                    g.segment_length[candidates[j]]
                ),
                "candidate_within_radius": int(
                    distances[j] <= search_radius_m
                ),
                "label": (
                    int(arc_id == g.labels[i])
                    if include_labels
                    else None
                ),
            }

            for column in obs.columns:
                row[column] = float(
                    obs[column].iloc[i]
                )

            # Preserve the existing feature semantics.
            row["prev_obs_distance_m"] = float(
                prev_distances[j]
            )

            row["next_obs_distance_m"] = float(
                next_distances[j]
            )

            rows.append(row)

    columns = [
        "trip_id",
        "obs_idx",
        "arc_id",
        *FEATURE_COLUMNS,
        "label",
    ]

    return pd.DataFrame.from_records(
        rows,
        columns=columns,
    )


def candidate_miss_rates(
    g: TripGeometry,
    radii_m: tuple[int, ...] = (
        25, 50, 75, 100, 150
    ),
) -> list[dict[str, Any]]:
    """Measure reference-arc candidate misses at fixed radii.

    This preserves the original radius-comparison diagnostic.
    It does not measure the adaptive min_candidates strategy.
    """
    if g.labels is None:
        raise ValueError(
            "Reference labels are required for miss-rate analysis"
        )

    rows = []

    arc_ids = g.arcs.arc_id.to_numpy(
        dtype=np.int64
    )

    for radius in radii_m:
        misses = 0

        for i, point in enumerate(g.points):
            candidates = select_candidate_indices(
                g,
                point,
                search_radius_m=radius,
                min_candidates=0,
                max_radius_m=max(150.0, float(radius)),
                fallback_nearest=True,
            )

            candidate_arc_ids = arc_ids[candidates]

            if g.labels[i] not in candidate_arc_ids:
                misses += 1

        rows.append({
            "trip_id": g.trip["trip_id"],
            "search_radius_m": radius,
            "n_observations": len(g.points),
            "candidate_misses": misses,
            "candidate_miss_rate": (
                misses / len(g.points)
                if len(g.points)
                else 0.0
            ),
        })

    return rows
