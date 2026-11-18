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
    "distance_m", "gps_bearing_deg", "road_bearing_deg", "bearing_diff_deg",
    "vehicle_speed_mps", "time_diff_s", "stopped_flag",
    "prev_obs_distance_m", "next_obs_distance_m", "prev_time_diff_s",
    "next_time_diff_s", "prev_bearing_deg", "next_bearing_deg",
    "segment_length_m", "candidate_within_radius",
]


def read_trip(trip_path: str | Path, include_reference_route: bool = False) -> dict[str, Any]:
    """Read the four headerless whitespace-separated files for one trip."""
    p = Path(trip_path)
    trip_id = p.name
    track = pd.read_csv(p / f"{trip_id}.track", sep=r"\s+", header=None,
                        names=["lon", "lat", "timestamp"], dtype="float64")
    nodes = pd.read_csv(p / f"{trip_id}.nodes", sep=r"\s+", header=None,
                        names=["lon", "lat"], dtype="float64")
    arcs = pd.read_csv(p / f"{trip_id}.arcs", sep=r"\s+", header=None,
                       names=["from_node", "to_node"], dtype="int64")
    route_path = p / f"{trip_id}.route"
    route = (pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"],
                         dtype="int64")["arc_id"].to_numpy()
             if include_reference_route else None)
    return {"trip_id": trip_id, "track": track, "nodes": nodes,
            "arcs": arcs, "route": route, "path": p}


def _bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Clockwise degrees from north, returned in [0, 360)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def _route_labels(points: np.ndarray, route_lines: np.ndarray,
                  route_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest route arc labels with a monotone route-order constraint.

    Dynamic programming allows staying on an arc or advancing to later route
    positions. This handles several GPS samples per arc without using GPS row
    indices as route indices. It also documents the route-as-reference assumption.
    """
    if len(route_ids) == 0:
        raise ValueError("Reference .route file has no arcs")
    # Drop invalid arc IDs while preserving the supplied sequence.
    if len(route_ids) == 0:
        raise ValueError("Reference route contains no valid arc IDs")
    # Compute point-to-route-segment distances in chunks to cap temporary memory.
    n, m = len(points), len(route_ids)
    costs = np.empty((n, m), dtype=np.float32)
    for start in range(0, n, 128):
        end = min(start + 128, n)
        d = shapely.distance(points[start:end, None], route_lines[None, :])
        costs[start:end] = np.asarray(d, dtype=np.float32) ** 2
    dp = np.empty_like(costs)
    back = np.zeros((n, m), dtype=np.int32)
    dp[0] = costs[0]
    for i in range(1, n):
        prefix_cost = np.minimum.accumulate(dp[i - 1])
        # Exact prefix argmin; stable earliest route position on ties.
        best_idx = np.empty(m, dtype=np.int32)
        best = 0
        for j in range(m):
            if dp[i - 1, j] < dp[i - 1, best]:
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
    return route_ids[path_idx], np.sqrt(costs[np.arange(n), path_idx])


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
    labels: np.ndarray
    label_distance: np.ndarray
    road_bearing: np.ndarray
    segment_length: np.ndarray


def prepare_trip(trip_path: str | Path, include_reference_route: bool = False) -> TripGeometry:
    trip = read_trip(trip_path, include_reference_route=include_reference_route)
    gps0 = trip["track"].copy()
    gps0["obs_idx"] = np.arange(len(gps0), dtype=np.int32)
    good = (np.isfinite(gps0[["lon", "lat", "timestamp"]]).all(axis=1)
            & gps0.lon.between(-180, 180) & gps0.lat.between(-90, 90))
    gps = gps0.loc[good].copy().reset_index(drop=True)
    if gps.empty:
        raise ValueError(f"{trip['trip_id']}: no valid GPS observations")
    lon0, lat0 = float(gps.lon.median()), float(gps.lat.median())
    zone = int(np.floor((lon0 + 180) / 6) + 1)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    gx, gy = transformer.transform(gps.lon.to_numpy(), gps.lat.to_numpy())
    points = shapely.points(np.asarray(gx), np.asarray(gy))
    nodes = trip["nodes"]
    nx, ny = transformer.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    nodes_xy = np.column_stack([nx, ny])
    arcs = trip["arcs"].copy()
    endpoints_ok = arcs.from_node.between(0, len(nodes_xy)-1) & arcs.to_node.between(0, len(nodes_xy)-1)
    arcs = arcs.loc[endpoints_ok].copy()
    arc_idx = arcs.index.to_numpy(dtype=np.int64)
    start = nodes_xy[arcs.from_node.to_numpy()]
    end = nodes_xy[arcs.to_node.to_numpy()]
    lines = shapely.linestrings(np.stack([start, end], axis=1))
    road_bearing = _bearing(end[:, 0]-start[:, 0], end[:, 1]-start[:, 1])
    seg_length = np.hypot(end[:, 0]-start[:, 0], end[:, 1]-start[:, 1])
    # Keep original row IDs as arc_id even when invalid endpoints are filtered.
    arcs["arc_id"] = arc_idx
    tree = STRtree(lines)
    if include_reference_route:
        route = trip["route"]
        valid_route = (route >= 0) & (route < len(trip["arcs"]))
        route_ids = route[valid_route]
        line_lookup = np.full(len(trip["arcs"]), -1, dtype=np.int64)
        line_lookup[arc_idx] = np.arange(len(arc_idx))
        route_keep = line_lookup[route_ids] >= 0
        route_ids = route_ids[route_keep]
        route_lines = lines[line_lookup[route_ids]]
        labels, label_distance = _route_labels(points, route_lines, route_ids)
    else:
        labels = label_distance = None
    return TripGeometry(trip, transformer, gps, nodes_xy, arcs, lines, tree,
                        points, labels, label_distance, road_bearing, seg_length)


def _obs_features(g: TripGeometry) -> pd.DataFrame:
    gps = g.gps
    xy = np.asarray(shapely.get_coordinates(g.points), dtype=float)
    n = len(xy)
    dt_prev = np.r_[np.nan, np.diff(gps.timestamp.to_numpy())]
    dt_next = np.r_[np.diff(gps.timestamp.to_numpy()), np.nan]
    dx_prev, dy_prev = np.r_[np.nan, np.diff(xy[:, 0])], np.r_[np.nan, np.diff(xy[:, 1])]
    d_prev = np.hypot(dx_prev, dy_prev)
    d_next = np.r_[d_prev[1:], np.nan]
    b_prev = _bearing(dx_prev, dy_prev)
    b_prev[~np.isfinite(dx_prev)] = np.nan
    b_next = np.r_[b_prev[1:], np.nan]
    # Centered heading is less noisy; endpoints use their single adjacent step.
    dx_next = np.r_[np.diff(xy[:, 0]), np.nan]
    dy_next = np.r_[np.diff(xy[:, 1]), np.nan]
    dx_center = np.nan_to_num(dx_prev) + np.nan_to_num(dx_next)
    dy_center = np.nan_to_num(dy_prev) + np.nan_to_num(dy_next)
    bearing = _bearing(dx_center, dy_center)
    bearing[(np.hypot(dx_center, dy_center) < 0.5)] = np.nan
    # Speed uses the displacement to the next sample, with valid positive time.
    speed = d_next / dt_next
    speed[(dt_next <= 0) | ~np.isfinite(speed)] = np.nan
    # Stop threshold is 1 m/s; isolated invalid/endpoint speeds are treated as moving.
    speed_arr = np.asarray(speed, dtype=float)
    stopped = (np.isfinite(speed_arr) & (speed_arr < 1.0)).astype(np.int8)
    f = pd.DataFrame({
        "gps_bearing_deg": bearing, "vehicle_speed_mps": speed_arr,
        "time_diff_s": dt_prev, "stopped_flag": stopped,
        "prev_obs_distance_m": d_prev, "next_obs_distance_m": d_next,
        "prev_time_diff_s": dt_prev, "next_time_diff_s": dt_next,
        "prev_bearing_deg": b_prev, "next_bearing_deg": b_next,
    })
    return f.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_candidates(trip_path: str | Path, search_radius_m: float = 50.0,
                     fallback_nearest: bool = True,
                     include_labels: bool = False) -> pd.DataFrame:
    """Build one row per (valid GPS observation, directed arc candidate).

    A Shapely STRtree finds segment candidates within the projected-meter radius.
    Both directed arcs are retained as separate row-indexed arc IDs.
    """
    g = prepare_trip(trip_path, include_reference_route=include_labels)
    obs = _obs_features(g)
    rows: list[dict[str, Any]] = []
    for i, pt in enumerate(g.points):
        cand = np.asarray(g.tree.query(pt, predicate="dwithin", distance=search_radius_m), dtype=np.int64)
        if cand.size == 0 and fallback_nearest:
            nearest = g.tree.nearest(pt)
            cand = np.asarray([nearest], dtype=np.int64)
        if cand.size == 0:
            continue
        dist = np.asarray(shapely.distance(g.lines[cand], pt), dtype=float)
        arc_ids = g.arcs.arc_id.to_numpy(dtype=np.int64)[cand]
        diff = np.abs((obs.gps_bearing_deg.iloc[i] - g.road_bearing[cand] + 180.0) % 360.0 - 180.0)
        prev_pt = g.points[i-1] if i > 0 else None
        next_pt = g.points[i+1] if i+1 < len(g.points) else None
        prev_d = np.asarray(shapely.distance(g.lines[cand], prev_pt), dtype=float) if prev_pt is not None else np.zeros(len(cand))
        next_d = np.asarray(shapely.distance(g.lines[cand], next_pt), dtype=float) if next_pt is not None else np.zeros(len(cand))
        for j, arc_id in enumerate(arc_ids):
            row = {"trip_id": g.trip["trip_id"], "obs_idx": int(g.gps.obs_idx.iloc[i]),
                   "arc_id": int(arc_id), "distance_m": float(dist[j]),
                   "road_bearing_deg": float(g.road_bearing[cand[j]]),
                   "bearing_diff_deg": float(diff[j]) if np.isfinite(diff[j]) else 180.0,
                   "segment_length_m": float(g.segment_length[cand[j]]),
                   "candidate_within_radius": int(dist[j] <= search_radius_m),
                   "label": int(arc_id == g.labels[i]) if include_labels else None}
            for c in obs.columns:
                row[c] = float(obs[c].iloc[i])
            row["prev_obs_distance_m"] = float(prev_d[j])
            row["next_obs_distance_m"] = float(next_d[j])
            rows.append(row)
    cols = ["trip_id", "obs_idx", "arc_id", *FEATURE_COLUMNS, "label"]
    return pd.DataFrame.from_records(rows, columns=cols)


def miss_rates(g: TripGeometry, radii_m: tuple[int, ...] = (25, 50, 75, 100, 150)) -> list[dict[str, Any]]:
    rows = []
    for radius in radii_m:
        misses = int(np.count_nonzero(g.label_distance > radius))
        rows.append({"trip_id": g.trip["trip_id"], "search_radius_m": radius,
                     "n_observations": len(g.points), "candidate_misses": misses,
                     "candidate_miss_rate": misses / len(g.points) if len(g.points) else 0.0})
    return rows

