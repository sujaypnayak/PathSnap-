from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except Exception as exc:
    # Keep preprocessing usable for EDA and feature verification where a native
    # Parquet extension is unavailable; never disguise CSV bytes as Parquet.
    pa = pq = None
    PARQUET_AVAILABLE = False
    PARQUET_IMPORT_ERROR = repr(exc)
import shapely
import folium
from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.candidates import FEATURE_COLUMNS, build_candidates, prepare_trip, miss_rates

DATA_ROOT = ROOT / "miscallaneous" / "student folder" / "Student folder" / "Student folder"
OUT = ROOT / "preprocessing"
RADIUS_M = 50.0


def _components(n_nodes: int, arcs: pd.DataFrame) -> tuple[int, int]:
    parent = list(range(n_nodes))
    rank = [0] * n_nodes
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    degree = [0] * n_nodes
    for a, b in arcs[["from_node", "to_node"]].itertuples(index=False, name=None):
        union(int(a), int(b)); degree[int(a)] += 1; degree[int(b)] += 1
    comps = len({find(i) for i in range(n_nodes)})
    return comps, sum(d == 0 for d in degree)


def _eda_row(path: Path, geom=None) -> dict:
    is_train = "train" in path.parts
    g = geom or prepare_trip(path, include_reference_route=is_train)
    trip = g.trip
    gps0, nodes, arcs = trip["track"], trip["nodes"], g.arcs
    route = trip["route"] if is_train else np.array([], dtype=np.int64)
    times = g.gps.timestamp.to_numpy()
    gaps = np.diff(times)
    xy = np.asarray(shapely.get_coordinates(g.points))
    speeds = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])) / gaps
    speeds[(gaps <= 0) | ~np.isfinite(speeds)] = np.nan
    stop_mask = np.isfinite(speeds) & (speeds < 1.0)
    starts = np.flatnonzero(stop_mask & ~np.r_[False, stop_mask[:-1]])
    ends = np.flatnonzero(stop_mask & ~np.r_[stop_mask[1:], False])
    stop_durations = [float(np.nansum(gaps[s:e+1])) for s, e in zip(starts, ends)]
    comps, isolates = _components(len(nodes), arcs)
    route_ok = ((route[:-1] >= 0) & (route[:-1] < len(trip["arcs"]))
                & (route[1:] >= 0) & (route[1:] < len(trip["arcs"])))
    raw_arcs = trip["arcs"].to_numpy()
    ridx = route[:-1][route_ok]
    rnext = route[1:][route_ok]
    connected = int(np.count_nonzero(raw_arcs[ridx, 1] == raw_arcs[rnext, 0])) if len(ridx) else 0
    reverse = {(int(a), int(b)) for a, b in arcs[["from_node", "to_node"]].itertuples(index=False, name=None)}
    bidir = sum((b, a) in reverse for a, b in reverse) / max(1, len(reverse))
    return {
        "trip_id": trip["trip_id"], "partition": "train" if "train" in path.parts else "student_test",
        "files_present": sum((path / f"{trip['trip_id']}{ext}").is_file() for ext in (".track", ".nodes", ".arcs", ".route")),
        "track_file_bytes": (path / f"{trip['trip_id']}.track").stat().st_size,
        "nodes_file_bytes": (path / f"{trip['trip_id']}.nodes").stat().st_size,
        "arcs_file_bytes": (path / f"{trip['trip_id']}.arcs").stat().st_size,
        "route_file_bytes": (path / f"{trip['trip_id']}.route").stat().st_size if is_train else None,
        "track_rows": len(gps0), "valid_gps_rows": len(g.gps), "invalid_gps_rows": len(gps0)-len(g.gps),
        "nodes": len(nodes), "arcs": len(trip["arcs"]), "valid_arcs": len(arcs), "route_arcs": len(route) if is_train else None,
        "missing_files": 0, "gps_missing_values": int(gps0.isna().sum().sum()),
        "node_missing_values": int(nodes.isna().sum().sum()),
        "invalid_node_coordinates": int((~(nodes.lon.between(-180, 180) & nodes.lat.between(-90, 90))).sum()),
        "invalid_arc_endpoints": int(len(trip["arcs"]) - len(arcs)),
        "invalid_route_arc_ids": int(np.count_nonzero((route < 0) | (route >= len(trip["arcs"])) )) if is_train else None ,
        "timestamp_nonincreasing_steps": int(np.count_nonzero(gaps <= 0)),
        "gap_median_s": float(np.nanmedian(gaps)) if len(gaps) else 0.0,
        "gap_p95_s": float(np.nanpercentile(gaps, 95)) if len(gaps) else 0.0,
        "gap_max_s": float(np.nanmax(gaps)) if len(gaps) else 0.0,
        "stationary_steps_lt_1mps": int(np.count_nonzero(np.isfinite(speeds) & (speeds < 1.0))),
        "stationary_period_count": int(len(starts)),
        "stationary_max_period_s": max(stop_durations, default=0.0),
        "gps_lon_min": float(g.gps.lon.min()), "gps_lon_max": float(g.gps.lon.max()),
        "gps_lat_min": float(g.gps.lat.min()), "gps_lat_max": float(g.gps.lat.max()),
        "weak_components": comps, "isolated_nodes": isolates,
        "reverse_arc_fraction": float(bidir),
        "route_connected_pairs": connected if is_train else None,
        "route_pair_count": int(len(ridx)) if is_train else None,
        "route_connectivity_rate": connected / max(1, len(ridx)) if is_train else None,
        "mean_segment_length_m": float(np.mean(g.segment_length)),
        "median_segment_length_m": float(np.median(g.segment_length)),
    }


def _write_map(trip_path: Path, html_path: Path, include_reference_route: bool = True) -> None:
    from features.candidates import read_trip
    trip = read_trip(trip_path, include_reference_route=include_reference_route)
    g = prepare_trip(trip_path, include_reference_route=include_reference_route)
    center = [float(g.gps.lat.mean()), float(g.gps.lon.mean())]
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    gps_layer = FeatureGroup(name="GPS observations", show=True)
    for row in g.gps.itertuples():
        CircleMarker([row.lat, row.lon], radius=2, color="#1565c0", fill=True,
                     fill_opacity=.65, popup=f"obs_idx={row.obs_idx}").add_to(gps_layer)
    gps_layer.add_to(m)
    raw_arcs = trip["arcs"]
    nodes = trip["nodes"]
    road_features = []
    # Show the same relevant neighborhood used by candidate generation, avoiding
    # an unwieldy rendering of a trip's entire raw road network.
    pairs = g.tree.query(g.points, predicate="dwithin", distance=RADIUS_M)
    nearby_arc_positions = np.unique(pairs[1]) if pairs.size else np.array([], dtype=np.int64)
    nearby_arc_ids = g.arcs.arc_id.to_numpy(dtype=np.int64)[nearby_arc_positions]
    for arc_id in nearby_arc_ids:
        u, v = raw_arcs.iloc[int(arc_id)]
        road_features.append({"type": "Feature", "properties": {}, "geometry": {
            "type": "LineString", "coordinates": [[float(nodes.lon.iloc[u]), float(nodes.lat.iloc[u])],
                                                        [float(nodes.lon.iloc[v]), float(nodes.lat.iloc[v])]]}})
    road_layer = folium.GeoJson({"type": "FeatureCollection", "features": road_features},
                                name=f"Nearby road candidates within {RADIUS_M:g} m ({len(road_features)} arcs)",
                                style_function=lambda _: {"color": "#9e9e9e", "weight": 1, "opacity": .35})
    road_layer.add_to(m)
    if include_reference_route:
        route_layer = FeatureGroup(name="Reference route", show=True)
        coords = []
        for arc_id in trip["route"]:
            if 0 <= arc_id < len(raw_arcs):
                u, v = raw_arcs.iloc[int(arc_id)]
                if 0 <= u < len(nodes) and 0 <= v < len(nodes):
                    coords.append([float(nodes.lat.iloc[u]), float(nodes.lon.iloc[u])])
                    coords.append([float(nodes.lat.iloc[v]), float(nodes.lon.iloc[v])])
        if coords:
            PolyLine(coords, color="#2e7d32", weight=4, opacity=.9, tooltip="Reference route").add_to(route_layer)
        route_layer.add_to(m)
    LayerControl(collapsed=False).add_to(m)
    m.save(str(html_path))


def _write_partition(paths: list[Path], output: Path, partition: str,
                     miss_rows: list[dict], eda_rows: list[dict], radius: float) -> tuple[int, int]:
    writer = None
    csv_fallback = output.with_suffix(".csv") if not PARQUET_AVAILABLE else None
    if csv_fallback:
        csv_fallback.unlink(missing_ok=True)
    trip_count = row_count = 0
    for path in paths:
        df = build_candidates(path, search_radius_m=radius, include_labels=(partition in ("train", "pilot")))
        if df.empty:
            raise RuntimeError(f"{path.name}: candidate generator produced no rows")
        if PARQUET_AVAILABLE:
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
        else:
            df.to_csv(csv_fallback, mode="a", index=False, header=not csv_fallback.exists())
        trip_count += 1
        row_count += len(df)
        print(f"{partition}: {path.name}: {len(df):,} candidate rows", flush=True)
        g = prepare_trip(path, include_reference_route=(partition in ("train", "pilot")))
        if partition in ("train", "pilot"):
            miss_part = miss_rates(g)
            for miss_row in miss_part:
                miss_row["partition"] = partition
            miss_rows.extend(miss_part)
        eda_rows.append(_eda_row(path, g))
    if writer is not None:
        writer.close()
    return trip_count, row_count


def run(pilot: int = 0, maps_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = sorted((DATA_ROOT / "train").iterdir())
    student = sorted((DATA_ROOT / "student_test").iterdir())
    if len(train) != 80 or len(student) != 10:
        raise RuntimeError(f"Expected 80 train and 10 student_test trips; found {len(train)} and {len(student)}")
    if maps_only:
        _write_map(train[0], OUT / "sample_train_map.html")
        _write_map(student[0], OUT / "sample_student_test_map.html", include_reference_route=False)
        def data_rows(path: Path) -> int:
            with path.open("rb") as stream:
                return max(0, sum(1 for _ in stream) - 1)
        train_csv, student_csv = OUT / "train_candidates.csv", OUT / "student_test_candidates.csv"
        summary = {
            "train_trips": 80, "train_candidate_rows": data_rows(train_csv),
            "student_test_trips": 10, "student_test_candidate_rows": data_rows(student_csv),
            "radius_m": RADIUS_M, "feature_columns": FEATURE_COLUMNS, "random_seed": 42,
            "source_partitions": {"train": "train/ (80 trips)", "student_test": "student_test/ (10 trips)"},
            "evaluator_data_accessed": False, "parquet_available": PARQUET_AVAILABLE,
        }
        if not PARQUET_AVAILABLE:
            summary["parquet_error"] = PARQUET_IMPORT_ERROR
        (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return
    (OUT / "feature_config.json").write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "schema": ["trip_id", "obs_idx", "arc_id", *FEATURE_COLUMNS, "label"],
        "search_radius_m": RADIUS_M,
        "crs_policy": "Per-trip WGS84 to local UTM zone inferred from median GPS location",
        "random_seed": 42,
        "label_policy": "Route arc minimizing point distance under monotone route-order assignment; label=1 for matching candidate arc_id",
    }, indent=2), encoding="utf-8")
    if pilot:
        pilot_rows: list[dict] = []; pilot_eda: list[dict] = []
        pilot_path = OUT / ("pilot_train_candidates.parquet" if PARQUET_AVAILABLE else "pilot_train_candidates.csv")
        _write_partition(train[:pilot], pilot_path, "pilot", pilot_rows, pilot_eda, RADIUS_M)
        pd.DataFrame(pilot_eda).to_csv(OUT / "pilot_eda_summary.csv", index=False)
        pd.DataFrame(pilot_rows).to_csv(OUT / "pilot_candidate_miss_rates.csv", index=False)
        print(f"Pilot complete: {pilot} trips; {pilot_path}", flush=True)
        if pilot < len(train):
            return
    eda_rows: list[dict] = []; miss_rows: list[dict] = []
    train_path = OUT / ("train_candidates.parquet" if PARQUET_AVAILABLE else "train_candidates.csv")
    student_path = OUT / ("student_test_candidates.parquet" if PARQUET_AVAILABLE else "student_test_candidates.csv")
    nt, nr = _write_partition(train, train_path, "train", miss_rows, eda_rows, RADIUS_M)
    ns, nsr = _write_partition(student, student_path, "student_test", miss_rows, eda_rows, RADIUS_M)
    pd.DataFrame(eda_rows).to_csv(OUT / "eda_summary.csv", index=False)
    miss = pd.DataFrame(miss_rows)
    totals = miss.groupby("search_radius_m", as_index=False).agg(
        n_observations=("n_observations", "sum"), candidate_misses=("candidate_misses", "sum"))
    totals["trip_id"] = "__ALL_TRAIN__"
    totals["partition"] = "train_only"
    totals["candidate_miss_rate"] = totals.candidate_misses / totals.n_observations
    pd.concat([miss, totals], ignore_index=True).to_csv(OUT / "candidate_miss_rates.csv", index=False)
    _write_map(train[0], OUT / "sample_train_map.html")
    _write_map(student[0], OUT / "sample_student_test_map.html", include_reference_route=False)
    summary = {
        "train_trips": nt, "train_candidate_rows": nr,
        "student_test_trips": ns, "student_test_candidate_rows": nsr,
        "radius_m": RADIUS_M, "feature_columns": FEATURE_COLUMNS,
        "random_seed": 42,
        "source_partitions": {"train": "train/ (80 trips)", "student_test": "student_test/ (10 trips)"},
        "evaluator_data_accessed": False,
        "parquet_available": PARQUET_AVAILABLE,
    }
    if not PARQUET_AVAILABLE:
        summary["parquet_error"] = PARQUET_IMPORT_ERROR
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0, help="Write a small training pilot and stop")
    parser.add_argument("--maps-only", action="store_true", help="Generate maps and summary from completed CSV outputs")
    args = parser.parse_args()
    run(args.pilot, args.maps_only)
