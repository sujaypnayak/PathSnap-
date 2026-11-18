
"""PathSnap FastAPI endpoints."""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.config import (
    STUDENT_TEST_DIR,
    EVALUATOR_DATA_DIR,
    API_TITLE,
    API_VERSION,
)
from src.api.logger import log_prediction_request
from src.api.prediction import predict_trip
from src.features.candidates import read_trip

# Reuse the exact definitions used for batch evaluation.
from scripts.evaluate_evaluator import (
    calculate_metrics,
    connectivity,
)


app = FastAPI(
    title=API_TITLE,
    description="Distributed GPS Map-Matching and Route Reconstruction",
    version=API_VERSION,
)


class PredictRequest(BaseModel):
    trip_id: str = Field(..., min_length=8, max_length=8)


def list_trip_ids(directory: Path):
    if not directory.exists():
        return []

    return sorted(
        folder.name
        for folder in directory.iterdir()
        if folder.is_dir()
        and (folder / f"{folder.name}.track").is_file()
    )


def load_gps_points(track_file: Path):
    gps_points = []

    try:
        with track_file.open("r", encoding="utf-8") as file:
            for line in file:
                values = line.strip().split()

                if len(values) < 2:
                    continue

                longitude = float(values[0])
                latitude = float(values[1])

                gps_points.append({
                    "lat": latitude,
                    "lng": longitude,
                })

    except (ValueError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read GPS track: {error}",
        )

    return gps_points


def run_prediction(trip_id, dataset_dir, evaluate=False):
    """Run prediction and optionally score against a reference route."""

    trip_path = dataset_dir / trip_id
    track_file = trip_path / f"{trip_id}.track"

    if not track_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Trip not found in the requested dataset.",
        )

    gps_points = load_gps_points(track_file)

    try:
        # Inference does not use reference-route labels.
        prediction = predict_trip(trip_path)

    except RuntimeError as error:
        return {
            "trip_id": trip_id,
            "status": "model_pending",
            "message": str(error),
            "predicted_arc_ids": [],
            "gps_points": gps_points,
            "route_geometry": None,
            "accuracy": None,
            "route_f1": None,
            "ordered_route_f1": None,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )

    response = {
        "trip_id": trip_id,
        "status": "success",
        "message": "Route prediction completed successfully.",
        "predicted_arc_ids": prediction["predicted_arc_ids"],
        "gps_points": gps_points,
        "route_geometry": prediction["route_geometry"],
        "accuracy": None,
        "route_f1": None,
        "ordered_route_f1": None,
        "candidate_count": prediction["candidate_count"],
        "observation_count": prediction["observation_count"],
    }

    if evaluate:
        try:
            trip = read_trip(
                trip_path,
                include_reference_route=True,
            )

            reference = trip["route"]
            arcs = trip["arcs"]

            if reference is None or len(reference) == 0:
                raise ValueError("Reference route is empty.")

            if np.any(reference < 0) or np.any(
                reference >= len(arcs)
            ):
                raise ValueError(
                    "Reference route contains invalid arc IDs."
                )

            metrics = calculate_metrics(
                prediction["predicted_arc_ids"],
                reference,
            )

            connection_metrics = connectivity(
                prediction["predicted_arc_ids"],
                arcs,
            )

            response.update({
                "route_precision": metrics["route_precision"],
                "route_recall": metrics["route_recall"],
                "route_f1": metrics["route_f1"],
                "ordered_route_precision": metrics["ordered_precision"],
                "ordered_route_recall": metrics["ordered_recall"],
                "ordered_route_f1": metrics["ordered_f1"],
                "connectivity_rate": connection_metrics[
                    "connectivity_rate"
                ],
                "metric_type": "route_level",
                "evaluation_note": (
                    "Reference routes contain ordered road arcs, "
                    "not verified per-GPS-point ground-truth labels. "
                    "GPS-point accuracy is unavailable."
                ),
            })

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Route evaluation failed: {error}",
            )

    return response


@app.get("/")
def root():
    return {
        "project": "PathSnap",
        "status": "running",
        "message": "PathSnap API is operational",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/trips")
def get_trips():
    trip_ids = list_trip_ids(EVALUATOR_DATA_DIR)

    if len(trip_ids) != 10:
        raise HTTPException(
            status_code=503,
            detail="Evaluator dataset is not configured.",
        )

    return {
        "trips": trip_ids,
        "count": len(trip_ids),
    }


@app.get("/dev/trips")
def get_dev_trips():
    trip_ids = list_trip_ids(STUDENT_TEST_DIR)

    if not trip_ids:
        raise HTTPException(
            status_code=503,
            detail="Student testing dataset not found.",
        )

    return {
        "trips": trip_ids,
        "count": len(trip_ids),
        "dataset": "student_test",
    }


@app.post("/predict")
def predict(request: PredictRequest):
    trip_id = request.trip_id

    log_prediction_request(trip_id, "received")

    if not trip_id.isdigit():
        log_prediction_request(trip_id, "invalid")
        raise HTTPException(
            status_code=400,
            detail="Trip ID must contain only digits.",
        )

    result = run_prediction(
        trip_id,
        STUDENT_TEST_DIR,
        evaluate=False,
    )

    log_prediction_request(
        trip_id,
        "prediction_completed"
        if result["status"] == "success"
        else "model_pending",
    )

    return result


@app.post("/evaluate")
def evaluate(request: PredictRequest):
    trip_id = request.trip_id

    if not trip_id.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Trip ID must contain only digits.",
        )

    return run_prediction(
        trip_id,
        EVALUATOR_DATA_DIR,
        evaluate=True,
    )
