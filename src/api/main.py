
from pathlib import Path

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

    trip_path = STUDENT_TEST_DIR / trip_id
    trip_file = trip_path / f"{trip_id}.track"

    if not trip_file.is_file():
        log_prediction_request(trip_id, "not_found")
        raise HTTPException(
            status_code=404,
            detail="Trip not found in the student testing dataset.",
        )

    gps_points = load_gps_points(trip_file)

    try:
        result = predict_trip(trip_path)

    except RuntimeError as error:
        log_prediction_request(trip_id, "model_pending")

        return {
            "trip_id": trip_id,
            "status": "model_pending",
            "message": str(error),
            "predicted_arc_ids": [],
            "gps_points": gps_points,
            "route_geometry": None,
            "accuracy": None,
        }

    except Exception as error:
        log_prediction_request(trip_id, "prediction_failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )

    log_prediction_request(trip_id, "prediction_completed")

    return {
        "trip_id": trip_id,
        "status": "success",
        "message": "Route prediction completed successfully.",
        "predicted_arc_ids": result["predicted_arc_ids"],
        "gps_points": gps_points,
        "route_geometry": None,
        "accuracy": None,
        "candidate_count": result["candidate_count"],
        "observation_count": result["observation_count"],
    }
