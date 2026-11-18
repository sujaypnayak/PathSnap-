
"""PathSnap API configuration using project-relative paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "Student folder"
)

TRAIN_DATA_DIR = DATASET_ROOT / "train"
STUDENT_TEST_DIR = DATASET_ROOT / "student_test"
EVALUATOR_DATA_DIR = DATASET_ROOT / "evaluator"

API_TITLE = "PathSnap API"
API_VERSION = "1.0.0"
