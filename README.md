# PathSnap

A distributed machine learning system for GPS map-matching and route reconstruction using candidate classification, Hidden Markov Models (HMM), and Viterbi decoding.

## Project Overview

PathSnap reconstructs vehicle routes from noisy GPS readings by identifying the most likely road segments travelled.

The system uses a two-stage pipeline:

1. **Candidate Classifier:** Generates and scores nearby road segments for each GPS observation.
2. **HMM + Viterbi Decoder:** Uses road-network connectivity to reconstruct the most likely sequence of road segments.

## Project Structure

The repository is organised into the following directories:

- `data/` – Raw and processed datasets.
- `src/features/` – Feature engineering and candidate generation.
- `src/models/` – Machine learning models and training.
- `src/hmm/` – Hidden Markov Model and Viterbi decoding.
- `src/api/` – FastAPI backend.
- `ui/` – User interface.
- `experiments/` – Model experiments and hyperparameter tuning.
- `reports/` – Project reports and visualisations.
- `configs/` – Shared configuration files.
- `tests/` – Unit and integration tests.

## How to Run

Installation, execution and deployment instructions will be added as the project implementation is completed.

## Dataset Setup

Place the raw development dataset locally inside:

`data/raw/`

The raw dataset directory is gitignored and must not be committed to the public repository.

The evaluator-testing dataset is maintained exclusively by Person 4 on Laptop 1. It must remain local and must never be uploaded to GitHub.

## Team Development

All team members must work on their assigned branches.

Changes to the `main` branch must be submitted through pull requests.

Refer to [CONTRACTS.md](CONTRACTS.md) for the shared development contracts and interfaces.