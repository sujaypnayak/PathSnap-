# PathSnap Frontend

React + Vite + React Leaflet interface for the PathSnap GPS map-matching system.

## Run locally

Start the FastAPI backend from the repository root:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd ui
npm ci
npm run dev
```

Open http://localhost:5173.

The Vite development proxy forwards `/api/*` to `http://127.0.0.1:8000/*`. Override the backend address using the `API_TARGET` environment variable if needed.

## Features

- Switch between Student Test and Evaluator datasets.
- Select a trip and run the prediction pipeline.
- Display GPS observations, predicted road route, start/end markers, and trip status.
- In Evaluator mode, display server-computed route F1, ordered route F1, and connectivity.
- Per-GPS-point accuracy is shown as unavailable because verified pointwise labels are not provided.

Raw datasets and the deployed model remain on the backend. The frontend receives prediction geometry and metrics, not reference routes.

## Checks

```powershell
npm run lint
npm run build
```

For dataset layout, backend setup, metrics and experiment details, see the [project README](../README.md).
