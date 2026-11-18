
import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

function FitMap({ points, routeGeometry }) {
  const map = useMap();

  useEffect(() => {
    const bounds = [
      ...points.map((point) => [point.lat, point.lng]),
      ...(routeGeometry || []),
    ];

    if (bounds.length > 0) {
      map.fitBounds(bounds, { padding: [30, 30] });
    }
  }, [points, routeGeometry, map]);

  return null;
}

function GPSMap({ points, routeGeometry }) {
  if (!points || points.length === 0) {
    return <p className="notice">No GPS coordinates available.</p>;
  }

  const gpsPositions = points.map((point) => [
    point.lat,
    point.lng,
  ]);

  const predictedPositions = (routeGeometry || []).map(
    (point) => [point[0], point[1]]
  );

  return (
    <div className="map-wrapper">
      <MapContainer
        center={gpsPositions[0]}
        zoom={13}
        scrollWheelZoom={true}
        style={{ height: "450px", width: "100%" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitMap
          points={points}
          routeGeometry={predictedPositions}
        />

        <Polyline
          positions={gpsPositions}
          pathOptions={{
            color: "#38bdf8",
            weight: 4,
            opacity: 0.8,
          }}
        />

        {predictedPositions.length > 0 && (
          <Polyline
            positions={predictedPositions}
            pathOptions={{
              color: "#f97316",
              weight: 6,
              opacity: 0.95,
            }}
          />
        )}

        <CircleMarker
          center={gpsPositions[0]}
          radius={7}
          pathOptions={{
            color: "#22c55e",
            fillColor: "#22c55e",
            fillOpacity: 1,
          }}
        >
          <Popup>Starting GPS point</Popup>
        </CircleMarker>

        <CircleMarker
          center={gpsPositions[gpsPositions.length - 1]}
          radius={7}
          pathOptions={{
            color: "#ef4444",
            fillColor: "#ef4444",
            fillOpacity: 1,
          }}
        >
          <Popup>Ending GPS point</Popup>
        </CircleMarker>
      </MapContainer>

      <div className="map-legend">
        <span>Blue: Original GPS track</span>
        <span>Orange: Predicted road route</span>
      </div>

      {predictedPositions.length === 0 && (
        <p className="notice">
          Predicted route will appear when a trained model is available.
        </p>
      )}
    </div>
  );
}

function App() {
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/dev/trips")
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load trips");
        return response.json();
      })
      .then((data) => setTrips(data.trips))
      .catch(() => setError("Could not connect to the PathSnap API."));
  }, []);

  async function handlePredict() {
    if (!selectedTrip) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ trip_id: selectedTrip }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="navbar">
        <h2>PathSnap</h2>
        <span>GPS Map Matching System</span>
      </header>

      <main className="container">
        <section className="hero">
          <p className="eyebrow">DISTRIBUTED MAP MATCHING</p>
          <h1>Reconstruct Your Route.</h1>
          <p>
            Select a GPS trip and let PathSnap process the route.
          </p>
        </section>

        <section className="panel">
          <h2>Trip Selection</h2>
          <p>Select a student testing trip to begin.</p>

          <select
            value={selectedTrip}
            onChange={(e) => {
              setSelectedTrip(e.target.value);
              setResult(null);
            }}
          >
            <option value="">Choose a trip ID</option>
            {trips.map((trip) => (
              <option key={trip} value={trip}>
                {trip}
              </option>
            ))}
          </select>

          <button
            onClick={handlePredict}
            disabled={!selectedTrip || loading}
          >
            {loading ? "Processing..." : "Predict Route"}
          </button>

          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <>
            <section className="panel results">
              <h2>Trip Results</h2>

              <div className="result-grid">
                <div>
                  <span>Trip ID</span>
                  <strong>{result.trip_id}</strong>
                </div>

                <div>
                  <span>Status</span>
                  <strong>{result.status}</strong>
                </div>

                <div>
                  <span>GPS Points</span>
                  <strong>{result.gps_points.length}</strong>
                </div>

                <div>
                  <span>Accuracy</span>
                  <strong>
                    {result.accuracy === null
                      ? "Not available"
                      : `${result.accuracy}%`}
                  </strong>
                </div>

                <div>
                  <span>Predicted Arcs</span>
                  <strong>{result.predicted_arc_ids.length}</strong>
                </div>
              </div>

              <p className="notice">{result.message}</p>
            </section>

            <section className="panel">
              <h2>Route Visualization</h2>
              <p>
                Compare the original GPS observations with the predicted road route.
              </p>
              <GPSMap
                points={result.gps_points}
                routeGeometry={result.route_geometry}
              />
            </section>
          </>
        )}
      </main>

      <footer>
        PathSnap | GPS Map Matching and Route Reconstruction
      </footer>
    </div>
  );
}

export default App;
