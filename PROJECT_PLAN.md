# PathSnap: Project Plan

Team project plan for the PathSnap hackathon (Distributed Map-Matching & Route Reconstruction, UE24CS352A Machine Learning, PES University).

## 1. What the project is

GPS readings from a moving vehicle are noisy. In cities, buildings, signal interference and multipath effects make a car appear to be on a parallel street or jumping across blocks. **Map-matching** means snapping those noisy GPS points back onto the road segments the vehicle actually drove, then reconstructing the full route.

**PathSnap** is a distributed machine learning system that does this end to end: from raw GPS data, to trained models, to route reconstruction, to a deployed API and UI.

### What we are building

A two-stage pipeline:

1. **Candidate classifier.** For every GPS point we generate nearby road segments (arcs) as candidates and compute features for each one (distance to the arc, bearing difference between the vehicle heading and the road direction, speed, neighbouring-point context, etc.). A model scores how likely each candidate is to be the true segment. Labels come from the reference `.route` file.
2. **HMM + Viterbi decoder.** The classifier judges each point independently, so it can pick segments that are not connected. The HMM fixes this using road connectivity:
   - Hidden states: candidate road segments.
   - Emission probability: the classifier's score.
   - Transition probability: based on road-network connectivity (connected arcs are likely, disconnected ones are penalised).
   - Viterbi finds the most coherent sequence of segments.

We must show, with real results, whether classifier + HMM beats classifier alone, and discuss honestly where it does not help.

### The data

Each trip has four files (the sample trip is `00000085`):

| File | Contents |
| --- | --- |
| `.track` | GPS points: longitude, latitude, timestamp (about 1 Hz) |
| `.nodes` | Road network nodes: longitude, latitude (row index = node ID) |
| `.arcs` | Road segments: from\_node, to\_node (row index = arc ID) |
| `.route` | Reference route: ordered arc IDs the vehicle traversed |

A GPS row does **not** correspond to a row in any road-network file. We must build that relationship ourselves.

**Partitions:** 80 training trips, 10 student-testing trips, 10 evaluator-testing trips. The evaluator-testing trips are used only in the final demo and must never be touched during development.

### Findings from the sample trip

| Observation | Why it matters |
| --- | --- |
| 1,092 GPS points, 690 nodes, 1,554 arcs, 170-arc route | Basic scale of one trip |
| Every arc has a reverse arc (1,554 of 1,554) | Each road is two directed arcs. Heading and bearing features are essential to separate the true arc from its reverse twin. Candidate sets must include both directions. |
| All 169 consecutive route arc pairs connect (to\_node of one = from\_node of the next) | Gives a clean HMM transition rule. Verify across the 90 development trips only, never the evaluator ten. |
| 98.4% of time steps are 1 s, but the maximum gap is 70 s | The HMM must handle GPS gaps |
| Vehicle stops for a long stretch (around t=530 to 770 s) | Heading and speed are unreliable when stationary. Needs a stopped flag or similar. |
| Median arc length about 59 m; route about 9.95 km vs GPS path about 12.8 km | GPS jitter inflates the path by about 28%. Several points fall on each arc, so sequence context helps. |
| Network has 2 connected components | The decoder must handle candidates in disconnected parts of the network |
| Latitude about 53.85 | Degrees are not equal distances in lon and lat. Use a projected metric CRS. |

These come from one trip only. They must be re-checked across all trips and recorded in the EDA.

## 2. Deliverables

Each team submits exactly two deliverables, plus a working live demo.

### Deliverable 1: Public GitHub repository

Must contain:

| Component | What must be included |
| --- | --- |
| Data processing | Code to read, inspect, clean and preprocess raw data |
| Feature engineering | Complete feature extraction code |
| Model training | Scripts or notebooks for every model experimented with |
| Model evaluation | Comparison code, metrics, plots, confusion matrices |
| Hyperparameter optimisation | Bayesian optimisation code, search configs and results |
| HMM and Viterbi | Route reconstruction and decoding code |
| API | Complete FastAPI backend, including model loading |
| UI | Complete evaluator interface source |
| Deployment | Config files, dependencies, run instructions |
| Visualisations | Folium map generation and plotting scripts |
| Supporting results | CSVs, experiment logs, configs, result summaries |

Requirements: clearly named folders; a README with installation, execution and deployment instructions; saved model weights and configs (subject to size limits); no passwords, keys or secrets; code must match what is demonstrated at evaluation.

### Deliverable 2: Technical report (one consolidated PDF)

| Section | Content |
| --- | --- |
| 2.2 Dataset understanding and EDA | Preprocessing, missing values, GPS trajectory plots, road-network visualisations, at least one Folium map, which partition each analysis used |
| 3 Model experimentation | At least five models. Consolidated comparison table. Per-model answers (suitability, strengths and limitations, hyperparameters, loss, student-test results, over/underfitting evidence). Required plots. |
| 4 Bayesian optimisation | Search space, ranges, objective, trial count, best values, tuned vs untuned comparison |
| 5 HMM and Viterbi | Hidden states, emission and transition design, gap and missing-candidate handling, classifier-only vs classifier + HMM, cases where HMM did not help |
| 6 Folium visualisation | Maps showing GPS observations, road network, reference route and predicted route, with legend, layer control and a screenshot in the PDF. The HTML or generating code goes in the repo. |
| 7 API, UI and deployment | UI screenshots and screenshots of API responses on the receiving laptop's command line |
| 8 Academic integrity | Disclose AI assistance as required |

Every result in the PDF must be verifiable from the repository.

### Live evaluation

- API running on **Laptop 1** (model, HMM/Viterbi, road network, ground truth).
- UI running on **Laptop 2**, showing the ten evaluator trips.
- The evaluator picks any trip. The UI sends the trip ID to the API, the API runs the full pipeline, computes accuracy against its internal reference, and the UI shows the route and results.
- Must work for all ten trips. Ground truth must stay server-side. API logs must be shown on Laptop 1.
- Two evaluations on 18 November 2026: 11:30 AM (data preprocessing and feature engineering, model selection and training) and 3:00 PM (hyperparameter tuning, deployment, demonstration). Section 5 has the schedule.

### Minimum experiments required

- At least five models from different algorithm families, with recorded configurations, seeds, training and inference times.
- Bayesian hyperparameter optimisation with group-aware cross-validation on the 80 training trips (split by trip, never within a trip).
- Metrics: observation-level Top-1 accuracy, macro F1, precision, recall, log loss, candidate miss rate, route continuity, confusion matrices, training and inference time.

## 3. Rules that govern the work

- **No fabricated results.** Every number, graph and table must come from code in the repo. AI-generated analysis must not stand in for experiments we did not run.
- **No sharing** of code, weights, configs or results with other teams. Artifacts are compared across teams.
- **No leakage.** Evaluator-test data is never used for training, feature selection, model choice, hyperparameters or thresholds. Learned preprocessing is fitted on training data only.
- **No hardcoded predictions.** Ground truth never reaches the prediction pipeline or UI.
- **Viva.** Any team member can be asked about any part, including Viterbi, the Bayesian search, the API and the UI.
- **AI use is allowed** for debugging, boilerplate and explaining concepts. Do not use it to invent results, and do not submit code or explanations we cannot explain.
- Commit regularly under each person's own name. Repo history is compared.

## 4. Work division

The split is designed so all four people can work in parallel. The main dependency is the candidate dataset, so we agree on interfaces first and build against stubs until real data arrives.

### Phase 0 contracts (all four, before splitting)

Committed in a `CONTRACTS.md`:

1. Repo layout: `data/`, `src/features/`, `src/models/`, `src/hmm/`, `src/api/`, `ui/`, `experiments/`, `reports/`, `README.md`.
2. Candidate table schema (parquet), one row per (GPS point, candidate arc): `trip_id, obs_idx, arc_id, <feature columns>, label`, with a shared `feature_cols` list in a config file.
3. Model interface: sklearn-style `predict_proba(X)`, saved with joblib, fixed seed (42) everywhere.
4. Decoder interface: `decode(candidates_df, probs, road_graph)` returns one arc\_id per observation.
5. API schema: `GET /trips` returns the ten trip IDs. `POST /predict {trip_id}` returns predicted arcs, GPS points, route geometry and accuracy.
6. One shared `build_candidates(trip_path)` function used identically for training and API inference.
7. Split lists: the 80/10/10 trip IDs in one file. Person 4 alone holds the evaluator ten.
8. Coordinate system: a projected metric CRS, never raw degrees for distances.

### Phase 1: 9:30 AM to 11:30 AM (evaluation 1, 10 marks)

**Goal:** pass evaluation 1: data preprocessing and feature engineering (5 marks) and model selection and training (5 marks). Persons 3 and 4 support Persons 1 and 2 in this phase, because candidates and models are the critical path.

First 30 minutes (all four): read the supporting documents, create the repo and branches, and commit `CONTRACTS.md` (the eight contracts above).

| Person | Role | Tasks | Done when |
| --- | --- | --- | --- |
| 1 | Owns data preprocessing and feature engineering (5 marks) | Loaders for the four file types; EDA (missing values, time gaps, stationary periods, network connectivity); project to a metric CRS; candidate generation with a spatial index and both arc directions; features (distance to arc, bearing difference, speed, stopped flag, neighbour context); candidate miss rate vs search radius; one Folium map | Rough candidate file for about 5 trips pushed by 10:30. Full 80 + 10 trip file and frozen features by 11:00. |
| 2 | Owns model selection and training (5 marks) | Experiment harness first (GroupKFold by trip, metrics, logger, seeds, timing); then train at least 5 models from different families on the rough file, then on the full data; comparison table | Table with CV macro F1, student-test top-1 and macro F1, log loss and timings. Provisional model picked by 11:20. |
| 3 | Supports Persons 1 and 2, starts the decoder | Label generation (candidate arc matched to the `.route` arc) and the candidate-miss check for Person 1; route connectivity check on the 90 development trips; log-space Viterbi unit-tested on a toy graph | Labels verified, Viterbi skeleton passing its toy test |
| 4 | Supports Person 2, starts the API | Comparison plots and results-CSV writer for Person 2; FastAPI skeleton with a dummy model; UI against it; server-side ground truth handling; request logging; two-laptop connection test | Plots generated from real logs, dummy API reachable from Laptop 2 |

At 11:30 Persons 1 and 2 present. Confirm with the organisers whether all four must attend.

### Phase 2: 11:30 AM to 3:00 PM (tuning, deployment, report, submission)

**Goal:** a tuned model, a working two-laptop deployment, a verified report and a public repo, all submitted at 3:00 PM. This phase prepares the three 3:00 PM blocks: hyperparameter tuning (5), deployment (5) and demonstration (5).

| Person | Role | Tasks | Done when |
| --- | --- | --- | --- |
| 1 | Report and API support | Report sections 2.2 and 6 (EDA, Folium maps, screenshot, map HTML in the repo); help Person 4 run the same `build_candidates` inside the API; review others' code | Sections written from logs and verified by 2:45 |
| 2 | Model report and handover | Confusion matrices, comparison plots, per-model answers and overfitting evidence (sections 3.1 to 3.3); refit the final model with Person 3's best parameters; check logged numbers | Section 3 verified by 2:45 |
| 3 | Owns hyperparameter tuning (5 marks) | Optuna with group-aware CV on the 80 training trips; save the study; tuned vs untuned (CV macro F1, student-test top-1 and macro F1, log loss, inference time); tune decoder parameters; classifier-only vs HMM; route continuity; sections 4 and 5 | Tuned model and HMM config exported to Person 4 by 1:45 |
| 4 | Owns deployment (5 marks) | Real pipeline in the API; run all student-test trips; swap in the tuned model at 1:45; two-laptop test of all ten evaluator trips (deployment check only); API log screenshots; UI polish; README and requirements; assemble the PDF; repo public; secrets check; submit the Google form | All ten trips pass on two laptops by 2:30. Repo frozen and form submitted by 3:00. |

2:00 to 2:45, all four: rehearse the demo and practise the viva questions on your own part.

### Phase 3: 3:00 PM onwards (final evaluation, 15 marks)

**Goal:** score hyperparameter tuning (5), deployment (5) and demonstration (5). The Google form closes at 4:30 PM and the hackathon ends at 5:00 PM. No new features. Only fix bugs.

| Person | Role | Tasks | Done when |
| --- | --- | --- | --- |
| 1 | Explains data | Answers on dataset structure, preprocessing, features and the Folium maps; keeps the map HTML and screenshots ready | Evaluator questions answered |
| 2 | Explains models | Answers on model comparison, metrics and selection; keeps the comparison table and plots open | Evaluator questions answered |
| 3 | Explains tuning and decoding | Answers on Optuna, HMM, Viterbi and classifier-only vs HMM; keeps the saved study open | Evaluator questions answered |
| 4 | Runs the demo | API on Laptop 1 and UI on Laptop 2 with the API log window visible; confirms the form submission went through; is the only person who pushes bug fixes | All ten trips run live and logs shown |

Any bug fix must be pushed to GitHub before 4:30 PM so the repo matches what was demonstrated.

### Shared responsibilities

- One branch per person, merged through pull requests.
- Each person writes their report sections as they go, from their own logs.
- Everyone reads at least one other person's code. Hold a short walkthrough of each part before the viva.
- Keep a log of any AI assistance for the disclosure.

## 5. Task flow to completion

The event has two evaluations on 18 November 2026, worth 25 marks in total, so the schedule works backwards from 11:30 AM and 3:00 PM. Only data and models are due at 11:30 AM. Section 4 divides the work by phase; this section holds the deadlines, gates and fallback plan. Clock times assume work starts at 9:30 AM; shift every slot by the same amount if the team starts later.

### Evaluation deadlines

| Time | Assessed | Marks | Owner |
| --- | --- | --- | --- |
| 11:30 AM | Data preprocessing and feature engineering | 5 | Person 1 |
| 11:30 AM | Model selection and training | 5 | Person 2 |
| 3:00 PM | Hyperparameter tuning | 5 | Person 3 |
| 3:00 PM | Deployment per the instructions PDF | 5 | Person 4 |
| 3:00 PM | Demonstration | 5 | All four |

All files are submitted through the Google form at 3:00 PM. The form closes at 4:30 PM and the hackathon ends at 5:00 PM. Aim to submit at 3:00 PM and treat 4:30 PM as the hard limit, not the plan.

### Gates

| Time | Gate | If missed |
| --- | --- | --- |
| 10:30 AM | Rough candidate file pushed | Persons 2 and 3 keep working on stubs. Person 1 pushes whatever exists. |
| 11:20 AM | Provisional model chosen from real CV results | Take the best CV macro F1 so far. Do not wait for every model. |
| 11:30 AM | Features frozen, 5 models in the comparison table | Present what is done, and say plainly what is not. |
| 1:45 PM | Tuned model and HMM config exported | Use the best Optuna trials so far and report the true trial count. |
| 2:30 PM | All 10 evaluator trips run through the API on two laptops | Fix the API first. Report writing continues in parallel. |
| 3:00 PM | Repo frozen and public, form submitted | Submit what works. The form stays open until 4:30 PM. |

Running the evaluator ten through the API is a deployment check only. Their results must never be used to change features, models, hyperparameters or thresholds.

### If the team falls behind

Cut in this order:

1. Fewer Optuna trials, reporting the actual number.
2. Simpler UI: a trip dropdown, a map and an accuracy figure.
3. Fewer report plots beyond the required comparisons and one Folium map.

Never cut: five trained models, the Optuna search record, the classifier-only vs HMM comparison, and server-side ground truth.

### To confirm with the organisers

- Whether the whole team must attend the 11:30 AM evaluation or only the people presenting.
- What exactly the Google form asks for (the timeline's list is blank): repo link, report PDF, anything else.
- When lunch falls.

### Completion checklist

- [ ] Public GitHub repo with all required components and a working README
- [ ] At least five models trained and compared, with logged results
- [ ] Bayesian optimisation record retained, tuned vs untuned compared
- [ ] HMM/Viterbi implemented, classifier-only vs classifier + HMM compared
- [ ] Folium maps generated by repo code
- [ ] API on Laptop 1 and UI on Laptop 2 working for all ten trips
- [ ] Ground truth server-side only, no hardcoded predictions
- [ ] Single PDF report with every result verifiable from the repo
- [ ] AI assistance disclosed
- [ ] Every team member can explain every part
