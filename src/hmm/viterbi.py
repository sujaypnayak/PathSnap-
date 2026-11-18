
"""Log-space Viterbi decoder for road-segment map matching."""

import numpy as np


DISCONNECTED_PENALTY = -20.0


def decode(candidates_df, probs, road_graph):
    """
    Find the most likely sequence of road segments.

    Disconnected transitions receive a large finite penalty.
    Staying on the same arc is always allowed.
    """

    required_columns = {"obs_idx", "arc_id"}

    if not required_columns.issubset(candidates_df.columns):
        raise ValueError(
            f"candidates_df must contain {required_columns}"
        )

    if "trip_id" in candidates_df.columns:
        if candidates_df["trip_id"].nunique() != 1:
            raise ValueError("decode() expects candidates from one trip.")

    if candidates_df.empty:
        raise ValueError("No candidate road segments were provided.")

    probs = np.asarray(probs, dtype=float).reshape(-1)

    if len(probs) != len(candidates_df):
        raise ValueError(
            "There must be exactly one probability per candidate row."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities must be finite numbers.")

    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    working = candidates_df.copy()
    working["_original_position"] = np.arange(len(working))
    working = working.sort_values("obs_idx", kind="stable")

    observations = []

    for _, group in working.groupby("obs_idx", sort=False):
        arc_ids = group["arc_id"].tolist()

        if len(set(arc_ids)) != len(arc_ids):
            raise ValueError("Duplicate arc_id within an observation.")

        positions = group["_original_position"].to_numpy()
        emissions = probs[positions]

        total = emissions.sum()

        if total <= 0:
            raise ValueError(
                "Emission probabilities for an observation sum to zero."
            )

        emissions = emissions / total
        observations.append((arc_ids, emissions))

    # Use a tiny finite floor so zero emissions do not create -inf.
    min_probability = np.finfo(float).tiny

    first_arcs, first_emissions = observations[0]
    first_emissions = np.maximum(first_emissions, min_probability)
    first_emissions /= first_emissions.sum()

    dp = (
        np.log(np.full(len(first_arcs), 1.0 / len(first_arcs)))
        + np.log(first_emissions)
    )

    backpointers = []

    for t in range(1, len(observations)):
        previous_arcs = observations[t - 1][0]
        current_arcs, current_emissions = observations[t]

        current_emissions = np.maximum(
            current_emissions, min_probability
        )
        current_emissions /= current_emissions.sum()
        log_emissions = np.log(current_emissions)

        next_dp = np.full(len(current_arcs), -np.inf)
        backpointer = np.full(len(current_arcs), -1, dtype=int)

        for j, current_arc in enumerate(current_arcs):
            best_score = -np.inf
            best_previous = -1

            for i, previous_arc in enumerate(previous_arcs):

                # Staying on the same arc is always allowed.
                connected = (
                    previous_arc == current_arc
                    or road_graph.has_edge(previous_arc, current_arc)
                )

                if connected:
                    reachable = sum(
                        previous_arc == candidate
                        or road_graph.has_edge(previous_arc, candidate)
                        for candidate in current_arcs
                    )

                    transition_score = -np.log(max(1, reachable))
                else:
                    transition_score = DISCONNECTED_PENALTY

                score = (
                    dp[i]
                    + transition_score
                    + log_emissions[j]
                )

                if score > best_score:
                    best_score = score
                    best_previous = i

            next_dp[j] = best_score
            backpointer[j] = best_previous

        dp = next_dp
        backpointers.append(backpointer)

    best_last = int(np.argmax(dp))
    best_path = [best_last]

    for backpointer in reversed(backpointers):
        best_previous = backpointer[best_path[-1]]

        if best_previous == -1:
            raise ValueError("Could not reconstruct a valid path.")

        best_path.append(int(best_previous))

    best_path.reverse()

    return [
        observations[t][0][candidate_index]
        for t, candidate_index in enumerate(best_path)
    ]
