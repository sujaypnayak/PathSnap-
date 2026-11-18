
"""Configurable log-space Viterbi decoder for road-segment map matching."""

import numpy as np


DISCONNECTED_PENALTY = -20.0
DEFAULT_EMISSION_WEIGHT = 1.0
DEFAULT_TRANSITION_WEIGHT = 1.0
DEFAULT_REVERSAL_PENALTY = 0.0


def decode(
    candidates_df,
    probs,
    road_graph,
    disconnected_penalty=DISCONNECTED_PENALTY,
    emission_weight=DEFAULT_EMISSION_WEIGHT,
    transition_weight=DEFAULT_TRANSITION_WEIGHT,
    reversal_penalty=DEFAULT_REVERSAL_PENALTY,
):
    """Find the most likely sequence of directed road arcs."""

    required_columns = {"obs_idx", "arc_id"}

    if not required_columns.issubset(candidates_df.columns):
        raise ValueError(
            f"candidates_df must contain {required_columns}"
        )

    if "trip_id" in candidates_df.columns:
        if candidates_df["trip_id"].nunique() != 1:
            raise ValueError(
                "decode() expects candidates from one trip."
            )

    if candidates_df.empty:
        raise ValueError("No candidate road segments were provided.")

    if not np.isfinite(disconnected_penalty) or disconnected_penalty > 0:
        raise ValueError(
            "disconnected_penalty must be finite and <= 0."
        )

    if not np.isfinite(emission_weight) or emission_weight < 0:
        raise ValueError(
            "emission_weight must be finite and >= 0."
        )

    if not np.isfinite(transition_weight) or transition_weight < 0:
        raise ValueError(
            "transition_weight must be finite and >= 0."
        )

    if not np.isfinite(reversal_penalty) or reversal_penalty < 0:
        raise ValueError(
            "reversal_penalty must be finite and >= 0."
        )

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
            raise ValueError(
                "Duplicate arc_id within an observation."
            )

        positions = group["_original_position"].to_numpy()
        emissions = probs[positions]

        total = emissions.sum()

        if total <= 0:
            raise ValueError(
                "Emission probabilities for an observation sum to zero."
            )

        emissions = emissions / total
        observations.append((arc_ids, emissions))

    min_probability = np.finfo(float).tiny

    first_arcs, first_emissions = observations[0]
    first_emissions = np.maximum(
        first_emissions, min_probability
    )
    first_emissions /= first_emissions.sum()

    dp = (
        np.log(np.full(len(first_arcs), 1.0 / len(first_arcs)))
        + emission_weight * np.log(first_emissions)
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
        backpointer = np.full(
            len(current_arcs), -1, dtype=int
        )

        for j, current_arc in enumerate(current_arcs):
            best_score = -np.inf
            best_previous = -1

            for i, previous_arc in enumerate(previous_arcs):
                connected = (
                    previous_arc == current_arc
                    or road_graph.has_edge(
                        previous_arc, current_arc
                    )
                )

                if connected:
                    reachable = sum(
                        previous_arc == candidate
                        or road_graph.has_edge(
                            previous_arc, candidate
                        )
                        for candidate in current_arcs
                    )

                    transition_score = -np.log(
                        max(1, reachable)
                    )
                else:
                    transition_score = disconnected_penalty

                # Reverse-direction arcs share the same
                # road endpoints in opposite order.
                if (
                    reversal_penalty > 0
                    and previous_arc != current_arc
                ):
                    previous_data = road_graph.nodes[
                        previous_arc
                    ]
                    current_data = road_graph.nodes[
                        current_arc
                    ]

                    if (
                        "from_node" in previous_data
                        and "to_node" in previous_data
                        and "from_node" in current_data
                        and "to_node" in current_data
                        and previous_data["from_node"]
                        == current_data["to_node"]
                        and previous_data["to_node"]
                        == current_data["from_node"]
                    ):
                        transition_score -= reversal_penalty

                score = (
                    dp[i]
                    + transition_weight * transition_score
                    + emission_weight * log_emissions[j]
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
            raise ValueError(
                "Could not reconstruct a valid path."
            )

        best_path.append(int(best_previous))

    best_path.reverse()

    return [
        observations[t][0][candidate_index]
        for t, candidate_index in enumerate(best_path)
    ]
