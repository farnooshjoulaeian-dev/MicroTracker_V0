import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def calculate_step_alignment(p0, p1, p2):
    """
    Calculate directional alignment between two consecutive steps.
    """

    v1 = p1 - p0
    v2 = p2 - p1

    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denominator == 0:
        return np.nan

    return np.dot(v1, v2) / denominator


def link_detections(
    detections,
    fps=25,
    max_distance_um=5.6,
    DP_weight_um=0.25,
):
    """
    Link detections frame by frame using Hungarian assignment.

    Parameters
    ----------
    detections : pandas.DataFrame
        Detection table with at least: frame, x_um, y_um.

    Returns
    -------
    links : pandas.DataFrame
        Accepted frame-to-frame links.

    tracks : pandas.DataFrame
        Detections with assigned track_id.
    """

    tracked_detections = detections.copy()
    tracked_detections["track_id"] = np.nan

    track_history = {}
    next_track_id = 0

    frame_numbers = sorted(detections["frame"].unique())

    first_frame_number = frame_numbers[0]
    first_frame = detections[detections["frame"] == first_frame_number]

    for detection_index in first_frame.index:
        tracked_detections.loc[detection_index, "track_id"] = next_track_id
        track_history[next_track_id] = [detection_index]
        next_track_id += 1

    all_links = []
    invalid_assignment_cost = 1e6

    for frame_number in frame_numbers[1:]:

        current_frame = detections[detections["frame"] == frame_number]

        if len(current_frame) == 0:
            continue

        current_positions = current_frame[["x_um", "y_um"]].to_numpy()
        current_indices = current_frame.index.to_numpy()

        active_tracks = []

        for track_id, history in track_history.items():
            last_detection = history[-1]
            last_frame = tracked_detections.loc[last_detection, "frame"]

            if last_frame == frame_number - 1:
                active_tracks.append((track_id, history, last_detection))

        if len(active_tracks) == 0:
            for detection_index in current_frame.index:
                tracked_detections.loc[detection_index, "track_id"] = next_track_id
                track_history[next_track_id] = [detection_index]
                next_track_id += 1
            continue

        cost_matrix = np.full(
            (len(active_tracks), len(current_indices)),
            invalid_assignment_cost,
            dtype=float,
        )

        link_lookup = {}

        for track_row, (track_id, history, last_detection) in enumerate(active_tracks):

            p1 = tracked_detections.loc[
                last_detection,
                ["x_um", "y_um"],
            ].to_numpy()

            for detection_col, candidate_detection in enumerate(current_indices):

                p2 = current_positions[detection_col]
                distance_um = np.linalg.norm(p2 - p1)

                if distance_um > max_distance_um:
                    continue

                if len(history) < 2:
                    DP = np.nan
                    score = distance_um

                else:
                    previous_detection = history[-2]

                    p0 = tracked_detections.loc[
                        previous_detection,
                        ["x_um", "y_um"],
                    ].to_numpy()

                    DP = calculate_step_alignment(p0, p1, p2)

                    if np.isnan(DP):
                        score = distance_um
                    else:
                        score = distance_um - DP_weight_um * DP

                cost_matrix[track_row, detection_col] = score

                link_lookup[(track_row, detection_col)] = {
                    "track_id": track_id,
                    "detection_a": last_detection,
                    "detection_b": candidate_detection,
                    "frame_a": frame_number - 1,
                    "frame_b": frame_number,
                    "distance_um": distance_um,
                    "DP": DP,
                    "score": score,
                }

        assigned_detections = set()

        if np.any(cost_matrix < invalid_assignment_cost):

            assigned_track_rows, assigned_detection_cols = linear_sum_assignment(
                cost_matrix
            )

            for track_row, detection_col in zip(
                assigned_track_rows,
                assigned_detection_cols,
            ):

                if cost_matrix[track_row, detection_col] >= invalid_assignment_cost:
                    continue

                link = link_lookup[(track_row, detection_col)]

                track_id = link["track_id"]
                detection_b = link["detection_b"]

                tracked_detections.loc[detection_b, "track_id"] = track_id
                track_history[track_id].append(detection_b)

                assigned_detections.add(detection_b)
                all_links.append(link)

        for detection_index in current_frame.index:
            if detection_index not in assigned_detections:
                tracked_detections.loc[detection_index, "track_id"] = next_track_id
                track_history[next_track_id] = [detection_index]
                next_track_id += 1

    links = pd.DataFrame(all_links)

    tracks = tracked_detections.dropna(subset=["track_id"]).copy()
    tracks["track_id"] = tracks["track_id"].astype(int)
    tracks = tracks.sort_values(["track_id", "frame"]).copy()

    return links, tracks