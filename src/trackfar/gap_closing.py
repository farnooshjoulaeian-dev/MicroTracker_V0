import numpy as np
import pandas as pd

from trackfar.linking import calculate_step_alignment


def summarize_tracks_for_jumps(tracks):
    track_summary = []

    for track_id, track in tracks.groupby("track_id"):
        track = track.sort_values("frame")

        track_summary.append({
            "track_id": track_id,
            "start_frame": track["frame"].iloc[0],
            "end_frame": track["frame"].iloc[-1],
            "start_x_um": track["x_um"].iloc[0],
            "start_y_um": track["y_um"].iloc[0],
            "end_x_um": track["x_um"].iloc[-1],
            "end_y_um": track["y_um"].iloc[-1],
            "length": len(track),
        })

    return pd.DataFrame(track_summary)


def find_jump_candidates(
    track_summary,
    fps=25,
    max_jmp_frames=4,
    max_jmp_speed_um_s=130,
    min_segment_length=20,
):
    jmp_candidates = []

    track_summary = track_summary[
        track_summary["length"] >= min_segment_length
    ].copy()

    for _, end_track in track_summary.iterrows():

        possible_starts = track_summary[
            (track_summary["start_frame"] > end_track["end_frame"])
            & (
                track_summary["start_frame"]
                <= end_track["end_frame"] + max_jmp_frames
            )
        ]

        for _, start_track in possible_starts.iterrows():

            if end_track["track_id"] == start_track["track_id"]:
                continue

            jmp_frames = start_track["start_frame"] - end_track["end_frame"]

            p_end = np.array([
                end_track["end_x_um"],
                end_track["end_y_um"],
            ])

            p_start = np.array([
                start_track["start_x_um"],
                start_track["start_y_um"],
            ])

            distance_um = np.linalg.norm(p_start - p_end)
            allowed_distance_um = max_jmp_speed_um_s * jmp_frames / fps

            if distance_um > allowed_distance_um:
                continue

            speed_um_s = distance_um / jmp_frames * fps

            jmp_candidates.append({
                "track_a": end_track["track_id"],
                "track_b": start_track["track_id"],
                "jmp_frames": jmp_frames,
                "distance_um": distance_um,
                "allowed_distance_um": allowed_distance_um,
                "speed_um_s": speed_um_s,
                "length_a": end_track["length"],
                "length_b": start_track["length"],
            })

    return pd.DataFrame(jmp_candidates)


def calculate_jump_DP(tracks, jmp_candidates):
    jmp_candidates_with_DP = []

    for _, candidate in jmp_candidates.iterrows():

        track_a = tracks[
            tracks["track_id"] == candidate["track_a"]
        ].sort_values("frame")

        track_b = tracks[
            tracks["track_id"] == candidate["track_b"]
        ].sort_values("frame")

        if len(track_a) < 2 or len(track_b) < 2:
            continue

        p0 = track_a[["x_um", "y_um"]].iloc[-2].to_numpy()
        p1 = track_a[["x_um", "y_um"]].iloc[-1].to_numpy()

        p2 = track_b[["x_um", "y_um"]].iloc[0].to_numpy()
        p3 = track_b[["x_um", "y_um"]].iloc[1].to_numpy()

        DP_A = calculate_step_alignment(p0, p1, p2)
        DP_B = calculate_step_alignment(p1, p2, p3)

        row = candidate.to_dict()
        row["jmp_DP_A"] = DP_A
        row["jmp_DP_B"] = DP_B
        row["jmp_DP"] = DP_A

        jmp_candidates_with_DP.append(row)

    return pd.DataFrame(jmp_candidates_with_DP)


def select_jump_links(
    jmp_candidates_with_DP,
    max_jmp_frames=4,
    max_jmp_speed_um_s=130,
    min_jmp_DP=0.85,
    min_jmp_DP_B=0.50,
    min_segment_length=20,
):
    strong_jmp_candidates = jmp_candidates_with_DP[
        (jmp_candidates_with_DP["jmp_frames"] <= max_jmp_frames)
        & (jmp_candidates_with_DP["speed_um_s"] <= max_jmp_speed_um_s)
        & (jmp_candidates_with_DP["jmp_DP_A"] >= min_jmp_DP)
        & (jmp_candidates_with_DP["jmp_DP_B"] >= min_jmp_DP_B)
        & (jmp_candidates_with_DP["length_a"] >= min_segment_length)
        & (jmp_candidates_with_DP["length_b"] >= min_segment_length)
    ].copy()

    strong_jmp_candidates = strong_jmp_candidates.sort_values(
        ["jmp_frames", "speed_um_s", "jmp_DP_A", "jmp_DP_B"],
        ascending=[True, True, False, False],
    )

    selected_jmp_links = []
    used_end_tracks = set()
    used_start_tracks = set()

    for _, row in strong_jmp_candidates.iterrows():
        track_a = int(row["track_a"])
        track_b = int(row["track_b"])

        if track_a in used_end_tracks:
            continue

        if track_b in used_start_tracks:
            continue

        selected_jmp_links.append(row)
        used_end_tracks.add(track_a)
        used_start_tracks.add(track_b)

    return pd.DataFrame(selected_jmp_links)


def merge_tracks(tracks, selected_jmp_links):
    track_parent = {
        int(track_id): int(track_id)
        for track_id in tracks["track_id"].unique()
    }

    def find_parent(track_id):
        while track_parent[track_id] != track_id:
            track_id = track_parent[track_id]

        return track_id

    for _, row in selected_jmp_links.iterrows():

        track_a = int(row["track_a"])
        track_b = int(row["track_b"])

        parent_a = find_parent(track_a)
        parent_b = find_parent(track_b)

        if parent_a == parent_b:
            continue

        track_parent[parent_b] = parent_a

    tracks_merged = tracks.copy()
    tracks_merged["original_track_id"] = tracks_merged["track_id"]

    tracks_merged["track_id"] = tracks_merged["track_id"].apply(
        lambda track_id: find_parent(int(track_id))
    )

    tracks_merged = tracks_merged.sort_values(
        ["track_id", "frame"]
    ).copy()

    return tracks_merged, track_parent