import numpy as np


def calculate_track_velocities(
    tracks,
    fps=25,
):
    """
    Calculate frame-to-frame displacements and instantaneous
    velocities for diagnostic purposes.

    Parameters
    ----------
    tracks : pandas.DataFrame
        Initial track table.

    fps : float
        Video frame rate.

    Returns
    -------
    tracks_velocity_check : pandas.DataFrame
        Copy of tracks with additional velocity columns.

    valid_steps : pandas.DataFrame
        Only normal frame-to-frame steps (dt_frame = 1).
    """

    # Make a diagnostic copy.
    tracks_velocity_check = tracks.copy()

    # Sort by track and time.
    tracks_velocity_check = (
        tracks_velocity_check
        .sort_values(["track_id", "frame"])
        .copy()
    )

    # Consecutive displacements.
    tracks_velocity_check["dx_um"] = (
        tracks_velocity_check
        .groupby("track_id")["x_um"]
        .diff()
    )

    tracks_velocity_check["dy_um"] = (
        tracks_velocity_check
        .groupby("track_id")["y_um"]
        .diff()
    )

    # Frame difference.
    tracks_velocity_check["dt_frame"] = (
        tracks_velocity_check
        .groupby("track_id")["frame"]
        .diff()
    )

    # Step distance.
    tracks_velocity_check["step_distance_um"] = np.sqrt(
        tracks_velocity_check["dx_um"]**2
        + tracks_velocity_check["dy_um"]**2
    )

    # Convert to seconds.
    tracks_velocity_check["dt_s"] = (
        tracks_velocity_check["dt_frame"] / fps
    )

    # Instantaneous velocity.
    tracks_velocity_check["instant_velocity_um_s"] = (
        tracks_velocity_check["step_distance_um"]
        / tracks_velocity_check["dt_s"]
    )

    # Keep only normal frame-to-frame links.
    valid_steps = tracks_velocity_check[
        (tracks_velocity_check["dt_frame"] == 1)
        & (
            tracks_velocity_check[
                "instant_velocity_um_s"
            ].notna()
        )
    ].copy()

    return tracks_velocity_check, valid_steps









def summarize_velocities(valid_steps, fps=25):
    velocity_summary = valid_steps["instant_velocity_um_s"].describe()

    velocity_percentiles = valid_steps["instant_velocity_um_s"].quantile(
        [0.5, 0.75, 0.9, 0.95, 0.99]
    )

    distance_percentiles = velocity_percentiles / fps

    return velocity_summary, velocity_percentiles, distance_percentiles



def compare_track_lengths(tracks_before, tracks_after):
    lengths_before = tracks_before.groupby("track_id").size()
    lengths_after = tracks_after.groupby("track_id").size()

    summary = {
        "n_tracks_before": tracks_before["track_id"].nunique(),
        "n_tracks_after": tracks_after["track_id"].nunique(),
        "max_length_before": lengths_before.max(),
        "max_length_after": lengths_after.max(),
        "mean_length_before": lengths_before.mean(),
        "mean_length_after": lengths_after.mean(),
        "tracks_over_100_before": (lengths_before >= 100).sum(),
        "tracks_over_100_after": (lengths_after >= 100).sum(),
        "tracks_over_200_before": (lengths_before >= 200).sum(),
        "tracks_over_200_after": (lengths_after >= 200).sum(),
    }

    return summary, lengths_before, lengths_after