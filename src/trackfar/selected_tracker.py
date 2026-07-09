import numpy as np

from trackfar.detection import detect_cells_in_frame


def find_nearest_detection(
    detections,
    previous_position_px,
    max_distance_px=20,
):
    """
    Find the nearest detected cell to a previous manually or automatically
    selected position.

    Parameters
    ----------
    detections : pandas.DataFrame
        Detection table from one frame.

    previous_position_px : tuple
        Previous cell position as (x_px, y_px).

    max_distance_px : float
        Maximum allowed distance in pixels.

    Returns
    -------
    selected_detection : pandas.Series or None
        Nearest valid detection, or None if no detection is close enough.
    """

    if len(detections) == 0:
        return None

    x_prev, y_prev = previous_position_px

    dx = detections["x_px"] - x_prev
    dy = detections["y_px"] - y_prev

    distances = np.sqrt(dx**2 + dy**2)

    nearest_index = distances.idxmin()
    nearest_distance = distances.loc[nearest_index]

    if nearest_distance > max_distance_px:
        return None

    selected_detection = detections.loc[nearest_index].copy()
    selected_detection["distance_from_previous_px"] = nearest_distance

    return selected_detection


def track_next_frame(
    frame,
    frame_number,
    previous_position_px,
    max_distance_px=20,
    pixel_size=1.178,
):
    x_prev, y_prev = previous_position_px
    r = max_distance_px

    x1 = max(0, int(x_prev - r))
    x2 = min(frame.shape[1], int(x_prev + r))
    y1 = max(0, int(y_prev - r))
    y2 = min(frame.shape[0], int(y_prev + r))

    roi = frame[y1:y2, x1:x2]

    detections, corrected, thresholded_img, closed_mask, label_image = detect_cells_in_frame(
        roi,
        frame_number=frame_number,
        pixel_size=pixel_size,
    )

    if len(detections) == 0:
        return None, detections

    detections["x_px"] += x1
    detections["y_px"] += y1

    selected = find_nearest_detection(
        detections,
        previous_position_px,
        max_distance_px=max_distance_px,
    )

    return selected, detections







from trackfar.trajectory import Trajectory


def track_many_frames(
    cap,
    start_frame,
    start_position_px,
    n_frames=100,
    max_distance_px=20,
    pixel_size=1.178,
):
    traj = Trajectory()

    traj.add_point(
        frame=start_frame,
        x_px=start_position_px[0],
        y_px=start_position_px[1],
        source="manual",
    )

    previous_position_px = start_position_px
    stop_reason = "finished"
    stop_frame = None

    for frame_number in range(start_frame + 1, start_frame + n_frames):

        ret, frame = cap.read()

        if not ret:
            stop_reason = "end_of_video"
            stop_frame = frame_number
            break

        selected_detection, detections = track_next_frame(
            frame=frame,
            frame_number=frame_number,
            previous_position_px=previous_position_px,
            max_distance_px=max_distance_px,
            pixel_size=pixel_size,
        )

        if selected_detection is None:
            stop_reason = "lost_track"
            stop_frame = frame_number
            break

        x_new = selected_detection["x_px"]
        y_new = selected_detection["y_px"]

        traj.add_point(
            frame=frame_number,
            x_px=x_new,
            y_px=y_new,
            source="auto",
        )

        previous_position_px = (x_new, y_new)

    return traj, stop_reason, stop_frame