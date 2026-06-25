import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from trackfar.detection import detect_cells_in_frame
from trackfar.selected_tracker import track_next_frame


video_path = "video_address/video.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Could not read frame 0")

features0, *_ = detect_cells_in_frame(
    frame0,
    frame_number=0
)

print("Number of detections in frame 0:", len(features0))

# Use the first detected cell as a fake manual click
cell0 = features0.iloc[0]
previous_position_px = (
    cell0["x_px"],
    cell0["y_px"]
)

print("Starting position:", previous_position_px)

ret, frame1 = cap.read()
if not ret:
    raise RuntimeError("Could not read frame 1")

selected_detection, detections1 = track_next_frame(
    frame=frame1,
    frame_number=1,
    previous_position_px=previous_position_px,
    max_distance_px=20,
)

cap.release()

print("Number of detections in frame 1:", len(detections1))

if selected_detection is None:
    print("No nearby cell found.")
else:
    print("Selected detection:")
    print(selected_detection[["frame", "x_px", "y_px", "distance_from_previous_px"]])
