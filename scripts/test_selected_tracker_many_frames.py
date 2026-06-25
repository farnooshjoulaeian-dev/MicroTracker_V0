import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import matplotlib.pyplot as plt

from trackfar.detection import detect_cells_in_frame
from trackfar.selected_tracker import track_many_frames


video_path = "video_address/video.mp4"

cap = cv2.VideoCapture(video_path)

ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Could not read frame 0")

detections0, *_ = detect_cells_in_frame(
    frame0,
    frame_number=0,
)

cell0 = detections0.iloc[0]

start_position_px = (
    cell0["x_px"],
    cell0["y_px"],
)

traj, stop_reason, stop_frame = track_many_frames(
    cap=cap,
    start_frame=0,
    start_position_px=start_position_px,
    n_frames=100,
    max_distance_px=20,
)

cap.release()

trajectory_df = traj.to_dataframe()

Path("outputs").mkdir(exist_ok=True)
trajectory_df.to_csv("outputs/selected_trajectory.csv", index=False)

print("Stop reason:", stop_reason)
print("Stop frame:", stop_frame)
print("Trajectory length:", len(trajectory_df))

plt.figure(figsize=(7, 7))
plt.imshow(frame0)
plt.plot(
    trajectory_df["x_px"],
    trajectory_df["y_px"],
    "-o",
    markersize=2,
)
plt.scatter(
    trajectory_df["x_px"].iloc[0],
    trajectory_df["y_px"].iloc[0],
    s=80,
    label="start",
)
plt.scatter(
    trajectory_df["x_px"].iloc[-1],
    trajectory_df["y_px"].iloc[-1],
    s=80,
    label="end",
)

plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")
plt.xlabel("x (px)")
plt.ylabel("y (px)")
plt.title("Selected trajectory")
plt.legend()
plt.tight_layout()
plt.show()
