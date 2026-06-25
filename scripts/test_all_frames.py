import sys
from pathlib import Path
import cv2
import time
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trackfar.load_video import load_video
from trackfar.detection import detect_cells_in_frame


video_path = "video_address/video.mp4"

video_info, first_frame = load_video(video_path)
total_frames = video_info["frame_count"]

all_features = []

cap = cv2.VideoCapture(video_path)
frame_number = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    features, corrected, thresholded_img, closed_mask, label_image = detect_cells_in_frame(
        frame,
        frame_number=frame_number
    )

    all_features.append(features)
    frame_number += 1

    if frame_number % 10 == 0 or frame_number == total_frames:
        progress = 100 * frame_number / total_frames
        elapsed = time.time() - start_time
        print(f"Processing: {progress:.1f}% | frame {frame_number}/{total_frames} | elapsed {elapsed:.1f} s")

cap.release()

detections = pd.concat(all_features, ignore_index=True)

output_path = Path("outputs/detections.csv")
output_path.parent.mkdir(exist_ok=True)
detections.to_csv(output_path, index=False)

total_time = time.time() - start_time

print(f"Finished processing in {total_time:.1f} s")
print(f"Number of detections: {len(detections)}")
print(f"Saved to: {output_path}")
