import sys
from pathlib import Path
import matplotlib.pyplot as plt 



sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from trackfar.load_video import load_video
from trackfar.detection import detect_cells_in_frame
from trackfar.visualization import plot_detection_steps


video_info, frame = load_video("video_address/video.mp4")

features, corrected, thresholded_img, closed_mask, label_image = (
    detect_cells_in_frame(
        frame,
        frame_number=0
    )
)

plot_detection_steps(
    frame,
    corrected,
    thresholded_img,
    closed_mask,
    label_image
)
