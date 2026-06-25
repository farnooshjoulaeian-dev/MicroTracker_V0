import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trackfar.linking import link_detections


detections = pd.read_csv("outputs/detections.csv")

links, tracks = link_detections(
    detections,
    fps=25,
    max_distance_um=5.6,
    DP_weight_um=0.25,
)

Path("outputs").mkdir(exist_ok=True)

links.to_csv("outputs/links.csv", index=False)
tracks.to_csv("outputs/tracks_initial.csv", index=False)

print("Number of detections:", len(detections))
print("Number of links:", len(links))
print("Number of tracks:", tracks["track_id"].nunique())

print(tracks.head())