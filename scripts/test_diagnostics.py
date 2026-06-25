import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trackfar.diagnostics import calculate_track_velocities


tracks = pd.read_csv("outputs/tracks_initial.csv")

tracks_velocity_check, valid_steps = calculate_track_velocities(
    tracks,
    fps=25
)

Path("outputs").mkdir(exist_ok=True)

tracks_velocity_check.to_csv("outputs/tracks_velocity_check.csv", index=False)
valid_steps.to_csv("outputs/valid_steps.csv", index=False)

print(valid_steps.head())
print("Number of valid steps:", len(valid_steps))



from trackfar.diagnostics import (
    calculate_track_velocities,
    summarize_velocities,
)

velocity_summary, velocity_percentiles, distance_percentiles = (
    summarize_velocities(valid_steps, fps=25)
)

print("Instantaneous velocity summary:")
print(velocity_summary)

print("\nVelocity percentiles:")
print(velocity_percentiles)

print("\nEquivalent displacement per frame:")
print(distance_percentiles)