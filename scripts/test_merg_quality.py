import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trackfar.diagnostics import compare_track_lengths


tracks_before = pd.read_csv("outputs/tracks_initial.csv")
tracks_after = pd.read_csv("outputs/tracks_merged.csv")

summary, lengths_before, lengths_after = compare_track_lengths(
    tracks_before,
    tracks_after,
)

print(summary)

Path("outputs").mkdir(exist_ok=True)
pd.Series(summary).to_csv("outputs/merge_quality_summary.csv")