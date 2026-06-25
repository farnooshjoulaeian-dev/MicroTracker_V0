import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trackfar.gap_closing import (
    summarize_tracks_for_jumps,
    find_jump_candidates,
    calculate_jump_DP,
    select_jump_links,
    merge_tracks,
)


tracks = pd.read_csv("outputs/tracks_initial.csv")

track_summary = summarize_tracks_for_jumps(tracks)

jmp_candidates = find_jump_candidates(
    track_summary,
    fps=25,
    max_jmp_frames=4,
    max_jmp_speed_um_s=130,
    min_segment_length=20,
)

jmp_candidates_with_DP = calculate_jump_DP(
    tracks,
    jmp_candidates,
)

selected_jmp_links = select_jump_links(
    jmp_candidates_with_DP,
    max_jmp_frames=4,
    max_jmp_speed_um_s=130,
    min_jmp_DP=0.85,
    min_jmp_DP_B=0.50,
    min_segment_length=20,
)

tracks_merged, track_parent = merge_tracks(
    tracks,
    selected_jmp_links,
)

Path("outputs").mkdir(exist_ok=True)

track_summary.to_csv("outputs/track_summary_for_jumps.csv", index=False)
jmp_candidates.to_csv("outputs/jmp_candidates.csv", index=False)
jmp_candidates_with_DP.to_csv("outputs/jmp_candidates_with_DP.csv", index=False)
selected_jmp_links.to_csv("outputs/selected_jmp_links.csv", index=False)
tracks_merged.to_csv("outputs/tracks_merged.csv", index=False)

#print("Initial tracks:", tracks["track_id"].nunique())
#print("Jump candidates:", len(jmp_candidates))
#print("Jump candidates with DP:", len(jmp_candidates_with_DP))
#print("Selected jump links:", len(selected_jmp_links))
#print("Merged tracks:", tracks_merged["track_id"].nunique())



from trackfar.visualization import plot_selected_jump_links



plot_selected_jump_links(
    tracks,
    selected_jmp_links,
    n_jumps_to_inspect=3,
)

