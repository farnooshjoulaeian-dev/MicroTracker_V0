# Optional manual check:

track_id_a = 54
track_id_b = 1561

# Select tracks and sort by time
track_a = tracks_merged[
    tracks_merged["track_id"] == track_id_a
].sort_values("frame")

track_b = tracks_merged[
    tracks_merged["track_id"] == track_id_b
].sort_values("frame")

print("A frames:", track_a["frame"].min(), "to", track_a["frame"].max())
print("B frames:", track_b["frame"].min(), "to", track_b["frame"].max())

# Check whether the two tracks exist in the same frames
overlap_frames = sorted(
    set(track_a["frame"]).intersection(set(track_b["frame"]))
)

print("Number of overlapping frames:", len(overlap_frames))

if len(overlap_frames) > 0:
    # If tracks overlap, measure how far apart they are during overlap
    overlap_distances = []

    for frame in overlap_frames:
        point_a = track_a[
            track_a["frame"] == frame
        ][["x_um", "y_um"]].iloc[0].to_numpy()

        point_b = track_b[
            track_b["frame"] == frame
        ][["x_um", "y_um"]].iloc[0].to_numpy()

        overlap_distances.append({
            "frame": frame,
            "distance_um": np.linalg.norm(point_a - point_b)
        })

    overlap_distances = pd.DataFrame(overlap_distances)

    print("Overlap distance summary:")
    print(overlap_distances["distance_um"].describe())

else:
    # If tracks do not overlap, test whether A can jump to B
    jmp_frames = track_b["frame"].iloc[0] - track_a["frame"].iloc[-1]

    print("A -> B jmp frames:", jmp_frames)

    if jmp_frames <= 0:
        print("A does not end before B starts. Try reversing track_id_a and track_id_b.")

    else:
        p_a_end = track_a[["x_um", "y_um"]].iloc[-1].to_numpy()
        p_b_start = track_b[["x_um", "y_um"]].iloc[0].to_numpy()

        jmp_distance_um = np.linalg.norm(p_b_start - p_a_end)
        jmp_speed_um_s = jmp_distance_um / jmp_frames * fps

        print("A -> B jmp distance um:", jmp_distance_um)
        print("A -> B jmp speed um/s:", jmp_speed_um_s)

        # DP needs two points at the end of A
        if len(track_a) >= 2:
            p0 = track_a[["x_um", "y_um"]].iloc[-2].to_numpy()
            p1 = track_a[["x_um", "y_um"]].iloc[-1].to_numpy()
            p2 = track_b[["x_um", "y_um"]].iloc[0].to_numpy()

            jmp_DP = calculate_step_alignment(p0, p1, p2)

            print("A -> B jmp DP:", jmp_DP)
        else:
            print("Track A is too short to calculate jmp DP.")


            


            # Check whether original track IDs were preserved during merging

print("tracks columns:")
print(tracks.columns.tolist())

print("tracks_merged columns:")
print(tracks_merged.columns.tolist())

if "original_track_id" in tracks_merged.columns:
    print("Good: original_track_id exists.")
    print("Manual inspection can use original IDs.")

    print("Number of original track IDs:")
    print(tracks_merged["original_track_id"].nunique())

    print("Number of merged track IDs:")
    print(tracks_merged["track_id"].nunique())

else:
    print("original_track_id is missing.")
    print("Renumbering may have overwritten the old IDs.")






    manual_track_parent = {
    int(track_id): int(track_id)
    for track_id in tracks_merged["track_id"].unique()
}

def manual_find_parent(track_id):
    while manual_track_parent[track_id] != track_id:
        track_id = manual_track_parent[track_id]
    return track_id

for _, row in manual_corrections.iterrows():
    if row["action"] == "merge":
        track_a = int(row["track_a"])
        track_b = int(row["track_b"])

        parent_a = manual_find_parent(track_a)
        parent_b = manual_find_parent(track_b)

        manual_track_parent[parent_b] = parent_a

tracks_corrected = tracks_merged.copy()
tracks_corrected["track_id_before_manual"] = tracks_corrected["track_id"]
tracks_corrected["track_id"] = tracks_corrected["track_id"].apply(
    lambda x: manual_find_parent(int(x))
)




# Check manual correction result

print("Tracks before manual correction:", tracks_merged["track_id"].nunique())
print("Tracks after manual correction:", tracks_corrected["track_id"].nunique())

print("Rows changed by manual correction:")
print(
    (tracks_corrected["track_id"] != tracks_corrected["track_id_before_manual"]).sum()
)