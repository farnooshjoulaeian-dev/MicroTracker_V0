import matplotlib.pyplot as plt


def plot_selected_jump_links(
    tracks,
    selected_jmp_links,
    n_jumps_to_inspect=12,
    random_state=0,
):
    """
    Visually inspect proposed broken-track joins before merging.
    """

    if len(selected_jmp_links) == 0:
        print("No selected jump links to inspect.")
        return

    n = min(n_jumps_to_inspect, len(selected_jmp_links))

    jump_links_to_plot = selected_jmp_links.sample(
        n=n,
        random_state=random_state
    )

    for _, row in jump_links_to_plot.iterrows():

        track_a = tracks[
            tracks["track_id"] == row["track_a"]
        ].sort_values("frame")

        track_b = tracks[
            tracks["track_id"] == row["track_b"]
        ].sort_values("frame")

        plt.figure(figsize=(5, 5))

        plt.plot(
            track_a["x_um"],
            track_a["y_um"],
            "-o",
            markersize=3,
            linewidth=1,
            label=f"A {int(row['track_a'])}",
        )

        plt.plot(
            track_b["x_um"],
            track_b["y_um"],
            "-o",
            markersize=3,
            linewidth=1,
            label=f"B {int(row['track_b'])}",
        )

        plt.plot(
            [track_a["x_um"].iloc[-1], track_b["x_um"].iloc[0]],
            [track_a["y_um"].iloc[-1], track_b["y_um"].iloc[0]],
            "k--",
            linewidth=1,
            label="proposed jump",
        )

        plt.xlabel("x position (um)")
        plt.ylabel("y position (um)")

        plt.title(
            f"Proposed merge: A {int(row['track_a'])} → B {int(row['track_b'])}\n"
            f"gap={int(row['jmp_frames'])} frames, "
            f"speed={row['speed_um_s']:.1f} um/s, "
            f"DP_A={row['jmp_DP_A']:.2f}, "
            f"DP_B={row['jmp_DP_B']:.2f}"
        )

        plt.gca().set_aspect("equal")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.show()