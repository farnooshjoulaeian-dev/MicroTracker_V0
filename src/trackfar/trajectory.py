import pandas as pd


class Trajectory:
    def __init__(self):
        self.points = []

    def add_point(self, frame, x_px, y_px, source="manual"):
        self.points.append({
            "frame": frame,
            "x_px": x_px,
            "y_px": y_px,
            "source": source,
        })

    def replace_point(self, frame, x_px, y_px, source="manual_correction"):
        self.points = [
            p for p in self.points
            if p["frame"] != frame
        ]

        self.add_point(
            frame=frame,
            x_px=x_px,
            y_px=y_px,
            source=source,
        )

        self.points = sorted(
            self.points,
            key=lambda p: p["frame"]
        )

    def to_dataframe(self):
        return pd.DataFrame(self.points)

    def save_csv(self, path):
        self.to_dataframe().to_csv(path, index=False)