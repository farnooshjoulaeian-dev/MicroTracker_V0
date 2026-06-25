import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trackfar.trajectory import Trajectory


traj = Trajectory()

traj.add_point(frame=0, x_px=100, y_px=200, source="manual")
traj.add_point(frame=1, x_px=105, y_px=203, source="auto")
traj.replace_point(frame=1, x_px=107, y_px=204)

print(traj.to_dataframe())

Path("outputs").mkdir(exist_ok=True)
traj.save_csv("outputs/test_trajectory.csv")