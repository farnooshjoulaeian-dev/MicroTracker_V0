import math
import unittest

import numpy as np

from trackfar.linking import calculate_step_alignment
from trackfar.trajectory import Trajectory


class TrajectoryTests(unittest.TestCase):
    def test_replace_point_keeps_frames_sorted(self):
        trajectory = Trajectory()
        trajectory.add_point(frame=2, x_px=20, y_px=30)
        trajectory.add_point(frame=0, x_px=1, y_px=2)

        trajectory.replace_point(frame=2, x_px=25, y_px=35)

        table = trajectory.to_dataframe()

        self.assertEqual(table["frame"].tolist(), [0, 2])
        self.assertEqual(table["x_px"].tolist(), [1, 25])
        self.assertEqual(table["source"].tolist(), ["manual", "manual_correction"])


class LinkingTests(unittest.TestCase):
    def test_step_alignment_returns_expected_direction_scores(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])

        aligned = calculate_step_alignment(p0, p1, np.array([2.0, 0.0]))
        reversed_direction = calculate_step_alignment(p0, p1, np.array([0.0, 0.0]))
        zero_length = calculate_step_alignment(p0, p0, p1)

        self.assertAlmostEqual(aligned, 1.0)
        self.assertAlmostEqual(reversed_direction, -1.0)
        self.assertTrue(math.isnan(zero_length))


if __name__ == "__main__":
    unittest.main()
