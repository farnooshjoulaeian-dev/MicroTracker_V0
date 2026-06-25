

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trackfar.load_video import load_video

frames, metadata = load_video("video_address/video.mp4")

print(metadata)
print(len(frames))
print(frames[0].shape)
