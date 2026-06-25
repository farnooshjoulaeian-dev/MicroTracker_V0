
import cv2 

def load_video(path):
    # Open the video file
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    # Read basic video metadata
    video_info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    # Read the first frame for testing
    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise ValueError("Could not read the first frame.")

    return video_info, frame

