import sys
import numpy as np
import cv2
from pathlib import Path

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from trackfar.detection import detect_cells_in_frame
from trackfar.selected_tracker import find_nearest_detection, track_next_frame


class VideoLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        self.clicked.emit(event.pos().x(), event.pos().y())


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("MicroTracker")
        self.setGeometry(100, 100, 900, 650)
         # Video
        self.cap = None
        self.video_path = None
        self.current_frame = 0
        self.n_frames = 0
        # Tracking
        self.selected_cell = None
        self.trajectory_points = []
        # Playback
        self.is_playing = False
        self.tracking_active = False
    
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.object_polarity = "bright" #darkfield, flourecent
        #self.object_polarity = "dark" #brightfield


        # ROI
       # self.use_roi = True
        #self.roi_x = 500
        #self.roi_y = 200
        #self.roi_w = 600
        #self.roi_h = 400


        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top buttons
        top_bar = QHBoxLayout()

        load_button = QPushButton("Load video")
        select_button = QPushButton("Select cell")
        track_button = QPushButton("Track")
        save_button = QPushButton("Save")

        top_bar.addWidget(load_button)
        top_bar.addWidget(select_button)
        top_bar.addWidget(track_button)
        top_bar.addWidget(save_button)

        main_layout.addLayout(top_bar)

        # Middle area
        middle_layout = QHBoxLayout()

        param_panel = QVBoxLayout()

        param_panel.addWidget(QLabel("Search radius"))
        self.search_radius_box = QSpinBox()
        self.search_radius_box.setValue(20)
        param_panel.addWidget(self.search_radius_box)

        param_panel.addWidget(QLabel("Min area"))
        self.min_area_box = QSpinBox()
        self.min_area_box.setValue(6)
        param_panel.addWidget(self.min_area_box)

        param_panel.addStretch()

        self.video_label = VideoLabel("Video display area")
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        self.video_label.setAlignment(Qt.AlignCenter)

        middle_layout.addLayout(param_panel)
        middle_layout.addWidget(self.video_label)

        main_layout.addLayout(middle_layout)

        # Frame slider
        frame_bar = QHBoxLayout()

        self.frame_label = QLabel("Frame: 0")

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(0)

        frame_bar.addWidget(self.frame_label)
        frame_bar.addWidget(self.frame_slider)

        main_layout.addLayout(frame_bar)

        # Connections
        load_button.clicked.connect(self.load_video)
        track_button.clicked.connect(self.start_tracking)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.video_label.clicked.connect(self.video_clicked)
        save_button.clicked.connect(self.save_trajectory)

        # Shortcuts
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("Right"), self, activated=self.next_frame)
        QShortcut(QKeySequence("Left"), self, activated=self.previous_frame)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.avi *.mov);;All files (*)"
        )

        if video_path == "":
            return

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        self.selected_cell = None
        self.trajectory_points = []
        self.tracking_active = False
        self.is_playing = False
        self.timer.stop()

        print("Loaded:", video_path)
        print("Number of frames:", self.n_frames)

        self.frame_slider.setMaximum(self.n_frames - 1)
        self.frame_slider.setValue(0)
        self.show_frame(0)
        


    def show_frame(self, frame_number):

        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        if not ret:
            return

        # Store original frame
        self.current_frame_bgr = frame
        self.frame_height, self.frame_width = frame.shape[:2]


        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_frame_gray = frame_gray

        display_frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)

        # Draw trajectory
        if len(self.trajectory_points) > 1:
            points = [
                (int(p["x_px"]), int(p["y_px"]))
                for p in self.trajectory_points
                if p["frame"] <= frame_number
            ]

            for p1, p2 in zip(points[:-1], points[1:]):
                cv2.line(display_frame, p1, p2, (0, 255, 0), 2)

        # Draw selected cell
        if self.selected_cell is not None:
            if int(self.selected_cell["frame"]) == frame_number:
                x = int(self.selected_cell["x_px"])
                y = int(self.selected_cell["y_px"])

                cv2.circle(display_frame, (x, y), 14, (255, 0, 0), 2)

        h, w, ch = display_frame.shape
        bytes_per_line = ch * w

        qimage = QImage(
            display_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimage)

        if pixmap.isNull():
            print("Pixmap is null")
            return

        pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio
        )

        self.video_label.setPixmap(pixmap)
        self.current_frame = frame_number
        self.frame_label.setText(f"Frame: {frame_number}")

    def slider_changed(self, value):
        if self.cap is None:
            return

        self.show_frame(value)

    def toggle_play(self):
        if self.cap is None:
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.timer.start(40)
        else:
            self.timer.stop()

    def next_frame(self):
        if self.cap is None:
            return

        next_frame = min(self.current_frame + 1, self.n_frames - 1)

        if next_frame == self.current_frame:
            self.timer.stop()
            self.is_playing = False
            self.tracking_active = False
            return

        self.frame_slider.setValue(next_frame)

        if self.tracking_active:
            self.track_current_frame()

    def previous_frame(self):
        if self.cap is None:
            return

        previous_frame = max(self.current_frame - 1, 0)
        self.frame_slider.setValue(previous_frame)

    def video_clicked(self, x, y):
        image_pos = self.label_to_image_coordinates(x, y)

        if image_pos is None:
            print("Clicked outside image.")
            return

        detections, *_ = detect_cells_in_frame(
            self.current_frame_bgr,
            frame_number=self.current_frame,
            min_area_px=self.min_area_box.value(),
            
        )

        selected = find_nearest_detection(
            detections=detections,
            previous_position_px=image_pos,
            max_distance_px=self.search_radius_box.value(),
        )

        if selected is None:
            print("No cell near click.")
            return
        self.selected_cell = selected
        manual_point = {
            "frame": self.current_frame,
            "x_px": float(selected["x_px"]),
            "y_px": float(selected["y_px"]),
            "source": "manual",
        }

        if len(self.trajectory_points) == 0:
            self.trajectory_points = [manual_point]
        else:
            self.trajectory_points.append(manual_point)


        print(
            "Selected cell:",
            f"x={selected['x_px']:.1f}",
            f"y={selected['y_px']:.1f}",
            f"frame={int(selected['frame'])}",
        )

        self.show_frame(self.current_frame)

    def label_to_image_coordinates(self, x_label, y_label):
        if self.video_label.pixmap() is None:
            return None

        pixmap = self.video_label.pixmap()

        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()

        label_w = self.video_label.width()
        label_h = self.video_label.height()

        x_offset = (label_w - pixmap_w) / 2
        y_offset = (label_h - pixmap_h) / 2

        x_display = x_label - x_offset
        y_display = y_label - y_offset

        if x_display < 0 or y_display < 0:
            return None

        if x_display > pixmap_w or y_display > pixmap_h:
            return None

        x_image = x_display * self.frame_width / pixmap_w
        y_image = y_display * self.frame_height / pixmap_h

        return int(x_image), int(y_image)

    def start_tracking(self):
        if self.cap is None:
            print("No video loaded.")
            return

        if self.selected_cell is None:
            print("No cell selected.")
            return

        self.tracking_active = True
        self.is_playing = True
        self.timer.start(40)

        print("Live tracking started.")

    def track_current_frame(self):
        if len(self.trajectory_points) == 0:
            return

        previous = self.trajectory_points[-1]

        previous_position_px = (
            previous["x_px"],
            previous["y_px"],
        )

        selected, detections = track_next_frame(
            frame=self.current_frame_bgr,
            frame_number=self.current_frame,
            previous_position_px=previous_position_px,
            max_distance_px=self.search_radius_box.value(),
        )

        if selected is None:

            print("Tracking paused at frame:", self.current_frame)
            self.timer.stop()
            self.is_playing = False
            self.tracking_active = False
            return

        self.selected_cell = selected

        self.trajectory_points.append({
            "frame": self.current_frame,
            "x_px": float(selected["x_px"]),
            "y_px": float(selected["y_px"]),
            "source": "auto",
        })

        self.show_frame(self.current_frame)

    def save_trajectory(self):

        if len(self.trajectory_points) == 0:
            print("No traj to save")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save traj",
            "traj.npy",
            "Numpy files(*.npy);;All files (*)"
        )
        if save_path == "":
            return
        
        np.save(save_path,self.trajectory_points)
        
        print("Traj saved", save_path)







if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())