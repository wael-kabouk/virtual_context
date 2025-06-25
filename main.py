import sys
import os
import cvzone
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QFrame, QGridLayout, QCheckBox)
from PyQt5.QtCore import QTimer, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import QWidget, QLineEdit, QPushButton
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt
import math

main_result_width = 1500
main_source_width = main_result_width // 3
main_height = 300


class CropRectangle(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_pos = None
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.rect = QRect(0, 0, main_source_width, main_height)  # Initial size for the rectangle
        self.dragging = False
        self.resizing = False
        self.resize_handle_size = 10
        self.resize_direction = None
        self.center_crop_rectangle()

    def center_crop_rectangle(self):
        """Centers the crop rectangle within its parent widget."""
        if self.parent():
            parent_rect = self.parent().rect()
            x = (parent_rect.width() - self.rect.width()) // 2
            y = (parent_rect.height() - self.rect.height()) // 2
            self.rect.moveTopLeft(QPoint(x, y))
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.rect.adjusted(0, 0, -self.resize_handle_size, -self.resize_handle_size).contains(event.pos()):
                self.dragging = True
            else:
                self.resize_direction = self.get_resize_direction(event.pos())
                if self.resize_direction:
                    self.resizing = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.last_pos
            new_rect = self.rect.translated(delta)
            if self.parent():
                parent_rect = self.parent().rect()
                new_rect = self.check_boundary_collision(new_rect, parent_rect)
            self.rect = new_rect
            self.last_pos = event.pos()
            self.update()
        elif self.resizing:
            self.resize_rectangle(event.pos())
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.resizing = False
            self.resize_direction = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 255), 2))
        painter.drawRect(self.rect)
        painter.setBrush(QColor(0, 0, 255))
        self.draw_resize_handles(painter)

    def check_boundary_collision(self, rect, bounds):
        """Keeps the rectangle within the given bounds."""
        if rect.left() < bounds.left():
            rect.moveLeft(bounds.left())
        if rect.right() > bounds.right():
            rect.moveRight(bounds.right())
        if rect.top() < bounds.top():
            rect.moveTop(bounds.top())
        if rect.bottom() > bounds.bottom():
            rect.moveBottom(bounds.bottom())
        return rect

    def draw_resize_handles(self, painter):
        """Draws the resize handles at the corners."""
        handle_size = self.resize_handle_size
        for corner in [self.rect.topLeft(), self.rect.topRight(),
                       self.rect.bottomLeft(), self.rect.bottomRight()]:
            painter.drawRect(corner.x() - handle_size // 2,
                             corner.y() - handle_size // 2,
                             handle_size, handle_size)

    def get_resize_direction(self, pos):
        """Determines which corner is being resized based on the mouse position."""
        handle_size = self.resize_handle_size
        corners = {
            'top_left': self.rect.topLeft(),
            'top_right': self.rect.topRight(),
            'bottom_left': self.rect.bottomLeft(),
            'bottom_right': self.rect.bottomRight()
        }

        for direction, corner in corners.items():
            if QRect(corner.x() - handle_size // 2,
                     corner.y() - handle_size // 2,
                     handle_size, handle_size).contains(pos):
                return direction
        return None

    def resize_rectangle(self, pos):
        """Resizes the rectangle based on the resize direction and mouse position."""
        delta = pos - self.last_pos
        new_rect = QRect(self.rect)
        min_size = self.resize_handle_size * 2

        # Update rectangle based on resize direction
        if self.resize_direction == 'top_left':
            new_rect.setTopLeft(self.rect.topLeft() + delta)
        elif self.resize_direction == 'top_right':
            new_rect.setTopRight(self.rect.topRight() + delta)
        elif self.resize_direction == 'bottom_left':
            new_rect.setBottomLeft(self.rect.bottomLeft() + delta)
        elif self.resize_direction == 'bottom_right':
            new_rect.setBottomRight(self.rect.bottomRight() + delta)

        # Ensure the rectangle stays within the parent's boundaries
        if self.parent():
            parent_rect = self.parent().rect()

            # Keep within parent bounds
            if new_rect.left() < parent_rect.left():
                new_rect.setLeft(parent_rect.left())
            if new_rect.top() < parent_rect.top():
                new_rect.setTop(parent_rect.top())
            if new_rect.right() > parent_rect.right():
                new_rect.setRight(parent_rect.right())
            if new_rect.bottom() > parent_rect.bottom():
                new_rect.setBottom(parent_rect.bottom())

            # Enforce minimum size
            if new_rect.width() < min_size:
                if self.resize_direction in ['top_left', 'bottom_left']:
                    new_rect.setLeft(self.rect.right() - min_size)
                else:
                    new_rect.setRight(self.rect.left() + min_size)
            if new_rect.height() < min_size:
                if self.resize_direction in ['top_left', 'top_right']:
                    new_rect.setTop(self.rect.bottom() - min_size)
                else:
                    new_rect.setBottom(self.rect.top() + min_size)

            self.rect = new_rect

    def get_rectangle_dimensions(self):
        """Returns the current rectangle's dimensions as (x, y, width, height)."""
        return self.rect.x(), self.rect.y(), self.rect.width(), self.rect.height()

    def set_rectangle_dimensions(self, x, y, width, height):
        """Sets the rectangle's dimensions."""
        self.rect = QRect(x, y, width, height)
        self.update()


class VideoPanel(QWidget):
    def __init__(self, parent, index, width, height):
        super().__init__()
        self.parent = parent
        self.index = index
        self.cap = None
        self.playing = False
        self.frame = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0

        # Set up UI
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setStyleSheet("background-color: #34495e;")
        self.layout.addWidget(self.image_label)

        # Controls
        self.controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")

        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.pause_button)
        self.controls_layout.addWidget(self.stop_button)
        self.layout.addLayout(self.controls_layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.layout.addWidget(self.slider)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.layout.addWidget(self.time_label)

        # Seek input layout
        self.seek_layout = QHBoxLayout()
        self.seek_input = QLineEdit()
        self.seek_input.setPlaceholderText("mm:ss.sss")
        self.seek_layout.addWidget(self.seek_input)
        self.seek_button = QPushButton("Seek")
        self.seek_layout.addWidget(self.seek_button)
        self.layout.addLayout(self.seek_layout)

        # Connect signals and slots
        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.slider.valueChanged.connect(self.seek_video)
        self.seek_button.clicked.connect(lambda: self.seek_to_timestamp())

    def update_dimensions(self):
        x, y, width, height = self.crop_rectangle.get_rectangle_dimensions()
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def show_preview(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.show_frame_on_label(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def play_video(self):
        if self.cap is not None and not self.playing:
            self.playing = True
            self.show_frame()

    def show_frame(self):
        if self.playing and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.slider.setValue(self.current_frame)
                self.update_time_label()
                self.show_frame_on_label(self.frame)
                QTimer.singleShot(int(1000 / self.fps), self.show_frame)
            else:
                self.playing = False

    def show_frame_on_label(self, frame):
        # Make a deep copy to avoid memory issues
        if frame is None or frame.size == 0:
            return

        frame_copy = frame.copy()

        # Ensure the frame is correctly sized
        resized_frame = cv2.resize(frame_copy, (self.width, self.height))

        # Convert color format
        cv2_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Create QImage with proper stride value
        bytes_per_line = cv2_image.strides[0]
        q_image = QImage(cv2_image.data.tobytes(), cv2_image.shape[1], cv2_image.shape[0],
                         bytes_per_line, QImage.Format_RGB888)

        # Convert to pixmap and keep a reference
        self.current_pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(self.current_pixmap)

    def pause_video(self):
        if self.playing:
            self.playing = False
            self.parent.status_label.setText(f"Video {self.index + 1} Paused")

    def stop_video(self):
        if self.cap is not None:
            self.playing = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.image_label.clear()
            self.slider.setValue(0)
            self.update_time_label()

    def seek_video(self):
        if self.cap is not None:
            self.current_frame = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_time_label()
            if not self.playing:
                self.show_frame()

    def update_time_label(self):
        current_time = self.convert_frame_to_time(self.current_frame, self.fps)
        total_time = self.convert_frame_to_time(self.total_frames, self.fps)
        self.time_label.setText(f"{current_time} / {total_time}")

    @staticmethod
    def convert_frame_to_time(frame, fps):
        total_seconds = frame / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{minutes:02}:{seconds:02}.{milliseconds:03}"

    def seek_to_timestamp(self, timestamp=None):
        if timestamp is None:
            timestamp = self.seek_input.text()
        try:
            # Parse the input format "mm:ss.sss"
            minutes, rest = timestamp.split(":")
            seconds, milliseconds = rest.split(".")
            minutes = int(minutes)
            seconds = int(seconds)
            milliseconds = int(milliseconds)

            # Convert the time to the corresponding frame number
            total_time_in_seconds = minutes * 60 + seconds + milliseconds / 1000.0
            frame_number = int(total_time_in_seconds * self.fps)

            # Set the slider and seek to the frame
            self.slider.setValue(frame_number)
            self.current_frame = frame_number
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_time_label()
            if not self.playing:
                self.show_frame()
        except ValueError:
            print("Invalid timestamp format. Use mm:ss.sss")


class SourceVideoPanel(VideoPanel):
    def __init__(self, parent, index, width=main_source_width, height=main_height):
        super().__init__(parent, index, width, height)
        self.image_label.setFixedSize(self.width, self.height)

        # Adding the crop rectangle directly onto the image label
        self.crop_rectangle = CropRectangle(self.image_label)
        self.crop_rectangle.setFixedSize(self.width, self.height)
        self.crop_rectangle.move(0, 0)
        self.crop_rectangle.center_crop_rectangle()

        self.open_button = QPushButton("Open Video")
        self.controls_layout.addWidget(self.open_button)
        self.open_button.clicked.connect(self.open_video)

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.playing = False
            self.current_frame = 0
            self.slider.setMaximum(self.total_frames)
            self.update_time_label()
            self.show_preview()
            self.parent.status_label.setText(f"Loaded: {file_path}")


class ResultVideoPanel(VideoPanel):
    def __init__(self, parent, width, height):
        super().__init__(parent, 3, width=width, height=height)
        self.isStreaming = False
        self.image_label.setFixedSize(self.width, self.height)

    def pre_stream(self, total_frames):
        if not self.isStreaming:
            self.total_frames = total_frames
            self.fps = 30
            self.current_frame = 0
            self.slider.setMaximum(self.total_frames)
            self.update_time_label()
            self.parent.status_label.setText("Streaming...")
            self.isStreaming = True

    def stream_frame(self, frame):
        if self.isStreaming:
            # Ensure frame has correct dimensions
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self.current_frame += 1
            self.slider.setValue(self.current_frame)
            self.update_time_label()
            self.show_frame_on_label(frame)


class VideoPlayerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.setWindowTitle("Virtual Context of Multi-Cameras")
        self.setGeometry(100, 100, 1800, 800)

        # Setup UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)

        # Create video panels
        # SourceVideoPanelWithLines = add_line_drawing_to_video_panel(ResultVideoPanel)

        self.panels = [SourceVideoPanel(self, i) for i in range(3)]
        for i, panel in enumerate(self.panels):
            self.grid_layout.addWidget(panel, 0, i)

        # Default parameters button
        self.default_parameters_button = QPushButton("Use Default Parameters")
        self.default_parameters_button.clicked.connect(self.set_default_parameters)
        self.main_layout.addWidget(self.default_parameters_button)

        # Checkboxes
        self.checkbox_layout = QHBoxLayout()
        self.normalize_checkbox = QCheckBox("Normalize Videos")
        self.stack_checkbox = QCheckBox("Stack Horizontally")
        self.yolo_checkbox = QCheckBox("Apply YOLO")

        self.checkbox_layout.addWidget(self.normalize_checkbox)
        self.checkbox_layout.addWidget(self.stack_checkbox)
        self.checkbox_layout.addWidget(self.yolo_checkbox)
        self.main_layout.addLayout(self.checkbox_layout)

        # Process button
        self.process_button = QPushButton("Process Videos")
        self.process_button.clicked.connect(self.process_videos)
        self.main_layout.addWidget(self.process_button)

        self.result_panel = None

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("background-color: #2c3e50; color: white; font: 12pt;")
        self.main_layout.addWidget(self.status_label)

        # Load YOLO model
        self.model = YOLO('yolo11x.pt')
        self.classNames = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def set_default_parameters(self):
        """Load default cropping parameters and timestamps from file."""
        try:
            with open('default_parameters.txt', 'r') as file:
                data = []
                for line in file:
                    values = line.strip().split(',')
                    x1 = int(values[0])
                    y1 = int(values[1])
                    width = int(values[2])
                    height = int(values[3])
                    start_time = values[4]
                    data.append((x1, y1, width, height, start_time))

            for i, panel in enumerate(self.panels):
                panel.crop_rectangle.set_rectangle_dimensions(*data[i][:4])
                panel.seek_to_timestamp(data[i][4])

            self.status_label.setText("Default parameters applied")
        except Exception as e:
            self.status_label.setText(f"Error loading parameters: {str(e)}")

    def process_videos(self):
        """Process the videos with selected options and create result video."""
        try:
            # Collect starting frames and update dimensions
            start_frames = [panel.slider.value() for panel in self.panels]
            result_video_height = main_height  # Set a default
            result_video_width = 0

            for i, panel in enumerate(self.panels):
                panel.update_dimensions()
                result_video_width += panel.width
                if panel.height > 0:  # Prevent division by zero
                    result_video_height = min(result_video_height, panel.height)

            # Set minimum dimensions to prevent errors
            result_video_height = max(result_video_height, 100)
            result_video_width = max(result_video_width, 100)

            # Create or update result panel
            if self.result_panel is None:

                self.result_panel = ResultVideoPanel(self, width=result_video_width, height=result_video_height)
                self.main_layout.addWidget(self.result_panel)
            else:
                self.result_panel.width = result_video_width
                self.result_panel.height = result_video_height
                self.result_panel.image_label.setFixedSize(result_video_width, result_video_height)

            # Check if all videos are loaded
            caps = []
            min_frame_count = float('inf')
            for i, panel in enumerate(self.panels):
                if panel.cap is not None and panel.cap.isOpened():
                    panel.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames[i])
                    caps.append(panel.cap)
                    frame_count = int(panel.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frames[i]
                    if frame_count > 0:
                        min_frame_count = min(min_frame_count, frame_count)

            if len(caps) < 3:
                self.status_label.setText("Please load all three videos before processing.")
                return

            if min_frame_count == float('inf') or min_frame_count <= 0:
                self.status_label.setText("Invalid frame count detected")
                return

            # Setup result video
            if hasattr(self.result_panel, 'cap') and self.result_panel.cap:
                self.result_panel.cap.release()
            self.result_panel.isStreaming = False
            self.result_panel.pre_stream(min_frame_count)

            # Ensure consistent dimensions
            output_size = (result_video_width, result_video_height)

            # Update result panel dimensions
            self.result_panel.width = output_size[0]
            self.result_panel.height = output_size[1]
            self.result_panel.image_label.setFixedSize(output_size[0], output_size[1])

            # Create video writer
            output_path = "result_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, output_size)

            frame_count = 0
            max_frames = min(min_frame_count, 3000)  # Cap at 3000 frames to prevent memory issues

            # Process frames
            try:
                for _ in range(max_frames):
                    if not self.is_running:
                        break

                    frames = []
                    valid_frames = True

                    for cap in caps:
                        if not cap.isOpened():
                            valid_frames = False
                            break

                        ret, frame = cap.read()
                        if not ret or frame is None:
                            valid_frames = False
                            break
                        frames.append(frame)

                    if not valid_frames or not frames:
                        break

                    # Apply selected processing options
                    try:
                        if self.normalize_checkbox.isChecked():
                            frames = self.normalize_videos(frames)

                        if self.stack_checkbox.isChecked():
                            stacked_frame = self.stack_videos_horizontally(frames)
                        else:
                            stacked_frame = frames[0]  # Default to the first frame if not stacking

                        if self.yolo_checkbox.isChecked():
                            stacked_frame = self.apply_yolo(stacked_frame)

                        # Display and save frame
                        if stacked_frame is not None and stacked_frame.size > 0:
                            # Make sure frame is the right size
                            if stacked_frame.shape[1] != output_size[0] or stacked_frame.shape[0] != output_size[1]:
                                stacked_frame = cv2.resize(stacked_frame, output_size)

                            self.result_panel.stream_frame(stacked_frame)
                            # cv2.imshow('stacked frame', stacked_frame)

                            out.write(stacked_frame)

                        frame_count += 1

                        # Process UI events to keep src responsive
                        QApplication.processEvents()

                    except Exception as e:
                        print(f"Frame processing error: {str(e)}")
                        continue

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Video processing error: {str(e)}")

            finally:

                out.release()

                try:
                    # Load the result video
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        self.result_panel.cap = cv2.VideoCapture(output_path)
                        self.result_panel.total_frames = int(self.result_panel.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        self.result_panel.fps = self.result_panel.cap.get(cv2.CAP_PROP_FPS)
                        self.result_panel.slider.setMaximum(self.result_panel.total_frames)
                        self.result_panel.show_preview()
                        self.status_label.setText(f"Processing completed. {frame_count} frames processed.")
                    else:
                        self.status_label.setText("Error: Result video could not be created")
                except Exception as e:
                    self.status_label.setText(f"Error loading result: {str(e)}")

        except Exception as e:
            self.status_label.setText(f"Processing error: {str(e)}")
            print(f"Exception in process_videos: {str(e)}")

    def stack_videos_horizontally(self, frames):
        return np.hstack(frames)

    def normalize_videos(self, frames):
        """Normalize the videos by cropping to the specified regions."""

        result_frames = []

        for i, panel in enumerate(self.panels):
            if i >= len(frames):
                continue

            frame = frames[i]
            if frame is None or frame.size == 0:
                continue

            try:
                frames[i] = cv2.resize(frames[i], (main_source_width, main_height))

                frames[i] = cv2.resize(frames[i][panel.y:panel.y + panel.height, panel.x:panel.x + panel.width],
                                       (panel.width, self.result_panel.height))

                result_frames.append(frames[i])

            except Exception as e:
                print(f"Error normalizing frame: {str(e)}")

        return result_frames

    def apply_yolo(self, frame):
        """Apply YOLO object detection to the frame and count objects using the embedded tracker."""
        try:
            if frame is None or frame.size == 0:
                return frame

            frame_copy = frame.copy()  # Work on a copy

            # Use the built-in tracker with the track method - this is the key change
            # The track method enables object tracking and assigns IDs
            results = self.model.track(frame_copy, persist=True, conf=0.3, iou=0.5, tracker='bytetrack.yaml')

            if results and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes

                for box in boxes:
                    try:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = (int(i) for i in box.xyxy[0])
                        w, h = ((x2 - x1), (y2 - y1))

                        # Check for valid dimensions
                        if w <= 0 or h <= 0:
                            continue

                        # Confidence
                        conf = float(box.conf[0])

                        # Class Name
                        cls = int(box.cls[0])
                        currentClass = self.classNames[cls]

                        # Filter by class and confidence
                        if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                            # Get tracking ID - accessing the embedded tracking information
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])

                                cvzone.cornerRect(frame_copy, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                                cvzone.putTextRect(frame_copy, f' {track_id}', (max(0, x1), max(35, y1)),
                                                   scale=2, thickness=3, offset=10)

                                # Calculate center point
                                cx, cy = x1 + w // 2, y1 + h // 2
                                cv2.circle(frame_copy, (cx, cy), 5, (255, 0, 255), cv2.FILLED)




                    except Exception as e:
                        print(f"Error processing detection box: {str(e)}")
                        continue
            return frame_copy

        except Exception as e:
            print(f"YOLO processing error: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return frame  # Return original frame on error

    def closeEvent(self, event):
        """Handle window close event with proper cleanup."""
        self.is_running = False

        # Release all video resources
        for panel in self.panels:
            if panel.cap:
                panel.cap.release()

        if self.result_panel and hasattr(self.result_panel, 'cap') and self.result_panel.cap:
            self.result_panel.cap.release()

        # Clean up references
        for panel in self.panels:
            panel.frame = None
            panel.current_pixmap = None

        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoPlayerApp()
    window.show()
    sys.exit(app.exec_())
