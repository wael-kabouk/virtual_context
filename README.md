# Virtual Context of Multi-Cameras

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Project Structure](#project-structure)
4. [Configuration](#configuration)
   - [YOLO Model](#yolo-model)
   - [ByteTrack Settings](#bytetrack-settings)
   - [Default Parameters](#default-parameters)
5. [Usage Guide](#usage-guide)
   - [Launching the Application](#launching-the-application)
   - [Loading and Navigating Videos](#loading-and-navigating-videos)
   - [Defining Regions of Interest (ROI)](#defining-regions-of-interest-roi)
   - [Processing Pipeline](#processing-pipeline)
   - [Result Panel Playback](#result-panel-playback)
6. [Customization](#customization)
7. [Troubleshooting & Tips](#troubleshooting--tips)
8. [Contributing](#contributing)

---

## Introduction

**Virtual Context of Multi-Cameras** is a PyQt5-based desktop application designed for synchronized processing of up to three video sources. It offers interactive ROI selection, video navigation controls, and a modular processing pipeline that includes frame normalization, horizontal stacking, and YOLO-based object detection with ByteTrack tracking. Processed outputs are previewed in real time and exported to a video file.

## Key Features

- **Multi-Source Input**: Simultaneously load and synchronize three video files.
- **Interactive ROI**: Drag-and-resize translucent rectangles to specify processing regions.
- **Standard Controls**: Play, pause, stop, and frame-wise navigation via sliders and timestamp input.
- **Preset Loading**: Import ROI coordinates and start times from `default_parameters.txt`.
- **Modular Pipeline**:
  - _Normalization_: Crop and resize frames to consistent dimensions.
  - _Stacking_: Concatenate frames horizontally or process primary source only.
  - _Detection_: Apply Ultralytics YOLO and ByteTrack for object detection and tracking.
- **Real-Time Preview**: Stream processed frames to a dedicated result panel during execution.
- **Output Export**: Save and reload the result video (`result_video.mp4` by default).
- **Robust Cleanup**: Ensures all video captures and GUI resources are released on exit.

## Getting Started

### Prerequisites

- **Operating System**: Windows, macOS, or Linux with GUI support.
- **Python**: 3.7 or newer.
- **Dependencies**:
  - PyQt5
  - opencv-python
  - numpy
  - cvzone
  - ultralytics
- **Hardware**:
  - CPU for video I/O.
  - (Optional) CUDA-enabled GPU for accelerated YOLO inference.
- **Video Codecs**: Ensure OpenCV supports your video formats (e.g., via FFmpeg).

### Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/wael-kabouk/virtual_context.git
   cd https://github.com/wael-kabouk/virtual_context.git
   ```
2. **(Optional) Create Virtual Environment**
   ```bash
   python -m venv venv
   # Activate venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install PyQt5 opencv-python numpy cvzone ultralytics
   ```
4. **Place Configuration Files**
   - YOLO model (e.g., `yolo11x.pt`)
   - `bytetrack.yaml`
   - `default_parameters.txt` (optional)

### Project Structure

```
/ project_root
|-- main.py                # Core application
|-- yolo11x.pt             # YOLO model weights
|-- bytetrack.yaml         # ByteTrack configuration
|-- default_parameters.txt # ROI & timestamp presets
|-- requirements.txt       # Python dependencies
|-- README.md              # This document
```

## Configuration

### YOLO Model

- Default: `yolo11x.pt` loaded in `VideoPlayerApp`
- To use a different model:
  1. Place your model file in the project root.
  2. Update:
     ```python
     self.model = YOLO('your_model.pt')
     ```

### ByteTrack Settings

Edit `bytetrack.yaml` to tune tracking:

```yaml
tracker_type: 'bytetrack'
track_high_thresh: 0.3
track_low_thresh: 0.2
new_track_thresh: 0.68
track_buffer: 100
match_thresh: 0.9
fuse_score: True
min_box_area: 10
frame_rate: 30
```

Adjust paths in `apply_yolo` if needed:

```python
results = self.model.track(frame_copy, persist=True, conf=0.3, iou=0.5, tracker='bytetrack.yaml')
```

### Default Parameters

Define ROIs and start times for each source. Format per line:

```
x,y,width,height,mm:ss.sss
```

Example:

```
0,0,500,300,00:00.500
107,0,345,297,00:02.000
111,0,389,291,00:03.300
```

Load via **Use Default Parameters** button.

## Usage Guide

### Launching the Application

```bash
python main.py
```

This initializes the Qt event loop and displays the main window.

### Loading and Navigating Videos

1. **Open Video**: Click per-panel button to select a file.
2. **Navigate**: Play, pause, stop, or drag the slider.
3. **Seek by Timestamp**: Enter `mm:ss.sss` and click **Seek**.

### Defining Regions of Interest (ROI)

- Drag the center of the blue rectangle to move.
- Drag corners to resize.
- The ROI defines the crop for normalization.

### Processing Pipeline

1. Set desired options:
   - Normalize Videos
   - Stack Horizontally
   - Apply YOLO
2. Click **Process Videos**.
3. Workflow:
   - Captures starting frame indexes from sliders.
   - Calculates minimum frame range (capped at 3000 frames).
   - Initializes/updates the result panel.
   - Iterates: read, crop, resize, stack, detect, track, preview, and write frames.
   - Maintains UI responsiveness via `QApplication.processEvents()`.
4. On finish, output is saved and loaded for playback.

### Result Panel Playback

Use the same controls as source panels to review the exported video.

## Customization

- **Detection Classes & Thresholds**: Modify `apply_yolo` logic.
- **Output Filename**: Change `result_video.mp4` in `process_videos`.
- **Tracking Parameters**: Tweak `bytetrack.yaml`.
- **Default Presets**: Extend parsing in `set_default_parameters`.

## Troubleshooting & Tips

- **Model Errors**: Verify model path and compatibility.
- **Video I/O Issues**: Ensure correct codecs and file access.
- **ROI Malfunction**: Check panel/frame size alignment.
- **Performance**: Use lighter models, reduce ROI, enable GPU, or lower frame cap.
- **Export Failures**: Try alternative codecs (e.g., `XVID`) and confirm write permissions.

## Contributing

1. Fork the repo and create a branch.
2. Adhere to PEP 8; include docstrings.
3. Test across platforms.
4. Submit a detailed pull request.

---

_End of README._
