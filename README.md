# Video Analysis Suite

A Streamlit-based video analysis project for object detection and traffic analysis using YOLO.

## Features

- Upload and analyze video files from a browser UI.
- Switch between Object Detection and Traffic Analysis modes.
- Detect objects using YOLO model weights.
- Track traffic vehicles such as cars, buses, trucks, and motorcycles.
- Generate annotated output videos.
- Export detection and traffic data as JSON.
- Display object and vehicle timeline graphs.
- Show traffic metrics such as unique vehicles, peak concurrent vehicles, and vehicles per minute.

## Tech Stack

- Python
- Streamlit
- OpenCV
- Ultralytics YOLO
- Matplotlib
- NumPy
- ImageIO and FFmpeg

## Project Files

- `Main_project_streamlit.py`: main Streamlit application.
- `run_app.sh`: one-command launcher for macOS/Linux terminal.
- `requirements.txt`: Python dependencies.
- `yolo11n.pt` and `yolo11s.pt`: included YOLO model weights.

## How To Run

```bash
cd "/Users/ayush/Desktop/object and traffic analyzer working "
python3 -m pip install -r requirements.txt
./run_app.sh
```

Then open:

```text
http://localhost:8501
```

## Notes

- Large generated files such as videos, JSON outputs, datasets, virtual environments, and training runs are not pushed to GitHub.
- `yolo11m.pt` is not included because the local copy is corrupted.
- `yolo11x.pt` is not included because it is larger than GitHub's normal file limit.
- To use additional YOLO weights, place the `.pt` files in the project folder.

## Modes

### Object Detection

Detects general objects in the uploaded video and produces an annotated output video, JSON data, and a timeline graph.

### Traffic Analysis

Detects and tracks vehicles, then reports traffic statistics such as unique vehicle count, peak concurrent vehicles, vehicles per minute, and vehicle type breakdown.
