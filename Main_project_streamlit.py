#!/usr/bin/env python3

import json
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics-cache"))

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from ultralytics import YOLO


def ensure_streamlit_config():
    try:
        config_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
        os.makedirs(config_dir, exist_ok=True)

        creds_file = os.path.join(config_dir, "credentials.toml")
        if not os.path.exists(creds_file):
            with open(creds_file, "w", encoding="utf-8") as file:
                file.write('email = ""\n')
    except Exception:
        pass


ensure_streamlit_config()
st.set_page_config(page_title="Video Analysis Suite", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stVideo"] {
        max-width: 760px;
        margin-left: auto;
        margin-right: auto;
    }

    div[data-testid="stVideo"] video {
        max-height: 420px;
        width: 100%;
        object-fit: contain;
        background: #000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_DIR = os.getcwd()
MODEL_MAP = {
    "Nano": "yolo11n.pt",
    "Small": "yolo11s.pt",
    "Medium": "yolo11m.pt",
    "Large": "yolo11l.pt",
    "XLarge": "yolo11x.pt",
}
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
TRAFFIC_CLASSES = {"car", "bus", "truck", "motorcycle"}


def get_available_models():
    available = []
    skipped = []
    for label, weights_path in MODEL_MAP.items():
        if not os.path.exists(weights_path):
            skipped.append(f"{label} ({weights_path}: file missing)")
            continue
        if not zipfile.is_zipfile(weights_path):
            skipped.append(f"{label} ({weights_path}: file is corrupted)")
            continue
        available.append(label)
    return available, skipped


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    try:
        return YOLO(weights_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model file '{weights_path}'. "
            "The file may be corrupted. Please choose another model or replace this weight file."
        ) from exc


def resolve_model_path(model_choice: str):
    weights_path = MODEL_MAP[model_choice]
    if not os.path.exists(weights_path):
        st.error(f"Model file not found: {weights_path}")
        st.stop()
    return weights_path


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(uploaded_file.getbuffer())
        return temp_video.name


def video_writer(output_path: str, fps: float):
    safe_fps = fps if fps and not np.isnan(fps) and fps > 0 else 25
    return imageio.get_writer(
        output_path,
        fps=safe_fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
    )


def draw_timeline(entries, group_key: str, title: str, ylabel: str):
    if not entries:
        st.info("No detections were recorded for this analysis.")
        return

    unique_names = sorted({entry[group_key] for entry in entries})
    name_to_idx = {name: idx for idx, name in enumerate(unique_names)}
    colors = plt.cm.get_cmap("tab20", max(len(unique_names), 1))

    fig_height = max(4, len(unique_names) * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for entry in entries:
        start = float(entry["startTime"][:-1])
        end = float(entry["endTime"][:-1])
        name = entry[group_key]
        idx = name_to_idx[name]
        duration = max(end - start, 0.05)
        ax.broken_barh([(start, duration)], (idx * 2 - 0.4, 0.8), facecolors=colors(idx))

    ax.set_yticks([name_to_idx[name] * 2 for name in unique_names])
    ax.set_yticklabels(unique_names)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    st.pyplot(fig)


def finalize_track(track_info):
    return {
        "name": track_info["class"],
        "label": track_info["label"],
        "trackId": track_info["trackId"],
        "startTime": f"{track_info['startTime']:.2f}s",
        "endTime": f"{track_info['lastSeenTime']:.2f}s",
        "duration": f"{(track_info['lastSeenTime'] - track_info['startTime']):.2f}s",
        "totalFramesDetected": track_info["detectedFrames"],
    }


def process_object_detection(input_path: str, model_choice: str):
    model = load_model(resolve_model_path(model_choice))
    class_list = model.names

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    safe_fps = fps if fps and not np.isnan(fps) and fps > 0 else 25

    output_video_path = os.path.join(APP_DIR, "output_video.mp4")
    json_path = os.path.join(APP_DIR, "tracking_data_output.json")
    writer = video_writer(output_video_path, safe_fps)

    tracking_timestamps = {}
    disappearance_counts = {}
    tracking_data_json = []
    debounce_limit = 10
    frame_number = 0
    progress = st.progress(0)

    st.write(f"Video Properties: {frame_width}x{frame_height}, {safe_fps:.2f} FPS, {total_frames} frames")

    with st.spinner("Processing object detection video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            timestamp = frame_number / safe_fps
            results = model.predict(frame, verbose=False)
            detected_ids = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)

                for idx, cls in enumerate(cls_arr):
                    cls = int(cls)
                    x1, y1, x2, y2 = xyxy_arr[idx]
                    conf = float(conf_arr[idx])
                    if conf < 0.7 or cls >= len(class_list):
                        continue

                    label = class_list[cls]
                    object_id = cls
                    detected_ids.append(object_id)

                    if object_id not in tracking_timestamps:
                        tracking_timestamps[object_id] = {
                            "class": label,
                            "label": label,
                            "trackId": object_id,
                            "startTime": timestamp,
                            "lastSeenTime": timestamp,
                            "detectedFrames": 1,
                        }
                        disappearance_counts[object_id] = 0
                    else:
                        tracking_timestamps[object_id]["lastSeenTime"] = timestamp
                        tracking_timestamps[object_id]["detectedFrames"] += 1
                        disappearance_counts[object_id] = 0

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (int(x1), max(int(y1) - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            for obj_id in list(tracking_timestamps.keys()):
                if obj_id in detected_ids:
                    continue
                disappearance_counts[obj_id] += 1
                if disappearance_counts[obj_id] > debounce_limit:
                    track_info = tracking_timestamps[obj_id]
                    if (track_info["lastSeenTime"] - track_info["startTime"]) >= 1:
                        tracking_data_json.append(finalize_track(track_info))
                    del tracking_timestamps[obj_id]
                    del disappearance_counts[obj_id]

            progress.progress(min(int((frame_number / total_frames) * 100), 100))
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for obj_id in list(tracking_timestamps.keys()):
        tracking_data_json.append(finalize_track(tracking_timestamps[obj_id]))

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(tracking_data_json, file, indent=4)

    cap.release()
    writer.close()
    progress.empty()
    return output_video_path, json_path, tracking_data_json


def process_traffic_analysis(input_path: str, model_choice: str):
    model = load_model(resolve_model_path(model_choice))
    class_list = model.names

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    safe_fps = fps if fps and not np.isnan(fps) and fps > 0 else 25

    output_video_path = os.path.join(APP_DIR, "traffic_output_video.mp4")
    json_path = os.path.join(APP_DIR, "traffic_tracking_data_output.json")
    writer = video_writer(output_video_path, safe_fps)

    tracking_timestamps = {}
    disappearance_counts = {}
    tracking_data_json = []
    per_frame_vehicle_counts = []
    class_counter = defaultdict(set)
    debounce_limit = 10
    frame_number = 0
    progress = st.progress(0)

    st.write(f"Video Properties: {frame_width}x{frame_height}, {safe_fps:.2f} FPS, {total_frames} frames")

    with st.spinner("Processing traffic video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            timestamp = frame_number / safe_fps
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            detected_ids = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                cls_arr = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([])
                conf_arr = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])
                ids_arr = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                if ids_arr is not None:
                    for idx in range(len(cls_arr)):
                        cls = int(cls_arr[idx])
                        if cls >= len(class_list):
                            continue

                        label = str(class_list[cls]).lower()
                        if label not in TRAFFIC_CLASSES:
                            continue

                        conf = float(conf_arr[idx]) if idx < len(conf_arr) else 0.0
                        if conf < 0.45:
                            continue

                        track_id = int(ids_arr[idx])
                        display_label = f"{label}#{track_id}"
                        detected_ids.append(track_id)
                        class_counter[label].add(track_id)

                        if track_id not in tracking_timestamps:
                            tracking_timestamps[track_id] = {
                                "class": label,
                                "label": display_label,
                                "trackId": track_id,
                                "startTime": timestamp,
                                "lastSeenTime": timestamp,
                                "detectedFrames": 1,
                            }
                            disappearance_counts[track_id] = 0
                        else:
                            tracking_timestamps[track_id]["lastSeenTime"] = timestamp
                            tracking_timestamps[track_id]["detectedFrames"] += 1
                            disappearance_counts[track_id] = 0

                        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(
                            frame,
                            display_label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 255),
                            2,
                        )

            per_frame_vehicle_counts.append(len(detected_ids))

            for obj_id in list(tracking_timestamps.keys()):
                if obj_id in detected_ids:
                    continue
                disappearance_counts[obj_id] += 1
                if disappearance_counts[obj_id] > debounce_limit:
                    tracking_data_json.append(finalize_track(tracking_timestamps[obj_id]))
                    del tracking_timestamps[obj_id]
                    del disappearance_counts[obj_id]

            progress.progress(min(int((frame_number / total_frames) * 100), 100))
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for obj_id in list(tracking_timestamps.keys()):
        tracking_data_json.append(finalize_track(tracking_timestamps[obj_id]))

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(tracking_data_json, file, indent=4)

    cap.release()
    writer.close()
    progress.empty()

    duration_seconds = total_frames / safe_fps if safe_fps else 0
    duration_minutes = duration_seconds / 60 if duration_seconds else 0
    unique_vehicle_count = len({entry["trackId"] for entry in tracking_data_json})
    analytics = {
        "unique_vehicles": unique_vehicle_count,
        "peak_concurrent_vehicles": max(per_frame_vehicle_counts) if per_frame_vehicle_counts else 0,
        "average_vehicles_per_minute": round(unique_vehicle_count / duration_minutes, 2) if duration_minutes else 0.0,
        "vehicle_breakdown": {label.title(): len(track_ids) for label, track_ids in sorted(class_counter.items())},
    }

    return output_video_path, json_path, tracking_data_json, analytics


def render_upload_section(title: str, upload_key: str, stored_path_key: str):
    st.subheader(title)
    uploaded_video = st.file_uploader("Upload a video file", type=VIDEO_TYPES, key=upload_key)

    if uploaded_video is not None:
        st.write(f"Selected: {uploaded_video.name}")
        if st.button("Use this video", key=f"{upload_key}_confirm"):
            st.session_state[stored_path_key] = save_uploaded_file(uploaded_video)
            st.success("Video ready for analysis.")

    return st.session_state.get(stored_path_key)


def render_compact_video(video_path: str, fit_screen: bool = False):
    left_col, center_col, right_col = st.columns([1, 3, 1])
    with center_col:
        if fit_screen:
            cap = cv2.VideoCapture(video_path)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            max_display_width = 760
            max_display_height = 720
            if video_width > 0 and video_height > 0:
                display_width = min(
                    max_display_width,
                    video_width,
                    int(max_display_height * video_width / video_height),
                )
            else:
                display_width = 480
            st.video(video_path, width=max(display_width, 1))
        else:
            st.video(video_path)


st.title("Video Analysis Suite")
st.caption("Choose a tool in the sidebar to analyze your video.")

AVAILABLE_MODELS, SKIPPED_MODELS = get_available_models()
if not AVAILABLE_MODELS:
    st.error("No valid YOLO model files were found in the project folder.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    option = st.radio("Choose a tool", ["Object Detection", "Traffic Analysis"], index=None)
    model_choice = st.selectbox(
        "Detection model",
        AVAILABLE_MODELS,
        index=min(AVAILABLE_MODELS.index("Small"), len(AVAILABLE_MODELS) - 1) if "Small" in AVAILABLE_MODELS else 0,
    )
    if SKIPPED_MODELS:
        st.caption("Unavailable models: " + ", ".join(SKIPPED_MODELS))

backgrounds = {
    None: "entry.css",
    "Object Detection": "objects.css",
    "Traffic Analysis": "traffic.css",
}
st.html(Path(__file__).with_name("backgrounds") / backgrounds[option])

if option is None:
    st.stop()

if option == "Object Detection":
    input_video_path = render_upload_section("Object Detection", "od_upload", "od_input_path")

    if input_video_path and st.button("Run Object Detection", type="primary"):
        output_video_path, json_file_path, tracking_data = process_object_detection(input_video_path, model_choice)

        st.subheader("Output Video")
        if os.path.exists(output_video_path):
            render_compact_video(output_video_path, fit_screen=True)
            with open(output_video_path, "rb") as file:
                st.download_button("Download annotated video", file, file_name="output_video.mp4")

        st.subheader("Detection Data")
        st.json(tracking_data)
        with open(json_file_path, "rb") as file:
            st.download_button("Download detection JSON", file, file_name="tracking_data_output.json")

        st.subheader("Timeline Graph")
        draw_timeline(tracking_data, group_key="name", title="Object Detection Timeline", ylabel="Objects")
else:
    input_video_path = render_upload_section("Traffic Analysis", "ta_upload", "ta_input_path")

    if input_video_path and st.button("Run Traffic Analysis", type="primary"):
        output_video_path, json_file_path, tracking_data, analytics = process_traffic_analysis(input_video_path, model_choice)

        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Vehicles", analytics["unique_vehicles"])
        col2.metric("Peak Concurrent Vehicles", analytics["peak_concurrent_vehicles"])
        col3.metric("Vehicles / Minute", analytics["average_vehicles_per_minute"])

        if analytics["vehicle_breakdown"]:
            st.subheader("Vehicle Breakdown")
            st.json(analytics["vehicle_breakdown"])

        st.subheader("Annotated Traffic Video")
        if os.path.exists(output_video_path):
            render_compact_video(output_video_path)
            with open(output_video_path, "rb") as file:
                st.download_button("Download traffic video", file, file_name="traffic_output_video.mp4")

        st.subheader("Traffic Data")
        st.json(tracking_data)
        with open(json_file_path, "rb") as file:
            st.download_button("Download traffic JSON", file, file_name="traffic_tracking_data_output.json")

        st.subheader("Vehicle Timeline")
        draw_timeline(tracking_data, group_key="label", title="Traffic Timeline", ylabel="Vehicles")
