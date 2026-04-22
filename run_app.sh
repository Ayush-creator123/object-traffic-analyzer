#!/bin/zsh
set -e

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-cache"
export YOLO_CONFIG_DIR="${TMPDIR:-/tmp}/ultralytics-cache"

mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"

python3 -m streamlit run Main_project_streamlit.py --server.headless true --server.enableCORS true
