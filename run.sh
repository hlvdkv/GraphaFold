#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/folder"
  exit 1
fi

BASE_DIR="$1"

echo "Running Python script..."
python3 script.py "$BASE_DIR"

echo "Running R script for visualization..."
Rscript visualization.r "$BASE_DIR"

echo "Done."
