#!/bin/bash

# Sprawdzenie czy podano argument (ścieżkę do folderu)
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/folder"
  exit 1
fi

BASE_DIR="$1"

echo "Uruchamiam skrypt Pythona..."
python3 script.py "$BASE_DIR"


echo "Uruchamiam skrypt R do wizualizacji..."
Rscript visualization.R "$BASE_DIR"


echo "Gotowe."
