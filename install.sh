#!/bin/bash

echo "Instalacja zależności Pythona..."

pip install numpy torch dgl scikit-learn matplotlib pandas

# Uwaga: w przypadku DGL może być konieczna instalacja specyficzna dla Twojej konfiguracji (CPU/CUDA).
# Sprawdź dokumentację DGL, jeśli potrzebujesz innej komendy instalacyjnej.

echo "Instalacja zależności R..."

Rscript -e "if (!require('R4RNA')) install.packages('R4RNA', repos='https://cran.r-project.org')"

echo "Instalacja zakończona."
