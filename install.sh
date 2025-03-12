#!/bin/bash
# Skrypt instalacyjny zależności dla projektu

echo "Instalacja zależności Pythona..."

# Instalacja bibliotek Python przy pomocy pip
pip install numpy torch dgl scikit-learn matplotlib pandas

# Uwaga: w przypadku DGL może być konieczna instalacja specyficzna dla Twojej konfiguracji (CPU/CUDA).
# Sprawdź dokumentację DGL, jeśli potrzebujesz innej komendy instalacyjnej.

echo "Instalacja zależności R..."
# Instalacja biblioteki R4RNA (CRAN)
Rscript -e "if (!require('R4RNA')) install.packages('R4RNA', repos='https://cran.r-project.org')"

echo "Instalacja zakończona."
