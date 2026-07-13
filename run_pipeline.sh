#!/bin/bash
set -e

echo "======================================================="
echo " MAGIC Gamma Telescope - ML Classification Pipeline"
echo "======================================================="

DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
DATA_FILE="data/magic04.data"

if [ ! -d "data" ]; then
    mkdir data
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "[UNIX] Downloading MAGIC dataset from UCI Machine Learning Repository..."
    wget -q --show-progress -O "$DATA_FILE" "$DATA_URL"
    echo "Download complete."
else
    echo "[UNIX] Found cached dataset at $DATA_FILE."
fi

echo ""
echo "[UNIX] Performing basic CLI data inspection..."
# Use wc to count total events
TOTAL_EVENTS=$(wc -l < "$DATA_FILE")
echo "Total events in dataset: $TOTAL_EVENTS"

# Use awk to count how many are Gamma (g) vs Hadron (h)
# The class label is the 11th column, separated by commas
GAMMA_COUNT=$(awk -F',' '$11 == "g" {count++} END {print count}' "$DATA_FILE")
HADRON_COUNT=$(awk -F',' '$11 == "h" {count++} END {print count}' "$DATA_FILE")

echo "Gamma Rays (Signal): $GAMMA_COUNT"
echo "Hadrons (Background): $HADRON_COUNT"
echo ""

echo "[PYTHON] Initializing Machine Learning Pipeline..."
python3 ml_analysis.py

echo "======================================================="
echo " Pipeline finished! Check output/ for the ROC plots."
echo "======================================================="
