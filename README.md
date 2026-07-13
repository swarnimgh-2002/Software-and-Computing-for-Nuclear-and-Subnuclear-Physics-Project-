# Astroparticle Physics: Gamma vs Hadron ML Classifier

## Overview
This project uses Machine Learning to solve a classic problem in high-energy astroparticle physics: distinguishing between atmospheric Cherenkov radiation showers caused by high-energy Gamma rays (Signal) and those caused by Hadronic cosmic rays (Background).

It uses the MAGIC Gamma Telescope dataset and trains a Random Forest Classifier to achieve high-accuracy separation.

## Project Structure
- `ml_analysis.py`: The Python script that trains the ML model and evaluates it using ROC curves.
- `run_pipeline.sh`: A UNIX shell script that downloads the dataset, performs command-line data inspection using `awk`, and runs the Python pipeline.
- `Dockerfile` & `requirements.txt`: Ensures the ML environment (scikit-learn, pandas) is completely reproducible.
- `REPORT.md`: A formal academic report detailing the physics context and software methodology.

## Execution (Docker)
1. Build the isolated environment:
   ```bash
   docker build -t magic-ml-classifier .
   ```
2. Run the pipeline:
   ```bash
   docker run --rm -v $(pwd)/output:/app/output magic-ml-classifier
   ```
   *(On Windows PowerShell, replace `$(pwd)` with `${PWD}`)*

The pipeline will download the data, train the AI, and output a `ml_evaluation_plots.pdf` in the `output/` folder showing the ROC curve and Feature Importances.

**Read REPORT_Final.md** for full explaination of the project and results.
