#!/usr/bin/env bash
# Usage: ./run_all.sh /path/to/train.csv      (Hadoop/HDFS must be running)
set -euo pipefail
CSV="${1:?path to Kaggle NYC taxi train.csv}"
hdfs dfs -mkdir -p /rides/raw
hdfs dfs -put -f "$CSV" /rides/raw/train.csv
for job in 01_clean 02_features 03_train 04_predict_grid 05_allocate; do
  echo "=== $job ==="
  spark-submit --driver-memory 4g spark/$job.py
done
echo "Done. Start the dashboard with: streamlit run dashboard/app.py"
