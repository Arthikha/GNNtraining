#!/bin/bash
set -e  # Exit on error

echo "Running create_neo4j.py..."
python src/train/create_neo4j.py

echo "Running train_graphsage.py..."
python src/train/train_graphsage.py

echo "All scripts completed."