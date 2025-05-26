#!/bin/bash
# Setup script for MeshFleet evaluation

# Set default data directory if not provided
if [ -z "$MESHFLEET_DATA_DIR" ]; then
    export MESHFLEET_DATA_DIR="./data/meshfleet"
fi

echo "Setting up MeshFleet evaluation data..."
echo "Data directory: $MESHFLEET_DATA_DIR"

# Download and unpack the benchmark data
python -m scripts.download_meshfleet_benchmark --output-dir "$MESHFLEET_DATA_DIR"

CONFIG_FILE="meshfleet_benchmark/benchmark_config.yaml"
echo -e "\nSetup complete!"
echo "Next steps:"
echo "1. Update generated_folder in $CONFIG_FILE"
echo "2. Place your generated images in the specified folder"
echo "3. Run: python run_meshfleet_eval.py"
