#!/bin/bash

# Usage: ./run_all.sh [train]
# If "train" is passed as an argument, train.py will be executed.

# Project root directory
PROJECT_ROOT=$(pwd)

# Function to run a Python file
run_python_file() {
    local file=$1
    echo "Running $file..."
    python -m "${file}" || { echo "Error running $file"; exit 1; }
    echo "$file executed successfully."
}

# Step 1: Run config/config.py (if necessary for validation)
echo "Checking config/config.py..."
python -m config.config || { echo "Error in config/config.py"; exit 1; }
echo "config/config.py is valid."

# Step 2: Run util/util.py
echo "Running util/util.py..."
run_python_file "util.util"

# Step 3: Run model/lstm_model.py
echo "Running model/lstm_model.py..."
run_python_file "model.lstm_model"

# Step 4: Optionally run train.py
if [ "$1" == "train" ]; then
    echo "Running train.py..."
    python -m train || { echo "Error in train.py"; exit 1; }
    echo "train.py executed successfully."
else
    echo "Skipping train.py execution. Pass 'train' as an argument to run it."
fi

echo "All files executed successfully."
