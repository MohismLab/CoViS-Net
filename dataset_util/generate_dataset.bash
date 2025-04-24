#!/usr/bin/env bash

echo "Generating dataset..."

# Default value
max=800
echo "Default max value set to $max"

# Set the destination folder name at the beginning
dest_folder="datasets/dataset_zip_$(date +%Y%m%d)"
echo "Destination folder set to $dest_folder"

# Check if an argument is provided
if [ $# -eq 1 ]; then
    # Check if the argument is a valid integer
    if [[ $1 =~ ^[0-9]+$ ]]; then
        max=$1
    else
        echo "Error: Argument must be a positive integer."
        exit 1
    fi
fi

echo "Starting dataset generation loop..."
for i in $(seq 0 $max); do 
    echo "Processing index $i"
    # python ./dataset_util/generate_dataset.py $i data/versioned_data/hm3d-1.0/hm3d/train $dest_folder
    python ./dataset_util/generate_dataset.py $i data/versioned_data/hm3d-0.2/hm3d/train $dest_folder
done

echo "Dataset generation completed."
