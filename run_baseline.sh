#!/bin/bash

# Check if the "N" argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <N>"
    echo "N: The 3-digit number to iterate up to (e.g., 117)"
    exit 1
fi

N=$1

# Loop from 1 to N
for (( i=1; i<=N; i++ ))
do
    # Format the number as a 3-digit string (e.g., "001", "042", "117")
    ID=$(printf "%03d" $i)

    echo "--- Processing ID: $ID ---"

    # Create the output directory
    mkdir -p "preds/$ID"

    # Run v3_landmarks.py without smoothing
    python3 v3_landmarks.py \
        --no_smooth \
        --video_input "300VW_Dataset_2015_12_14/$ID/vid.avi" \
        --o_landmarks "preds/$ID/baseline.npy"

    echo "--- Finished ID: $ID ---"
    echo ""
done

echo "Baseline processing complete."
