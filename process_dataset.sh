#!/bin/bash

# Variables
DATASET="zachary24/librispeech_train_clean_100"
SPLIT="train"
RATIO=1.0
# for debuging reduce dataset size 
# RATIO=0.00616 
SEGMENT_LENGTH=3.0

# Script to process the dataset
echo "Running process_dataset.py script with the following parameters:"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Ratio: $RATIO"
echo "Segment Length: $SEGMENT_LENGTH"

python process_dataset.py --dataset "$DATASET" --split "$SPLIT" --ratio "$RATIO" --segment_length "$SEGMENT_LENGTH"

echo "Dataset processing completed."
