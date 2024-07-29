#!/bin/bash

# Variables
CONFIG="config_modified.json"
AUDIO_DIR="processed_dataset"
REP_DIR="hubert_rep"
EXTS="flac,wav"
SPLIT_SEED=42
TEACHER="hubert"
VALID_SET_SIZE=0.00467
# VALID_SET_SIZE=0.34

# Script
echo "Extracting HuBERT representation."
python hubert_rep_modified.py --config "$CONFIG" --audio_dir "$AUDIO_DIR" --rep_dir "$REP_DIR" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"
echo "HuBERT representation extraction completed."

echo "Starting training with HuBERT..."
accelerate launch train_modified.py --config "$CONFIG" --teacher "$TEACHER"
echo "HuBERT training completed."

