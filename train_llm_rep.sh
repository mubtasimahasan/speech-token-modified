#!/bin/bash

# Variables
CONFIG="config_modified.json"
AUDIO_DIR="processed_dataset"
REP_DIR="llm_rep"
EXTS="flac,wav"
SPLIT_SEED=42
TEACHER="llm"
VALID_SET_SIZE=0.00467
# VALID_SET_SIZE=0.34

# Script
echo "Extracting LLM representation."
python llm_rep_extraction.py --config "$CONFIG" --audio_dir "$AUDIO_DIR" --rep_dir "$REP_DIR" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"
echo "LLM representation extraction completed."

echo "Starting training with LLM..."
accelerate launch train_modified.py --config "$CONFIG" --teacher "$TEACHER"
echo "LLM training completed."

