#!/bin/bash

# Variables
CONFIG="config_modified.json"
AUDIO_DIR="processed_dataset"
REP_DIR="llm_rep"
EXTS="flac,wav"
SPLIT_SEED=42
VALID_SET_SIZE=0.243
TEACHER="llm"
TENSORBOARD_LOGDIR="saved_files/llm/logs/"

# Script to run LLM-related tasks
echo "Running LLM representation extraction with the following parameters:"
echo "Config: $CONFIG"
echo "Audio Directory: $AUDIO_DIR"
echo "Representation Directory: $REP_DIR"
echo "Extensions: $EXTS"
echo "Split Seed: $SPLIT_SEED"
echo "Validation Set Size: $VALID_SET_SIZE"

python llm_rep_extraction.py --config "$CONFIG" --audio_dir "$AUDIO_DIR" --rep_dir "$REP_DIR" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"

echo "LLM representation extraction completed."

echo "Starting training with LLM..."
echo "Teacher: $TEACHER"

accelerate launch train_modified.py --config "$CONFIG" --teacher "$TEACHER"

echo "LLM training completed."

echo "Starting TensorBoard for LLM logs..."
echo "Log Directory: $TENSORBOARD_LOGDIR"

tensorboard --logdir="$TENSORBOARD_LOGDIR"

echo "TensorBoard started for LLM."
