#!/bin/bash

# Variables
CONFIG="config_modified.json"
AUDIO_DIR="processed_dataset"
REP_DIR="hubert_rep"
EXTS="flac,wav"
SPLIT_SEED=42
VALID_SET_SIZE=0.00467
# for debuging reduce dataset size 
# VALID_SET_SIZE=0.243
# dont change these values below: 
TEACHER="hubert"
TENSORBOARD_LOGDIR="saved_files/logs/"

# Script to run HuBERT-related tasks
echo "Running HuBERT representation extraction with the following parameters:"
echo "Config: $CONFIG"
echo "Audio Directory: $AUDIO_DIR"
echo "Representation Directory: $REP_DIR"
echo "Extensions: $EXTS"
echo "Split Seed: $SPLIT_SEED"
echo "Validation Set Size: $VALID_SET_SIZE"

python hubert_rep_modified.py --config "$CONFIG" --audio_dir "$AUDIO_DIR" --rep_dir "$REP_DIR" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"

echo "HuBERT representation extraction completed."

echo "Starting training with HuBERT..."
echo "Teacher: $TEACHER"

accelerate launch train_modified.py --config "$CONFIG" --teacher "$TEACHER"

echo "HuBERT training completed."

echo "Starting TensorBoard for HuBERT logs..."
echo "Log Directory: $TENSORBOARD_LOGDIR"

tensorboard --logdir="$TENSORBOARD_LOGDIR"

echo "TensorBoard started for HuBERT."
