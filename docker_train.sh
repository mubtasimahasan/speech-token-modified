#!/bin/bash

# Variables
DATASET="zachary24/librispeech_train_clean_100"
SPLIT="train"
SEGMENT_LENGTH=3.0
CONFIG="config_modified.json"
AUDIO_DIR="processed_dataset"
REP_DIR_HUBERT="hubert_rep"
REP_DIR_LLM="llm_rep"
EXTS="flac,wav"
SPLIT_SEED=42
HUBERT_TEACHER="hubert"
LLM_TEACHER="llm"
EPOCHS=50
RATIO=1.0
VALID_SET_SIZE=0.00467

# If argument "debug" is passed then these values will be changed to:
# Training samples: 12 and Valid samples: 6
if [[ "$1" == "debug" ]]; then
    EPOCHS=500
    RATIO=0.00084
    VALID_SET_SIZE=0.34
fi

# Step 1: Process Dataset
python process_dataset.py --dataset "$DATASET" --split "$SPLIT" --ratio "$RATIO" --segment_length "$SEGMENT_LENGTH"

# Step 2: Extract HuBERT Representations
python hubert_rep_modified.py --config "$CONFIG" --audio_dir "$AUDIO_DIR" --rep_dir "$REP_DIR_HUBERT" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"

# Step 3: Extract LLM Representations
python llm_rep_extraction.py --config "$CONFIG" --audio_dir "$AUDIO_DIR" --rep_dir "$REP_DIR_LLM" --exts "$EXTS" --split_seed "$SPLIT_SEED" --valid_set_size "$VALID_SET_SIZE"

# Step 4: Train with HuBERT
accelerate launch train_modified.py --epochs "$EPOCHS" --config "$CONFIG" --teacher "$HUBERT_TEACHER"

# Step 5: Train with LLM
accelerate launch train_modified.py --epochs "$EPOCHS" --config "$CONFIG" --teacher "$LLM_TEACHER"
