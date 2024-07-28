#!/bin/bash

echo "Before running this script, make sure you have previously executed all three './process_dataset.sh', './train_llm_rep.sh', and './train_llm_rep.sh' scripts."

# Variables
CONFIG="config_modified.json"           # Path to the configuration file
# Do not change the variables below:
HUBERT_TEACHER="hubert"                 # Teacher model name for HuBERT. 
LLM_TEACHER="llm"                       # Teacher model name for LLM.
TENSORBOARD_LOGDIR="saved_files/logs"   # TensorBoard log directory

# Training with HuBERT
echo "Starting training with HuBERT..."
accelerate launch train_modified.py --config "$CONFIG" --teacher "$HUBERT_TEACHER"

# Training with LLM
echo "Starting training with LLM..."
accelerate launch train_modified.py --config "$CONFIG" --teacher "$LLM_TEACHER"

# Launch TensorBoard
echo "Starting TensorBoard..."
tensorboard --logdir="$TENSORBOARD_LOGDIR"
