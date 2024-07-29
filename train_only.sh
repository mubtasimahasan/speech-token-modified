#!/bin/bash

echo "Before running this script, make sure you have previously executed all three './process_dataset.sh', './train_llm_rep.sh', and './train_llm_rep.sh' scripts."

# Variables
CONFIG="config_modified.json"           # Path to the configuration file
HUBERT_TEACHER="hubert"                 # Teacher model name for HuBERT. 
LLM_TEACHER="llm"                       # Teacher model name for LLM.

FLAG=""
# If argement "resume" is passed then train will continue from previously saved model.
if [[ "$1" == "resume" ]]; then
    FLAG="--continue_train"
fi

# Training with HuBERT
echo "Starting training with HuBERT..."
accelerate launch train_modified.py --config "$CONFIG" --teacher "$HUBERT_TEACHER" "$FLAG"

# Training with LLM
echo "Starting training with LLM..."
accelerate launch train_modified.py --config "$CONFIG" --teacher "$LLM_TEACHER" "$FLAG" 

