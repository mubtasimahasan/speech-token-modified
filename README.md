# Project Name

## Overview

**Work in progress.** 

## Scripts

To run the project, execute the following scripts in order:

### 1. Process Dataset

Run the `process_dataset.sh` script to process the audio dataset.

```bash
./process_dataset.sh
```

### 2. Train with HuBERT

Run the `train_hubert_rep.sh` script to extract representations using the HuBERT model, train the model, and start TensorBoard for monitoring.

```bash
./train_hubert_rep.sh
```

### 3. Train with LLM

Run the `train_llm_rep.sh` script to extract representations using a Large Language Model (LLM), train the model, and start TensorBoard for monitoring.

```bash
./train_llm_rep.sh
```

### 4. Train Only

Run the `train_only.sh` script to train both HuBERT and LLM models and start TensorBoard. 

```bash
# Note: Use this script when you have previously executed all three './process_dataset.sh', './train_llm_rep.sh', and './train_llm_rep.sh' scripts, and want to continue training or retrain now.

./train_only.sh
```

---

#### If Permission Denied
If you encounter a permission denied error when running the scripts, make sure they are executable by running:

```
chmod +x process_dataset.sh train_hubert.sh train_llm.sh
```

#### To Change Variables
Each script contains variables at the top that you can modify as needed:

- **`process_dataset.sh` Variables:**
  - `DATASET`: Dataset to be processed.
  - `SPLIT`: Split to use (e.g., `train`).
  - `RATIO`: Ratio of the dataset to process.
  - `SEGMENT_LENGTH`: Length of each audio segment in seconds.

- **`train_hubert.sh` and  `train_hubert.sh` Variables:**
  - `CONFIG`: Path to the configuration file.
  - `AUDIO_DIR`: Directory containing processed audio files.
  - `REP_DIR`: Directory to save HuBERT representations.
  - `EXTS`: Audio file extensions to process.
  - `SPLIT_SEED`: Random seed for dataset splitting.
  - `VALID_SET_SIZE`: Proportion of the dataset for validation.
  - `TENSORBOARD_LOGDIR`: Directory to save TensorBoard logs.

- **`train_only.sh` Variables:**
  - `HUBERT_TEACHER`: Teacher model name for HuBERT
  - `LLM_TEACHER`: Teacher model name for LLM
  - `TENSORBOARD_LOGDIR`: TensorBoard log directory
