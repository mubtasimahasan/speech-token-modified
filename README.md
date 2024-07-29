# Project Name

**Work in progress.**

## Run Using **Docker**

To run the project using a Docker image, follow these steps:

### 1. Pull the Docker Image

First, pull the Docker image from Docker Hub:

```bash
docker pull mubtasimahasan/speech-token:latest
```

### 2. Run the Docker Container

Launch the Docker container with GPU support and bind the `./saved_files` directory to save output files:

```bash
docker run --rm -it --gpus all \
    -v ./saved_files:/app/saved_files \
    mubtasimahasan/speech-token:latest
```

This command will start the container, mapping the `saved_files` directory inside the container to the `saved_files` directory on your local machine.

### 3. Execute the Training Script

Inside the running Docker container, execute the training script. This script will download and process the dataset, extract semantic features, and run the training:

```bash
./docker_train.sh
```

Alternatively, execute the training script for debugging. This script will run the training with only 12 training samples and for 500 epochs:

```bash
./docker_train.sh debug
```

### Notes

- **GPU Support**: Ensure that your system is set up with the necessary NVIDIA drivers and Docker's GPU support to utilize the `--gpus all` option.

- **Output Files**: All output files will be stored in the `./saved_files` directory on your host machine.

- **Permissions**: Make sure that the current user has write permissions for the `saved_files` directory on the host machine.

## Run Using **Scripts**

To run the project, execute the following scripts in order:

### 1. Download and Process Dataset

Run the `process_dataset.sh` script to process the audio dataset:

```bash
./process_dataset.sh
```

### 2. Train with HuBERT Representations

Run the `train_hubert_rep.sh` script to extract representations using the HuBERT model and train the model:

```bash
./train_hubert_rep.sh
```

### 3. Train with LLM Representations

Run the `train_llm_rep.sh` script to extract representations using a Large Language Model (LLM) and train the model:

```bash
./train_llm_rep.sh
```

### 4. Retrain or Resume Training

Run the `train_only.sh` script to train with a previously extracted dataset and representations. **Note:** Use this script only if you have previously executed the `./process_dataset.sh`, `./train_hubert_rep.sh`, and `./train_llm_rep.sh` scripts.

```bash
./train_only.sh
```

To continue training from a previously saved model, add the argument "resume" with the shell file:

```bash
./train_only.sh resume 
```

---

### Check Logs

Model training logs will be automatically saved to the `./saved_files/logs` directory using TensorBoard. To check the logs, run:

```bash
tensorboard --logdir="./saved_files/logs"
```

Alternatively, you can log into your Weights & Biases (WandB) account by setting your API key with the environment variable:

```bash
export WANDB_API_KEY={YOUR API KEY HERE}
```

### If Permission Denied

If you encounter a "permission denied" error when running the scripts, make sure they are executable by running:

```bash
chmod +x process_dataset.sh train_hubert_rep.sh train_llm_rep.sh train_only.sh
```

### Change Variables

Each script contains variables at the top that you can modify as needed:

- `process_dataset.sh` Variables:
  - `DATASET`: Dataset to be processed.
  - `SPLIT`: Split to use (e.g., `train`).
  - `RATIO`: Ratio of the dataset to process.
  - `SEGMENT_LENGTH`: Length of each audio segment in seconds.

- `train_hubert.sh` and  `train_hubert.sh` Variables:
  - `CONFIG`: Path to the configuration file.
  - `AUDIO_DIR`: Directory containing processed audio files.
  - `REP_DIR`: Directory to save HuBERT representations.
  - `EXTS`: Audio file extensions to process.
  - `SPLIT_SEED`: Random seed for dataset splitting.
  - `VALID_SET_SIZE`: Proportion of the dataset for validation.

- `train_only.sh` Variables:
  - `CONFIG`: Path to the configuration file.
  - `HUBERT_TEACHER`: Teacher model name for HuBERT
  - `LLM_TEACHER`: Teacher model name for LLM
