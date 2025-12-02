# Assignment 2: Semantic Role Labelling

## Project Structure
*   `2_serov.ipynb`: The main Jupyter Notebook containing the full pipeline (Data Loading, Preprocessing, Training, Evaluation, Submission).
*   `plots/`: Directory containing generated training loss and validation comparison charts.
*   `tables/`: Directory containing metric comparison CSVs.
*   `submissions/`: Directory containing `.zip` files for CodaLab submission (Dev/Test for Best/Last models).
*   `Dockerfile`, `build`, `launch`: Utilities for the containerized environment.

## Quick Start

This project uses Docker to ensure reproducibility and provide a stable environment with GPU support.

### Prerequisites
*   Docker installed.
*   NVIDIA Drivers & NVIDIA Container Toolkit (required for GPU acceleration).

### Setup & Launch

1.  **Navigate to the assignment directory:**
    ```bash
    cd 2
    ```

2.  **Build the Docker image:**
    This script automatically detects your user ID to prevent permission issues.
    ```bash
    bash build
    ```

3.  **Run the container:**
    This will start a Jupyter Notebook server and mount the current directory to `/app`.
    ```bash
    bash launch_container
    ```

4.  **Open Jupyter Notebook:**
    Click the link provided in the terminal (usually containing a token) or go to:
    **http://localhost:8883**

5.  **Run the Code:**
    Open `2_serov.ipynb`. The notebook is self-contained. The first cell installs all necessary Python dependencies (`torch`, `transformers`, `scikit-learn`, `seqeval`, etc.). Execute the cells sequentially to reproduce the results.

---

## Approach

The task is formulated as a **Sequence Labeling** problem (Token Classification) using the BIO tagging scheme. The goal is to extract **Objects**, **Aspects**, and **Predicates** from comparative sentences.

### 1. Data Processing
*   **Format:** CoNLL-style (`Word \t Tag`).
*   **Splitting:** The training data was split into **Train (90%)** and **Validation (10%)** to allow for local model evaluation before submitting to CodaLab.
*   **Tokenization:** Used `AutoTokenizer` (RoBERTa) with `add_prefix_space=True` to handle pre-tokenized inputs correctly. Sub-word tokens were aligned with labels (only the first sub-token retains the label; others are ignored via `-100`).

### 2. Baseline Model (LSTM)
A standard Bi-LSTM tagger was implemented to establish a baseline performance.
*   **Architecture:** Embedding Layer (dim=32) -> LSTM (hidden=64) -> Linear Layer.
*   **Optimization:** SGD (lr=0.1), NLLLoss.
*   **Result:** The model learned basic patterns but struggled with complex dependencies.

### 3. Proposed Method (RoBERTa Large)
To achieve State-of-the-Art (SOTA) results, I utilized a large pre-trained Transformer model.
*   **Model:** `roberta-large` (FaceBook AI).
*   **Why RoBERTa:** It optimizes the BERT pre-training approach (removing Next Sentence Prediction, training on more data), making it highly effective for context-heavy NLU tasks like SRL.
*   **Training Config:**
    *   **Epochs:** 8
    *   **Batch Size:** 16
    *   **Learning Rate:** 2e-5
    *   **Precision:** `bf16` (BFloat16) for L40 GPU optimization.
    *   **Metric:** Strict F1-score (via `seqeval`).

---

## Results

The table below summarizes the performance comparison. The RoBERTa model significantly outperforms the LSTM baseline due to its deep contextual understanding.

| Model | Val F1 | Val Accuracy | Dev F1 | Test F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline LSTM** | 0.7641 | 0.9109 | - | - |
| **RoBERTa (Best)** | **0.8858** | **0.9585** | *TBD* | *TBD* |

### Visualization
Training logs and comparisons are saved in the `plots/` directory:
*   `plots/training_loss.png`: Shows convergence over training steps.
*   `plots/validation_f1.png`: Visual comparison of validation F1 scores between LSTM and RoBERTa across epochs.