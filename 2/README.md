# Assignment 2: Semantic Role Labelling

## Project Structure
*   `2_serov.ipynb`: The main Jupyter Notebook containing the full pipeline (Data Loading, Preprocessing, Training Bi-LSTM/RoBERTa/DeBERTa, Ensembling, Evaluation, Submission).
*   `plots/`: Directory containing generated charts, specifically `all_models_loss.png` comparing training convergence.
*   `tables/`: Directory containing metric comparison CSVs (`final_leaderboard.csv`).
*   `submissions/`: Directory containing `.zip` files for CodaLab submission (Dev/Test results).
*   `models/`: Directory where model checkpoints are saved (excluded from version control).
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
    Open `2_serov.ipynb`. The notebook is self-contained. The first cell installs all necessary Python dependencies (including `transformers`, `accelerate`, `sentencepiece`, `seqeval`). Execute the cells sequentially to reproduce the results.

---

## Approach

The task is formulated as a **Sequence Labeling** problem (Token Classification) using the BIO tagging scheme. The goal is to extract **Objects**, **Aspects**, and **Predicates** from comparative sentences.

### 1. Data Processing
*   **Format:** CoNLL-style (`Word \t Tag`).
*   **Splitting:**
    *   For single models: **Train (90%)** / **Validation (10%)**.
    *   For Ensemble: **5-Fold Cross-Validation** (rotating 80/20 splits).
*   **Tokenization:** Used `AutoTokenizer` with `add_prefix_space=True` to handle pre-tokenized inputs correctly. Sub-word tokens were aligned with labels (only the first sub-token retains the label; others are ignored via `-100`).

### 2. Baseline Model (Bi-LSTM)
A Bidirectional LSTM was implemented to establish a baseline.
*   **Architecture:** Embedding Layer -> Bi-LSTM (hidden=128) -> Dropout -> Linear Layer.
*   **Result:** It captured local context better than a standard RNN but lacked the deep semantic understanding required for complex comparative structures.

### 3. Intermediate Model (RoBERTa Large)
A robust Transformer model optimized for context.
*   **Model:** `roberta-large`.
*   **Performance:** Significantly outperformed the LSTM but struggled slightly with the relative positioning of predicates and arguments compared to newer architectures.

### 4. SOTA Method (DeBERTa-V3 + Ensemble)
To achieve maximum performance on the Test set, I utilized DeBERTa-V3 and an ensembling strategy.
*   **Model:** `microsoft/deberta-v3-large`.
*   **Why DeBERTa:** It utilizes **Disentangled Attention** and **Relative Position Embeddings**, which are crucial for Semantic Role Labeling where the relationship/distance between the predicate and the object is defining.
*   **Ensemble Strategy:** I trained 5 separate DeBERTa models using K-Fold Cross-Validation. During inference, the logits (prediction scores) from all 5 models are averaged. This "smoothing" effect reduces variance and eliminates errors caused by data noise in specific splits.

---

## Results

The table below summarizes the performance comparison. The **DeBERTa 5-Fold Ensemble** achieved the highest generalization performance on the blind Test set.

| Model | Val F1 (Best) | Val Accuracy | Dev F1 | Test F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Bi-LSTM)** | 0.8090 | 0.9229 | - | - |
| **RoBERTa-Large** | 0.8836 | 0.9555 | **0.7396** | 0.8609 |
| **DeBERTa-V3-Large** | 0.8873 | **0.9587** | 0.7180 | 0.8744 |
| **DeBERTa (5-Fold Ensemble)** | **0.8897** | 0.9582 | 0.7297 | **0.8788** |

### Analysis
*   **Generalization:** While RoBERTa performed anomalously well on the small Dev set, the **Ensemble** proved superior on the larger, more representative Test set (+1.8% over RoBERTa).
*   **Stability:** The Ensemble method effectively mitigated the variance seen in single-model training runs.

### Visualization
Training logs and comparisons are saved in the `plots/` directory:
*   `plots/all_models_loss.png`: Visual comparison of training loss convergence between Bi-LSTM, RoBERTa, and DeBERTa.
*   `plots/final_comparison_with_ensemble.png`: Bar chart comparing final F1 scores across all approaches.