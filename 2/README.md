# Assignment 2: Semantic Role Labelling

## Project Structure
*   `2_serov.ipynb`: The main Jupyter Notebook containing the full pipeline (Data Loading, Preprocessing, Training Bi-LSTM/RoBERTa/DeBERTa, Optimization, Evaluation, Submission).
*   `plots/`: Directory containing generated charts, specifically `all_models_loss.png` and `final_comparison_grandmaster.png`.
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
    *   For Ensembles: **K-Fold Cross-Validation** (5 or 10 folds).
*   **Tokenization:** Used `AutoTokenizer` with `add_prefix_space=True` to handle pre-tokenized inputs correctly. Sub-word tokens were aligned with labels (only the first sub-token retains the label; others are ignored via `-100`).

### 2. Baseline Model (Bi-LSTM)
A Bidirectional LSTM was implemented to establish a baseline.
*   **Architecture:** Embedding Layer -> Bi-LSTM (hidden=128) -> Dropout -> Linear Layer.
*   **Result:** It captured local context better than a standard RNN but lacked the deep semantic understanding required for complex comparative structures.

### 3. Intermediate Model (RoBERTa Large)
A robust Transformer model optimized for context.
*   **Model:** `roberta-large`.
*   **Performance:** Significantly outperformed the LSTM but struggled slightly with the relative positioning of predicates and arguments compared to newer architectures.

### 4. SOTA Method (DeBERTa-V3 + 5-Fold Ensemble)
To achieve maximum performance on the Test set, I utilized DeBERTa-V3.
*   **Model:** `microsoft/deberta-v3-large`.
*   **Why DeBERTa:** It utilizes **Disentangled Attention** and **Relative Position Embeddings**, which are crucial for Semantic Role Labeling where the relationship/distance between the predicate and the object is defining.
*   **Ensemble Strategy:** Trained 5 separate models. Logits were averaged to reduce variance.

### 5. Final  Strategy (10-Fold + LLRD + Logic Repair)
To reach the top of the leaderboard (Rank #2), I added three advanced optimizations to the DeBERTa pipeline:
*   **10-Fold CV:** Increased folds from 5 to 10 to maximize data coverage.
*   **Layer-wise Learning Rate Decay (LLRD):** The bottom layers (embeddings) were trained with a very low learning rate to preserve pre-trained knowledge, while top layers trained faster. This prevented "catastrophic forgetting."
*   **Logic Repair (Post-Processing):** A custom script fixes invalid tag sequences (e.g., ensuring an `I-Object` is always preceded by a `B-Object`).

---

## Results

The table below summarizes the performance comparison. The **Optimized DeBERTa (10-Fold)** strategy achieved the best generalization.

| Model | Val F1 (Best) | Val Accuracy | Dev F1 | Test F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Bi-LSTM)** | 0.8090 | 0.9229 | - | - |
| **RoBERTa-Large** | 0.8836 | 0.9555 | 0.7396 | 0.8609 |
| **DeBERTa-V3-Large** | 0.8873 | **0.9587** | 0.7180 | 0.8744 |
| **DeBERTa (5-Fold Ensemble)** | **0.8897** | 0.9582 | 0.7297 | 0.8788 |
| **DeBERTa (10-Fold + LLRD)** | 0.8761 | 0.9524 | **0.7423** | **0.8802** |

### Analysis
*   **Generalization:** While RoBERTa performed anomalously well on the small Dev set, the **Final Ensemble** proved superior on the larger, more representative Test set (+1.9% over RoBERTa).
*   **Optimization:** The combination of LLRD and Logic Repair squeezed out the final 0.2% needed to reach 0.88+ F1, fixing syntax errors that raw models often make.
*   **Stability:** The 10-Fold approach ensured that the model didn't overfit to any specific subset of the training data.

### Visualization
Training logs and comparisons are saved in the `plots/` directory:
*   `plots/all_models_loss.png`: Visual comparison of training loss convergence between Bi-LSTM, RoBERTa, and DeBERTa.
*   `plots/final_comparison_grandmaster.png`: Bar chart comparing final F1 scores across all approaches.