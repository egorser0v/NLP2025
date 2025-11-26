# Assignment 1: Word Sense Induction (WSI)

This repository contains the solution for the **Word Sense Induction** task (Natural Language Processing course). The goal is to cluster contexts of target words into groups corresponding to their distinct meanings (senses) using distributional semantics.

## Project Structure

*   `1_serov.ipynb`: The main Jupyter Notebook containing the full pipeline (Data Loading, Training, Evaluation, Submission).
*   `russe-wsi-kit/`: External dataset and evaluation kit (cloned via git).
*   `reports/`: Generated figures and tables analyzing the models performance.
*   `results/`: Final submission CSV files and archives.
*   `Dockerfile` & Scripts: Utilities for containerized environment.

## Quick Start

This project uses Docker to ensure reproducibility. Follow these steps to run the solution.

### Prerequisites
*   Docker installed.
*   NVIDIA Drivers & NVIDIA Container Toolkit (for GPU support).

### Setup & Launch

1.  **Navigate to the assignment directory:**
    ```bash
    cd 1
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
    Click the link provided in the terminal or go to:
     **http://localhost:8881**

5.  **Run the Code:**
    Open `1_serov.ipynb`. The first cell installs all necessary Python dependencies (`torch`, `transformers`, `scikit-learn`, etc.). Execute cells sequentially.

---

## Approach

We employed a **"Fine-tune & Cluster"** pipeline:

1.  **Encoder Selection:**
    *   **wiki-wiki:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (Base model sufficient for coarse homonyms).
    *   **bts-rnc & active-dict:** `ai-forever/ruRoberta-large` (Large Russian-specific model required for fine-grained polysemy).
2.  **Fine-Tuning:**
    *   Models were trained using **Contrastive Learning** (`MultipleNegativesRankingLoss`).
    *   Input Format: `WORD ; CONTEXT` to bias attention.
    *   Adaptive pair generation: Chain sampling for large clusters, Full-Mesh for small clusters.
3.  **Clustering:**
    *   **Agglomerative Clustering** with **Ward Linkage**.
    *   Thresholds ($\tau$) optimized via Grid Search on the training set.

## Results

The table below summarizes the best performance (Adjusted Rand Index - ARI) achieved on the datasets.

| Dataset       | Model Architecture           | Threshold ($\tau$) | Practice (ARI) | Test (ARI) |
|:--------------|:-----------------------------|:-------------------|:---------------|:-----------|
| **wiki-wiki** | `paraphrase-mpnet-base-v2`   | 3.33               | **0.9649**     | *TBD*      |
| **bts-rnc**   | `ruRoberta-large`            | 1.71               | **0.6615**     | **0.5928** |
| **active-dict**| `ruRoberta-large`           | 0.77               | **0.5289**     | **0.4290** |

Detailed analysis and threshold plots can be found in the notebook and `reports/figures`.