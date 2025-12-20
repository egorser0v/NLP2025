# Assignment 3: Taxonomy Enrichment for Russian

## Project Structure
*   `3_serov.ipynb`: The main Jupyter Notebook containing the full pipeline (Data Loading, POS-aware RuWordNet Patching, Vectorization, Baseline and Advanced Voting Models, Evaluation, Submission).
*   `plots/`: Directory containing generated visualizations:
    *   `data_distribution.png`: Analysis of parent synset distribution in the training data for both nouns and verbs.
    *   `results_detailed.png`: POS-specific performance comparison (Nouns vs. Verbs) between Baseline and Advanced models.
*   `predictions/`: Directory containing `.tsv` files formatted for CodaLab submission (Public/Private test sets for both nouns and verbs).
*   `results/`: Directory containing `model_comparison.csv` with numerical results.
*   `models/`: Directory for pre-trained weights:
    *   `cc.ru.300.vec`: FastText word vectors for Russian.
    *   `vectors/`: Generated vectors for RuWordNet synsets and test words (N/V separated).
*   `taxonomy-enrichment/`: Cloned repository containing the competition data and original baseline scripts.

---

## Quick Start

This project is designed to run in a containerized environment to ensure reproducibility of embedding generation and hierarchical graph traversal.

### Prerequisites
*   Docker installed.
*   NVIDIA Container Toolkit (recommended for SBERT GPU acceleration).

### Setup & Launch

1.  **Navigate to the assignment directory:**
    ```bash
    cd 3
    ```

2.  **Build the Docker image:**
    ```bash
    bash build
    ```

3.  **Run the container:**
    ```bash
    bash launch_container
    ```

4.  **Open Jupyter Notebook:**
    Navigate to **http://localhost:8883** in your browser.

5.  **Run the Code:**
    Execute `3_serov.ipynb`. The notebook automatically handles:
    *   Cloning the competition repository.
    *   Downloading and extracting the 4GB FastText model.
    *   Patching and initializing the RuWordNet SQLite database.

---

## Approach

The task is to associate unseen Russian words (neologisms) with the most appropriate **Hypernyms** (parent synsets) from the RuWordNet hierarchy, covering both **Nouns** and **Verbs**.

### 1. Data Engineering & POS-Aware Preprocessing
*   **RuWordNet Patching:** Resolved a known initialization bug in the `ruwordnet` reader via a monkey-patch to ensure data is correctly inserted into the SQLite database from source XMLs.
*   **POS Isolation:** The system is strictly POS-aware. Noun neologisms are only compared against the Noun taxonomy, and Verb neologisms against the Verb taxonomy, significantly reducing semantic noise.
*   **MWE Handling:** Multi-word expressions (e.g., "ИДТИ ВПЕРЕД") were transformed using underscore substitution (`ИДТИ_ВПЕРЕД`) to comply with Word2Vec parsing standards.

### 2. Baseline Model (FastText + Second-Order)
*   **Method:** Uses static `cc.ru.300.vec` embeddings.
*   **Logic:** Finds the top $k=10$ similar synsets and retrieves their immediate hypernyms.
*   **Performance:** Shows high local validation scores but tends to overfit to direct similarity, performing less effectively on complex private test data.

### 3. Advanced Model (SBERT + Rank-Weighted Hierarchical Voting)
The primary method used for the final submission utilizes contextual embeddings and taxonomic graph weights:
*   **Embeddings:** `paraphrase-multilingual-mpnet-base-v2` (Sentence-Transformers) for superior multi-word and semantic representation.
*   **Voting Logic ($k=50$):** Instead of simple retrieval, the model uses a voting mechanism where the top 50 neighbors vote for their ancestors.
*   **Hierarchical Weighting:** Candidates are weighted by their position in the graph: immediate parents receive a weight of **1.0**, while grandparents (2nd-order) receive **0.5**.
*   **Rank Decay:** To prioritize the most relevant signals, each neighbor's vote is divided by its similarity rank ($1/rank$).

---

## Results

While the baseline showed higher scores on the local validation split, the **Advanced Voting model** demonstrated superior generalization on the **CodaLab Private Test set**, which is the definitive metric for the competition.

### Local Evaluation Summary
| Model | POS | MAP | MRR |
| :--- | :--- | :--- | :--- |
| **Baseline** | Noun | 0.7350 | 0.7394 |
| **Baseline** | Verb | 0.6816 | 0.6921 |
| **Advanced** | Noun | 0.4850 | 0.5138 |
| **Advanced** | Verb | 0.3256 | 0.3833 |

### CodaLab Private Leaderboard Performance
The Advanced model proved more robust on unseen data:
*   **Nouns (Private):** **0.4627** (Advanced) vs. 0.4169 (Baseline)
*   **Verbs (Private):** **0.3347** (Advanced) vs. 0.2998 (Baseline)

### Analysis
*   **Generalization:** The Advanced model's hierarchical voting successfully captures "consensus" ancestors, making it more robust to the semantic shifts found in the private test set.
*   **Hierarchy Depth:** Awarding points to 2nd-order hypernyms (grandparents) allowed the model to maintain high MAP scores even when the direct parent was not the closest semantic neighbor.
*   **POS Impact:** Verb enrichment proved more challenging than noun enrichment (averaging ~13% lower MAP), likely due to the more abstract nature of action hierarchies in RuWordNet.