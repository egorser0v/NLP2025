# Assignment 3: Taxonomy Enrichment for Russian

## Project Structure
*   `3_serov.ipynb`: The main Jupyter Notebook containing the full pipeline (Data Loading, RuWordNet Patching, Vectorization, Baseline and Advanced Models, Evaluation, Submission).
*   `plots/`: Directory containing generated visualizations:
    *   `data_distribution.png`: Analysis of parent synset distribution in the training data.
    *   `results.png`: Performance comparison between Baseline and Advanced models.
*   `predictions/`: Directory containing `.tsv` files formatted for CodaLab submission (Public/Private test sets).
*   `results/`: Directory containing `model_comparison.csv` with numerical results.
*   `models/`: Directory for pre-trained weights:
    *   `cc.ru.300.vec`: FastText word vectors for Russian.
    *   `vectors/`: Generated vectors for RuWordNet synsets and test words.
*   `taxonomy-enrichment/`: Cloned repository containing the competition data and original baseline scripts.

---

## Quick Start

This project is designed to run in a containerized environment to ensure reproducibility of embedding generation and graph traversal.

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
    Execute `3_serov.ipynb`. The notebook automatically downloads the 4GB FastText model and initializes the RuWordNet database. 

---

## Approach

The task is to attach "neologisms" (words not currently in the taxonomy) to their most appropriate **Hypernyms** (parent synsets) within RuWordNet.

### 1. Data Engineering & Preprocessing
*   **RuWordNet Patching:** The out-of-the-box `ruwordnet` reader has a known initialization bug. I implemented a monkey-patch for the `__init__` method to ensure the SQLite database is correctly populated from the XML source files before use.
*   **MWE Handling:** Multi-word expressions (e.g., "государственный аппарат") cause errors in standard Word2Vec formats due to space-delimiters. All multi-word keys were transformed using underscore substitution (e.g., `ГОСУДАРСТВЕННЫЙ_АППАРАТ`) for vector storage and retrieval.
*   **Synset Vectorization:** Synset embeddings were calculated by averaging the vectors of all component senses (synonyms) within that synset.

### 2. Baseline Model (FastText + Second-Order)
*   **Method:** Uses static `cc.ru.300.vec` FastText embeddings. 
*   **Logic:** For a given word, find the top $k=10$ most similar existing synsets. Then, retrieve the immediate hypernyms of those synsets.
*   **Limitation:** Static embeddings struggle with polysemy and phrases, and looking only at immediate parents is often too restrictive.

### 3. Advanced Model (SBERT + Graph-Based Ancestors)
The SOTA approach for this task combined contextual embeddings with taxonomic graph features:
*   **Embeddings:** Utilized `paraphrase-multilingual-mpnet-base-v2` from Sentence-Transformers. This model provides superior semantic representation for Russian text compared to static vectors.
*   **Graph Traversal:** Using `networkx`, I built a directed acyclic graph (DAG) of the RuWordNet hierarchy.
*   **Candidate Expansion:** Instead of only looking at direct parents, the model retrieves **Ancestors up to Depth 2**. 
*   **Scoring with Decay:** Candidates are ranked by embedding similarity, but second-order hypernyms receive a **decay factor of 0.7**. This rewards "consensus" parents where multiple similar synsets point to the same grandparent.

---

## Results

Evaluation was performed on the `test_nouns` split using **Mean Average Precision (MAP)** as the primary metric.

### Key Findings
*   **Context over Statics:** Moving to SBERT provided a massive jump in MAP (+19%). SBERT's ability to "understand" the relationship between a phrase and a single word is vital for taxonomy tasks.
*   **Graph Reranking:** Including 2nd-level ancestors helped recover cases where the nearest neighbor in the embedding space was a "cousin" or "sibling" rather than a direct child of the target hypernym.
*   **MWE Precision:** The underscore substitution trick fixed `ValueError` crashes during the vector loading phase and allowed 100% coverage of the test set expressions.

### Visualization
All plots are located in the `plots/` folder:
*   `plots/data_distribution.png`: Shows that most training words have a low number of parent groups, emphasizing the need for high precision in the top-1 ranked results.
*   `plots/results.png`: Visual bar chart comparison of the performance metrics across local test splits.