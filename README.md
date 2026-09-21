# Multi-Hop KGQA: Path Ranking Engine

A scalable, out-of-core Question Answering engine over large-scale Knowledge Graphs using the **Path Ranking Algorithm (PRA)** on the **MetaQA** benchmark (134k+ triples). 

This project addresses the combinatorial path explosion inherent in multi-hop relational reasoning through **question-guided heuristic pruning** and an **out-of-core streaming feature generation pipeline** built with PyArrow and Parquet.

---

## 📌 Key Architectural Highlights

* **Question-Guided Heuristic Path Pruning:** Resolves exponential state explosion over high-degree hub entities (1, 2, and 3 hops) by matching relation keywords directly against query text prior to random-walk probability computation.
* **Out-of-Core Feature Pipeline:** Eliminates memory saturation ($OOM$) during wide-matrix feature generation by streaming $50,000$-row chunks directly to disk with explicit `float32` type-casting, incremental `pyarrow.parquet.ParquetWriter` execution, and deterministic garbage collection.
* **Stratified Negative Sampling:** Mitigates label sparsity during training via bounded negative sampling ($5$ negatives per entity) paired with top-probability candidate ranking for evaluation splits.
* **Model Training & Evaluation:** Trains ranking models to score valid target entities across multi-hop reasoning chains, benchmarked systematically via **Hits@1** and **Mean Reciprocal Rank (MRR)**.

---

## 🏗 Pipeline Architecture
User Query + Topic Entity
│
▼
┌──────────────────┐
│  Entity Linker   │ ──► Ground raw surface text to KG nodes
└────────┬─────────┘
│
▼
┌──────────────────┐
│ Heuristic Filter │ ──► Discard relations absent from query lexical tokens
└────────┬─────────┘
│
▼
┌──────────────────┐
│  Path Traversal  │ ──► Compute random-walk transition probabilities
└────────┬─────────┘
│
▼
┌──────────────────┐
│ Out-of-Core Sink │ ──► Chunk-based memory management & PyArrow Parquet writer
└────────┬─────────┘
│
▼
┌──────────────────┐
│  Ranking Models  │ ──► Evaluation via Hits@1 and MRR
└──────────────────┘


---

## 📂 Project Structure

bash
├── data/
│   ├── entity_linker.py      # Entity linking module for topic entity groundings
│   ├── load_graph.py         # Graph loader & adjacency indexer
│   └── parse_qa.py           # QA parser for MetaQA splits
├── track_a/
│   ├── build_features.py     # Out-of-core feature extraction & Parquet generation
│   ├── path_traversal.py     # Graph traversal and path probability algorithms
│   └── dataset_processed/    # Target directory for partitioned Parquet splits
├── conf.py                   # Global configuration and relation keyword maps
├── requirements.txt          # Python dependencies
└── README.md
🚀 Getting Started
1. Prerequisites & Installation
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

Bash
git clone [https://github.com/fil2405/multi-hop-kgqa-pra.git](https://github.com/fil2405/multi-hop-kgqa-pra.git)
cd multi-hop-kgqa-pra
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Dependencies
pyarrow>=14.0.0

pandas>=2.0.0

scikit-learn>=1.3.0

3. Feature Extraction
Run the out-of-core feature extraction pipeline for 1-hop, 2-hop, or 3-hop settings:

Bash
python track_a/build_features.py
Processed datasets will be streamed incrementally into track_a/dataset_processed/<hop>-hop/ as compressed Parquet files (train_set.parquet, dev_set.parquet, test_set.parquet).

📊 Evaluation Metrics
Candidate answers are scored and ranked against ground-truth sets using:

Hits@1: Fraction of queries where the top-ranked candidate is a correct answer.

MRR (Mean Reciprocal Rank): Average reciprocal rank of the first correct answer across all test instances.
