# Multi-Hop KGQA – Path Ranking Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A scalable, out-of-core Question Answering engine over large-scale Knowledge Graphs.

---

## Engineering Highlights

* **Out-of-Core Processing:** Prevents RAM exhaustion ($O(N)$ memory growth) during large-scale graph traversals.
* **Search Space Pruning:** Mitigates combinatorial path explosion over high-degree nodes.
* **Bounded Negative Sampling:** Alleviates extreme class imbalance and label sparsity.
* **Production-Grade Containerization:** Fully reproducible end-to-end pipeline containerization.

---

## Pipeline Architecture

```text
[ Natural Language Query ]
        │
        ▼
   Entity Linker       ─────────► Grounds question topic entities to KG nodes
        │
        ▼
   Heuristic Filter    ─────────► Intersects query tokens with relation keywords
        │
        ▼
   Path Traversal      ─────────► Computes random-walk transition probabilities (PRA)
        │
        ▼
   Out-of-Core Writer  ─────────► Streams bounded memory chunks to Parquet
        │
        ▼
   Ranker Evaluation   ─────────► Trains Logistic Regression / SGD; scores Hits@1 & MRR

---

## Quickstart

### Option A: Run via Docker (Recommended, Zero Host Dependencies)

Ensure Docker Desktop is running, then clone and execute the entire pipeline with:

```bash
# Clone repository
git clone https://github.com/fil2405/multi-hop-kgqa-pra.git
cd multi-hop-kgqa-pra

# Build image
docker build -t kgqa-pra .

# Run end-to-end pipeline
docker run --rm kgqa-pra
```

To run a specific reasoning depth (e.g., 1-hop, 2-hop, or 3-hop), pass the `HOP` environment variable:

```bash
docker run --rm -e HOP=2 kgqa-pra
```

### Option B: Local Python Environment

```bash
# Clone repository
git clone https://github.com/fil2405/multi-hop-kgqa-pra.git
cd multi-hop-kgqa-pra

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Fetch and extract MetaQA dataset
python setup_data.py

# Extract features and evaluate
python src/pra_pipeline/feature_extractor.py
python src/pra_pipeline/model.py
```

## Tech Stack

| Technology | Role in Architecture |
| :--- | :--- |
| Python 3.11 | Core pipeline execution and orchestration |
| Docker | Isolated, multi-stage runtime for automated pipeline reproduction |
| Apache Arrow / PyArrow | Streaming record batching, schema typing, and out-of-core I/O |
| Apache Parquet | Columnar disk storage format for sparse feature matrices |
| Scikit-learn | Linear model training, ranking logic, and evaluation metrics |
| Pandas / NumPy | Vectorized score aggregation, ranking operations, and profiling |

## System Benchmarks & Results

### Data Engineering & System Profiling

Memory footprint and latency benchmarked across reasoning chain depths:

| Configuration | Extraction Time | Peak RAM (MB) | Inference Latency | Throughput (QPS) |
| :--- | :--- | :--- | :--- | :--- |
| 1-Hop | | | | |
| 2-Hop | | | | |
| 3-Hop | | | | |

> **Memory Stability Note:** *While standard in-memory DataFrame extraction triggers Out-Of-Memory (`OOM`) crashes on 2-hop and 3-hop traversals under constrained RAM environments, the chunked Parquet streaming pipeline guarantees bounded peak memory usage regardless of total dataset size.*

### Task Accuracy: Hits@1 & MRR

Evaluated on the official MetaQA vanilla benchmark:

| Split | Metric | 1-Hop | 2-Hop | 3-Hop |
| :--- | :--- | :--- | :--- | :--- |
| Train | Hits@1 | | | |
| | MRR | | | |
| Dev | Hits@1 | | | |
| | MRR | | | |
| Test | Hits@1 | | | |
| | MRR | | | |

## Repository Structure

```plaintext
├── Dockerfile              # Container definition for reproducible builds
├── .dockerignore           # Build context exclusions
├── .gitattributes          # Repository linguist configuration
├── .gitignore              # Ignore cache, venv, and raw dataset artifacts
├── LICENSE                 # MIT License
├── README.md               # Project documentation
├── requirements.txt        # Minimal runtime dependencies
├── setup_data.py           # Automated MetaQA dataset fetcher and parser
└── src/
    ├── data/               # Graph loader, QA parser, and Entity Linker
    ├── dataset_raw_KG/     # Downloaded Knowledge Graph and raw text splits
    └── pra_pipeline/       # Path traversal, feature extraction, and models
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
