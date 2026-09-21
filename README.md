# Multi-Hop KGQA – Path Ranking Engine
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

This project is a scalable, out-of-core Question Answering engine over large-scale Knowledge Graphs.

## What Multi-Hop KGQA does

At a high level:

* Resolves exponential state explosion over high-degree hub entities by matching relation keywords
* Eliminates memory saturation during wide-matrix feature generation by streaming chunks
* Mitigates label sparsity during training via bounded negative sampling
* Trains ranking models to score valid target entities across multi-hop reasoning chains

## Repository clone

```bash
git clone https://github.com/fil2405/multi-hop-kgqa-pra.git
cd multi-hop-kgqa-pra
```

## Run the project

### Requirements

* Python 3.10+
* pandas
* pyarrow
* scikit-learn

### Start the stack

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Logical Flow

1. Knowledge Graph and QA splits are loaded and parsed
2. Topic entities in natural language queries are grounded to KG nodes via an Entity Linker
3. A Heuristic Filter prunes the search space by intersecting relation keywords with query tokens
4. Random-walk transition probabilities are computed for valid multi-hop paths
5. Features are written out-of-core to compressed Parquet files in manageable chunks
6. Logistic Regression and SGD models are trained on the extracted features
7. Candidates are ranked and evaluated using Hits@1 and MRR metrics

## Technologies

| Technology | What does Here |
| --- | --- |
| Python | Core language for the pipeline execution |
| PyArrow | Handles out-of-core data streaming and explicit type-casting |
| Parquet | Compressed file format for storing wide-matrix features on disk |
| Pandas | Data manipulation and processing |
| Scikit-learn | Model training and evaluation |

## Benchmarking

For benchmarking purpose, the feature extraction and evaluation scripts can be used:

```bash
python src/pra_pipeline/feature_extractor.py
python src/pra_pipeline/model.py
```

## Results

### 1-hop -- Standard Logistic Regression

* Train Hits@1: 0.9330 | MRR: 0.9569
* Dev Hits@1: 0.9356 | MRR: 0.9589
* Test Hits@1: 0.9338 | MRR: 0.9579

### 2-hop -- Out-of-Core Logistic Regression

* Train Hits@1: 0.9630 | MRR: 0.9681
* Dev Hits@1: 0.9625 | MRR: 0.9675
* Test Hits@1: 0.9639 | MRR: 0.9684

### 3-hop -- Out-of-Core Logistic Regression

* Train Hits@1: 0.9190 | MRR: 0.9393
* Dev Hits@1: 0.8702 | MRR: 0.9059
* Test Hits@1: 0.8675 | MRR: 0.9049
