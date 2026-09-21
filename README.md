# Multi-Hop KGQA – Path Ranking Engine

This project is a scalable, out-of-core Question Answering engine over large-scale Knowledge Graphs[cite: 1].

## What Multi-Hop KGQA does

At a high level:

* Resolves exponential state explosion over high-degree hub entities by matching relation keywords[cite: 1]
* Eliminates memory saturation during wide-matrix feature generation by streaming chunks[cite: 1]
* Mitigates label sparsity during training via bounded negative sampling[cite: 1]
* Trains ranking models to score valid target entities across multi-hop reasoning chains[cite: 1]

## Repository clone

```bash
git clone [https://github.com/fil2405/multi-hop-kgqa-pra.git](https://github.com/fil2405/multi-hop-kgqa-pra.git)
cd multi-hop-kgqa-pra
```

## Run the project

### Requirements

* Python 3.10+[cite: 1]
* pandas[cite: 1]
* pyarrow[cite: 1]
* scikit-learn[cite: 2]

### Start the stack

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Logical Flow

1. Knowledge Graph and QA splits are loaded and parsed[cite: 2]
2. Topic entities in natural language queries are grounded to KG nodes via an Entity Linker[cite: 2]
3. A Heuristic Filter prunes the search space by intersecting relation keywords with query tokens[cite: 2]
4. Random-walk transition probabilities are computed for valid multi-hop paths[cite: 2]
5. Features are written out-of-core to compressed Parquet files in manageable chunks[cite: 2]
6. Logistic Regression and SGD models are trained on the extracted features[cite: 2]
7. Candidates are ranked and evaluated using Hits@1 and MRR metrics[cite: 2]

## Technologies

| Technology | What does Here |
| --- | --- |
| Python | Core language for the pipeline execution |
| PyArrow | Handles out-of-core data streaming and explicit type-casting |
| Parquet | Compressed file format for storing wide-matrix features on disk |
| Pandas | Data manipulation and processing |
| Scikit-learn | Model training and evaluation |

## Benchmarking

For benchmarking purpose, the feature extraction and evaluation scripts can be used:[cite: 3]

```bash
python src/pra_pipeline/feature_extractor.py
python src/pra_pipeline/model.py
```

## Results

### 1-hop (Standard Logistic Regression)

* Train Hits@1: 0.9330 | MRR: 0.9569[cite: 4]
* Dev Hits@1: 0.9356 | MRR: 0.9589[cite: 4]
* Test Hits@1: 0.9338 | MRR: 0.9579[cite: 4]

### 2-hop (Out-of-Core Logistic Regression)

* Train Hits@1: 0.9630 | MRR: 0.9681[cite: 4]
* Dev Hits@1: 0.9625 | MRR: 0.9675[cite: 4]
* Test Hits@1: 0.9639 | MRR: 0.9684[cite: 4]

### 3-hop (Out-of-Core Logistic Regression)

* Train Hits@1: 0.9190 | MRR: 0.9393[cite: 4, 5]
* Dev Hits@1: 0.8702 | MRR: 0.9059[cite: 4, 5]
* Test Hits@1: 0.8675 | MRR: 0.9049[cite: 5]
