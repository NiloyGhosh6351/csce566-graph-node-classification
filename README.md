# Citation Network Classification on Cora and CiteSeer

CSCE 566 final project, Problem 4. This project compares DeepWalk and a two-layer graph convolutional network (GCN) for predicting paper topics in the Cora and CiteSeer citation networks. DeepWalk learns node embeddings from citation links and uses logistic regression for classification. The GCN uses citation links and binary word features.

## Project files

| File | Purpose |
| --- | --- |
| [benchmark.py](benchmark.py) | Loads the datasets, trains both models, and prints test scores by seed and in aggregate |
| [cora_citeseer_benchmarks.ipynb](cora_citeseer_benchmarks.ipynb) | Documents the evaluation protocol and contains the benchmark tables and comparison chart |
| [cora_citeseer_eda.ipynb](cora_citeseer_eda.ipynb) | Explores topic balance, word features, and graph structure |

## Setup

From the repository root, install the dependencies with Python 3.12:

```sh
python -m pip install numpy pandas scipy networkx torch scikit-learn threadpoolctl matplotlib jupyterlab
```

Download the original [Cora archive](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) and [CiteSeer archive](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz) from [LINQS](https://linqs.org/datasets/). Save both archives in the repository root, then run:

```sh
mkdir dataset
tar -xzf cora.tgz -C dataset
tar -xzf citeseer.tgz -C dataset
```

The code expects this layout:

```text
dataset/
  cora/
    cora.content
    cora.cites
  citeseer/
    citeseer.content
    citeseer.cites
```

## Run the experiments

```sh
python benchmark.py
```

For the notebooks, run `python -m jupyterlab` from the repository root, open either notebook, and run all cells.

Both models use the same stratified splits for seeds 42, 43, and 44, with approximately 50% training, 15% validation, and 35% test. The GCN checkpoint is selected using validation loss. Both models have access to the full citation graph during training, and test labels are used only for evaluation. ROC AUC is computed from class probabilities using macro one-vs-rest averaging.

## Test results

Values are mean ± sample standard deviation across the three runs. Accuracy standard deviations are in percentage points.

| Dataset | Method | Accuracy (%) | ROC AUC |
| --- | --- | ---: | ---: |
| Cora | DeepWalk | 76.79 ± 1.01 | 0.9445 ± 0.0047 |
| Cora | GCN | **87.90 ± 1.46** | **0.9825 ± 0.0026** |
| CiteSeer | DeepWalk | 55.40 ± 1.74 | 0.8076 ± 0.0129 |
| CiteSeer | GCN | **77.30 ± 0.84** | **0.9404 ± 0.0031** |

## References

- Perozzi, Al-Rfou, and Skiena (2014). [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652).
- Kipf and Welling (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
