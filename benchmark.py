"""Reproducible DeepWalk and GCN benchmarks for the local LINQS citation data."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn

DATA = Path(__file__).resolve().parent / "dataset"
SEEDS = (42, 43, 44)


def load_dataset(name):
    stem = name.lower()
    folder = DATA / stem
    papers = pd.read_csv(folder / f"{stem}.content", sep="\t", header=None, dtype={0: str})
    citations = pd.read_csv(folder / f"{stem}.cites", sep="\t", header=None, dtype=str)
    node_index = {paper_id: i for i, paper_id in enumerate(papers.iloc[:, 0])}
    features = papers.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    classes, labels = np.unique(papers.iloc[:, -1], return_inverse=True)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(labels)))
    graph.add_edges_from(
        (node_index[u], node_index[v])
        for u, v in citations.itertuples(index=False, name=None)
        if u != v and u in node_index and v in node_index
    )
    return graph, features, labels, classes


def stratified_split(labels, seed):
    development, test = train_test_split(
        np.arange(len(labels)), test_size=0.35, stratify=labels, random_state=seed,
    )
    train, val = train_test_split(
        development, test_size=0.15 / 0.65, stratify=labels[development], random_state=seed,
    )
    return train, val, test


def deepwalk_embeddings(graph, seed):
    """Uniform random walks and skip-gram with negative sampling."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_nodes = len(graph)
    neighbors = [np.array(sorted(graph[node]), dtype=np.int32) for node in range(n_nodes)]
    degrees = np.array([len(adjacent) for adjacent in neighbors])
    connected = np.flatnonzero(degrees)
    walks = np.empty((5 * len(connected), 20), dtype=np.int32)
    walks[:, 0] = np.tile(connected, 5)
    for walk in walks:
        for step in range(1, walk.size):
            adjacent = neighbors[walk[step - 1]]
            walk[step] = adjacent[rng.integers(len(adjacent))]

    centers, contexts = [], []
    for offset in range(1, 6):
        left = walks[:, :-offset].ravel()
        right = walks[:, offset:].ravel()
        centers.extend((left, right))
        contexts.extend((right, left))
    centers = np.concatenate(centers)
    contexts = np.concatenate(contexts)
    frequencies = np.bincount(walks.ravel(), minlength=n_nodes)
    negative_weights = torch.tensor(frequencies ** 0.75, dtype=torch.float32)

    source = nn.Embedding(n_nodes, 64)
    context = nn.Embedding(n_nodes, 64)
    nn.init.uniform_(source.weight, -0.5 / 64, 0.5 / 64)
    nn.init.zeros_(context.weight)
    optimizer = torch.optim.Adam([source.weight, context.weight], lr=0.01)

    for _ in range(3):
        for batch in np.array_split(rng.permutation(len(centers)), np.ceil(len(centers) / 4096).astype(int)):
            center = source(torch.from_numpy(centers[batch]))
            positive = context(torch.from_numpy(contexts[batch]))
            negative_ids = torch.multinomial(
                negative_weights, len(batch) * 5, replacement=True,
            ).reshape(len(batch), 5)
            negatives = context(negative_ids)
            positive_scores = (center * positive).sum(dim=1)
            negative_scores = torch.bmm(negatives, center.unsqueeze(2)).squeeze(2)
            loss = -(F.logsigmoid(positive_scores) + F.logsigmoid(-negative_scores).sum(dim=1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    embeddings = F.normalize(source.weight.detach(), dim=1).numpy()
    embeddings[degrees == 0] = 0
    return embeddings


class GCN(nn.Module):
    """Two graph convolution layers with symmetric adjacency normalization."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.hidden = nn.Linear(n_features, 32, bias=False)
        self.output = nn.Linear(32, n_classes, bias=False)

    def forward(self, features, adjacency):
        features = F.dropout(features, p=0.5, training=self.training)
        hidden = torch.sparse.mm(adjacency, self.hidden(features)).relu()
        hidden = F.dropout(hidden, p=0.5, training=self.training)
        return torch.sparse.mm(adjacency, self.output(hidden))


def gcn_probabilities(graph, features, labels, n_classes, train, val, test, seed):
    torch.manual_seed(seed)
    adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float32, format="csr")
    adjacency = adjacency + sp.eye(len(labels), dtype=np.float32, format="csr")
    scale = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel() ** -0.5)
    normalized = (scale @ adjacency @ scale).tocoo()
    adjacency = torch.sparse_coo_tensor(
        np.stack([normalized.row, normalized.col]), normalized.data, normalized.shape,
    ).coalesce()
    row_sum = features.sum(axis=1, keepdims=True)
    features = torch.from_numpy(features / np.maximum(row_sum, 1))
    targets = torch.from_numpy(labels)
    model = GCN(features.shape[1], n_classes)
    optimizer = torch.optim.Adam([
        {"params": model.hidden.parameters(), "weight_decay": 5e-4},
        {"params": model.output.parameters()},
    ], lr=0.01)
    best_loss = float("inf")
    best_epoch = 0
    best_state = {}

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency)
        F.cross_entropy(logits[train], targets[train]).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(features, adjacency)[val], targets[val]).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: weight.detach().clone() for name, weight in model.state_dict().items()}
        if epoch - best_epoch >= 30:
            break

    model.load_state_dict(best_state)
    with torch.no_grad():
        probabilities = model(features, adjacency)[test].softmax(dim=1).numpy()
    return probabilities, best_epoch


def evaluate(labels, probabilities, n_classes):
    assert probabilities.shape == (len(labels), n_classes)
    assert np.allclose(probabilities.sum(axis=1), 1)
    return {
        "Accuracy": accuracy_score(labels, probabilities.argmax(axis=1)),
        "ROC AUC": roc_auc_score(
            labels, probabilities, labels=np.arange(n_classes), multi_class="ovr", average="macro",
        ),
    }


def run_benchmarks(progress=print):
    rows = []
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for name in ("Cora", "CiteSeer"):
            graph, features, labels, classes = load_dataset(name)
            for seed in SEEDS:
                train, val, test = stratified_split(labels, seed)
                progress(f"{name} seed {seed}: training DeepWalk and GCN")
                embeddings = deepwalk_embeddings(graph, seed)
                classifier = LogisticRegression(C=100, max_iter=1000)
                classifier.fit(embeddings[train], labels[train])
                assert np.array_equal(classifier.classes_, np.arange(len(classes)))
                deepwalk_probs = classifier.predict_proba(embeddings[test])
                gcn_probs, best_epoch = gcn_probabilities(
                    graph, features, labels, len(classes), train, val, test, seed,
                )
                for model, probabilities in (("DeepWalk", deepwalk_probs), ("GCN", gcn_probs)):
                    rows.append({
                        "Dataset": name, "Seed": seed, "Model": model,
                        "Train": len(train), "Validation": len(val), "Test": len(test),
                        "GCN checkpoint": best_epoch if model == "GCN" else pd.NA,
                        **evaluate(labels[test], probabilities, len(classes)),
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_benchmarks()
    print(results.to_string(index=False, float_format="{:.4f}".format))
    print(results.groupby(["Dataset", "Model"])[["Accuracy", "ROC AUC"]].agg(["mean", "std"]).round(4))
