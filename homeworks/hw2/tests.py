from collections import defaultdict
import copy
import gc
import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import polars as pl
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


def test_create_masked_tensor(func):
    # --- case 1: 1D indices ---
    data = torch.tensor([1, 2, 3, 4, 5, 6])
    lengths = torch.tensor([2, 3, 1])

    padded, mask = func(data, lengths)

    expected_padded = torch.tensor([
        [1, 2, 0],
        [3, 4, 5],
        [6, 0, 0],
    ])
    expected_mask = torch.tensor([
        [True, True, False],
        [True, True, True],
        [True, False, False],
    ])

    assert torch.equal(padded, expected_padded)
    assert torch.equal(mask, expected_mask)
    assert (padded[~mask] == 0).all()
    assert torch.equal(padded[mask], data)

    # --- case 2: embeddings/features (2D) ---
    data2 = torch.arange(5 * 4).view(5, 4)
    lengths2 = torch.tensor([2, 3])

    padded2, mask2 = func(data2, lengths2)

    assert padded2.shape == (2, 3, 4)
    assert mask2.shape == (2, 3)
    assert (padded2[~mask2] == 0).all()
    assert torch.equal(padded2[mask2], data2)

    print('All good! :)')


def test_yambda_dataset(dataset_cls):
    # synthetic data
    histories = {
        "u1": [1, 2, 3, 4],
        "u2": [10, 11],
        "u3": [100, 101, 102, 103, 104],
    }
    labels = {
        "u1": [999],
        "u3": [888],
    }

    max_seq_len = 2

    # -------- train mode checks --------
    ds_train = dataset_cls(histories=histories, labels={}, is_train=True, max_seq_len=max_seq_len)
    assert len(ds_train) == (3 + 1 + 4)

    s0 = ds_train[0]
    assert set(s0.keys()) == {"uid", "history", "label"}
    assert isinstance(s0["history"]["item_id"], list)
    assert isinstance(s0["history"]["length"], int)
    assert isinstance(s0["label"], int)

    sample_u1_t3 = next(
        s for s in (ds_train[i] for i in range(len(ds_train)))
        if s["uid"] == "u1" and s["label"] == 4
    )
    assert sample_u1_t3["history"]["item_id"] == [2, 3]
    assert sample_u1_t3["history"]["length"] == 2

    # -------- eval mode checks --------
    ds_eval = dataset_cls(histories=histories, labels=labels, is_train=False, max_seq_len=max_seq_len)
    assert len(ds_eval) == 2

    uids = {ds_eval[i]["uid"] for i in range(len(ds_eval))}
    assert uids == {"u1", "u3"}

    se = ds_eval[0]
    assert set(se.keys()) == {"uid", "history"}
    assert isinstance(se["history"]["item_id"], list)
    assert isinstance(se["history"]["length"], int)

    sample_u3 = next(s for s in (ds_eval[i] for i in range(len(ds_eval))) if s["uid"] == "u3")
    assert sample_u3["history"]["item_id"] == [103, 104]
    assert sample_u3["history"]["length"] == 2

    print('All good! :)')


def test_collate_fn(func):
    # --- train batch (with labels) ---
    batch_train = [
        {"uid": 7, "history": {"item_id": [1, 2], "length": 2}, "label": 3},
        {"uid": 8, "history": {"item_id": [10], "length": 1}, "label": 11},
        {"uid": 9, "history": {"item_id": [5, 6, 7], "length": 3}, "label": 8},
    ]
    out = func(batch_train)

    assert set(out.keys()) == {"history", "uid", "label"}
    assert torch.equal(out["history"]["item_id"], torch.tensor([1, 2, 10, 5, 6, 7], dtype=torch.long))
    assert torch.equal(out["history"]["length"], torch.tensor([2, 1, 3], dtype=torch.long))
    assert torch.equal(out["uid"], torch.tensor([7, 8, 9], dtype=torch.long))
    assert torch.equal(out["label"], torch.tensor([3, 11, 8], dtype=torch.long))

    # basic shapes / dtypes
    assert out["history"]["item_id"].dtype == torch.long
    assert out["history"]["length"].dtype == torch.long
    assert out["uid"].dtype == torch.long
    assert out["label"].dtype == torch.long
    assert out["history"]["length"].numel() == out["uid"].numel() == out["label"].numel()

    # --- eval batch (no labels) ---
    batch_eval = [
        {"uid": 7, "history": {"item_id": [1, 2], "length": 2}},
        {"uid": 8, "history": {"item_id": [10], "length": 1}},
    ]
    out2 = func(batch_eval)

    assert set(out2.keys()) == {"history", "uid"}
    assert "label" not in out2
    assert torch.equal(out2["history"]["item_id"], torch.tensor([1, 2, 10], dtype=torch.long))
    assert torch.equal(out2["history"]["length"], torch.tensor([2, 1], dtype=torch.long))
    assert torch.equal(out2["uid"], torch.tensor([7, 8], dtype=torch.long))

    print('All good! :)')


def test_user_encoder(encoder_cls):
    num_items = 10
    emb_dim = 3
    model = encoder_cls(num_items=num_items, embedding_dim=emb_dim)

    weight = torch.arange(num_items * emb_dim, dtype=torch.float32).view(num_items, emb_dim)
    with torch.no_grad():
        model.item_embeddings.weight.copy_(weight)

    item_id = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    lengths = torch.tensor([2, 3, 1], dtype=torch.long)

    out = model({"item_id": item_id, "length": lengths})

    expected = torch.stack([
        weight[[0, 1]].sum(dim=0),
        weight[[2, 3, 4]].sum(dim=0),
        weight[[5]].sum(dim=0),
    ], dim=0)

    assert out.shape == (3, emb_dim)
    assert torch.allclose(out, expected), f"Wrong output:\n{out}\nExpected:\n{expected}"

    print('All good! :)')


def test_two_tower(two_tower_cls):
    class DummyTwoTower(two_tower_cls):
        def __init__(self, num_items: int, embedding_dim: int) -> None:
            super().__init__(num_items=num_items, embedding_dim=embedding_dim)
            self.last_user_repr = None

        def compute_loss(self, user_repr: torch.Tensor, inputs):
            self.last_user_repr = user_repr.detach().clone()
            return user_repr.pow(2).mean()
    
    num_items = 6
    emb_dim = 3
    model = DummyTwoTower(num_items=num_items, embedding_dim=emb_dim)

    weight = torch.arange(num_items * emb_dim, dtype=torch.float32).view(num_items, emb_dim)
    with torch.no_grad():
        model.encoder.item_embeddings.weight.copy_(weight)

    inputs = {
        "uid": torch.tensor([10, 11], dtype=torch.long),
        "history": {
            "item_id": torch.tensor([0, 1, 2], dtype=torch.long),
            "length": torch.tensor([2, 1], dtype=torch.long),
        }
    }

    model.train()
    loss = model(inputs)

    assert isinstance(loss, torch.Tensor) and loss.ndim == 0, "Train forward must return scalar loss"
    assert model.last_user_repr is not None, "compute_loss was not called"

    expected_user_repr = torch.stack([weight[[0, 1]].sum(0), weight[[2]].sum(0)], dim=0)
    assert torch.allclose(model.last_user_repr, expected_user_repr)

    model.eval()
    scores = model(inputs)

    assert scores.shape == (2, num_items), "Eval forward must return (batch_size, num_items) scores"
    expected_scores = expected_user_repr @ weight.T
    assert torch.allclose(scores, expected_scores), "Scores must equal user_repr @ item_matrix.T"

    print('All good! :)')


def test_softmax_model(softmax_cls):
    model = softmax_cls(num_items=4, embedding_dim=2)

    W = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.0],
    ])
    with torch.no_grad():
        model.encoder.item_embeddings.weight.copy_(W)

    user_repr = torch.tensor([
        [1.0, 2.0],
        [0.5, -1.0],
    ])

    inputs = {"label": torch.tensor([2, 1], dtype=torch.long)}

    loss = model.compute_loss(user_repr, inputs)

    logits = user_repr @ W.T
    expected = F.cross_entropy(logits, inputs["label"])

    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert torch.allclose(loss, expected)

    print('All good! :)')


def check_all_metrics_geq(metrics, hitrate, recall, ndcg, coverage):
    assert metrics["recall"] >= recall, "Too low recall value"
    assert metrics["hitrate"] >= hitrate, "Too low hitrate value"
    assert metrics["ndcg"] >= ndcg, "Too low ndcg value"
    assert metrics["coverage"] >= coverage, "Too low coverage value"


def check_bpr_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.17,
        recall=0.046,
        ndcg=0.016,
        coverage=0.22
    )
    print('All good! :)')


def check_bce_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.23,
        recall=0.06,
        ndcg=0.025,
        coverage=0.25
    )
    print('All good! :)')


def check_softmax_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.32,
        recall=0.09,
        ndcg=0.035,
        coverage=0.6
    )
    print('All good! :)')


def check_softmax_uniform_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.32,
        recall=0.09,
        ndcg=0.035,
        coverage=0.6
    )
    print('All good! :)')


def check_softmax_inbatch_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.25,
        recall=0.07,
        ndcg=0.027,
        coverage=0.5
    )
    print('All good! :)')


def check_softmax_inbatch_logq_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.32,
        recall=0.09,
        ndcg=0.035,
        coverage=0.3
    )
    print('All good! :)')
