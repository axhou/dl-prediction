import numpy as np
import torch

from src.train import build_loader, selection_score


def test_loader_shuffle_is_reproducible():
    technical = np.arange(100, dtype="float32").reshape(20, 5)
    fundamental = np.arange(140, dtype="float32").reshape(20, 7)
    labels = np.arange(20) % 3
    first = build_loader(technical, fundamental, labels, 4, True, 17)
    second = build_loader(technical, fundamental, labels, 4, True, 17)
    first_batch = next(iter(first))
    second_batch = next(iter(second))
    assert all(
        torch.equal(left, right) for left, right in zip(first_batch, second_batch)
    )


def test_balanced_composite_penalizes_bucket_gap():
    balanced = {
        "macro_f1": 0.45,
        "bottom_bucket_accuracy": 0.40,
        "top_bucket_accuracy": 0.40,
    }
    imbalanced = {
        "macro_f1": 0.45,
        "bottom_bucket_accuracy": 0.70,
        "top_bucket_accuracy": 0.10,
    }
    assert selection_score(balanced, 1.0, "balanced_composite") > selection_score(
        imbalanced, 1.0, "balanced_composite"
    )
