from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .metrics import top_bottom_accuracy


class TabularDataset(Dataset):
    def __init__(self, technical, fundamental, labels):
        self.technical = torch.as_tensor(technical, dtype=torch.float32)
        self.fundamental = torch.as_tensor(fundamental, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.technical[index], self.fundamental[index], self.labels[index]


def build_loader(
    technical,
    fundamental,
    labels,
    batch_size,
    shuffle,
    random_state,
    pin_memory=False,
):
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return DataLoader(
        TabularDataset(technical, fundamental, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=pin_memory,
    )


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        cross_entropy = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
        )
        probability = torch.exp(-cross_entropy)
        return (((1.0 - probability) ** self.gamma) * cross_entropy).mean()


def make_class_weights(labels, config: TrainingConfig):
    counts = np.bincount(labels, minlength=3).astype("float32")
    base = counts.sum() / (3 * np.maximum(counts, 1))
    weights = base ** config.class_weight_power
    weights /= weights.mean()
    weights *= np.array(
        [
            config.bottom_weight_mult,
            config.middle_weight_mult,
            config.top_weight_mult,
        ],
        dtype="float32",
    )
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def selection_score(metrics, validation_loss, metric_name):
    if metric_name == "validation_loss":
        return -validation_loss
    if metric_name == "macro_f1":
        return metrics["macro_f1"]
    if metric_name == "top_bottom_macro":
        return 0.5 * (
            metrics["bottom_bucket_accuracy"] + metrics["top_bucket_accuracy"]
        )
    if metric_name == "balanced_composite":
        top_accuracy = metrics["top_bucket_accuracy"]
        bottom_accuracy = metrics["bottom_bucket_accuracy"]
        gap = abs(top_accuracy - bottom_accuracy)
        return (
            metrics["macro_f1"]
            + 0.25 * top_accuracy
            + 0.25 * bottom_accuracy
            - 0.15 * gap
        )
    raise ValueError(f"Unknown selection metric: {metric_name}")


def predict_probabilities(model, loader, device):
    model.eval()
    probability_batches = []
    label_batches = []
    with torch.no_grad():
        for technical, fundamental, labels in loader:
            technical = technical.to(device, non_blocking=True)
            fundamental = fundamental.to(device, non_blocking=True)
            logits = model(technical, fundamental)
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            label_batches.append(labels.numpy())
    return np.vstack(probability_batches), np.concatenate(label_batches)


def _loader_metrics(model, loader, device):
    probabilities, labels = predict_probabilities(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
    metrics.update(top_bottom_accuracy(labels, predictions))
    return metrics


def fit_model(model, train_loader, validation_loader, labels, config, device):
    model = model.to(device)
    class_weights = make_class_weights(labels, config).to(device)
    if config.loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif config.loss == "focal":
        criterion = FocalLoss(class_weights, config.focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {config.loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = None
    best_score = -float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for technical, fundamental, target in train_loader:
            technical = technical.to(device, non_blocking=True)
            fundamental = fundamental.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(technical, fundamental)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_loss_sum += loss.item() * len(target)
            train_count += len(target)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for technical, fundamental, target in validation_loader:
                technical = technical.to(device, non_blocking=True)
                fundamental = fundamental.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                loss = criterion(model(technical, fundamental), target)
                validation_loss_sum += loss.item() * len(target)
                validation_count += len(target)

        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        metrics = _loader_metrics(model, validation_loader, device)
        score = selection_score(
            metrics,
            validation_loss,
            config.selection_metric,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "selection_score": score,
                **metrics,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"bottom={metrics['bottom_bucket_accuracy']:.4f} "
            f"top={metrics['top_bucket_accuracy']:.4f}",
            flush=True,
        )

        if score > best_score + 1e-4:
            best_score = score
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), class_weights.detach().cpu().tolist()
