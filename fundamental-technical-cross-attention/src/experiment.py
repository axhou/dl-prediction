import json
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from .config import TrainingConfig
from .data import load_and_preprocess
from .metrics import evaluate_probabilities
from .models import build_model
from .train import build_loader, fit_model, predict_probabilities
from .utils import ensure_output_dirs, select_device, set_seed, write_json


@dataclass(frozen=True)
class ModelRunSpec:
    model_name: str
    config: TrainingConfig
    candidate_name: str | None = None

    @property
    def candidate(self):
        return self.candidate_name or self.model_name


def _plot_metric_summary(metrics, path):
    validation = metrics[metrics["split"] == "validation"]
    plot_columns = ["accuracy", "macro_f1", "rank_ic"]
    axis = validation.set_index("candidate")[plot_columns].plot.bar(
        figsize=(11, 5)
    )
    axis.set_title("Validation Metrics")
    axis.set_xlabel("Model")
    axis.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_history(history, path):
    plt.figure(figsize=(10, 5))
    for (candidate, seed), group in history.groupby(["candidate", "random_state"]):
        plt.plot(
            group["epoch"],
            group["validation_loss"],
            marker="o",
            label=f"{candidate}, seed={seed}",
        )
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _summary_table(metrics):
    columns = [
        "accuracy",
        "macro_f1",
        "cross_entropy",
        "rank_ic",
        "average_monthly_rank_ic",
        "positive_ic_month_fraction",
        "bottom_bucket_accuracy",
        "top_bucket_accuracy",
        "average_top_minus_bottom_spread",
        "long_short_sharpe",
        "average_turnover",
        "max_drawdown",
    ]
    validation = metrics[metrics["split"] == "validation"]
    return validation.groupby("candidate")[columns].agg(["mean", "std"])


def run_model_suite(
    data_path,
    output_dir,
    model_specs,
    n_rows=0,
    sample_seed=42,
    evaluate_test=False,
    save_models=True,
):
    directories = ensure_output_dirs(output_dir)
    device = select_device()
    print(f"Using device: {device}", flush=True)
    prepared = load_and_preprocess(
        data_path,
        n_rows=n_rows,
        sample_seed=sample_seed,
    )
    print(
        f"Rows: train={len(prepared.train_frame):,}, "
        f"validation={len(prepared.validation_frame):,}, "
        f"test={len(prepared.test_frame):,}",
        flush=True,
    )

    metric_rows = []
    histories = []
    reports = {}
    monthly_tables = []
    spread_tables = []
    best_validation = (-float("inf"), None, None)

    for spec in model_specs:
        config = spec.config
        set_seed(config.random_state)
        model = build_model(
            spec.model_name,
            prepared.x_train_technical.shape[1],
            prepared.x_train_fundamental.shape[1],
            config,
        )
        pin_memory = device.type == "cuda"
        train_loader = build_loader(
            prepared.x_train_technical,
            prepared.x_train_fundamental,
            prepared.y_train,
            config.batch_size,
            shuffle=True,
            random_state=config.random_state,
            pin_memory=pin_memory,
        )
        validation_loader = build_loader(
            prepared.x_validation_technical,
            prepared.x_validation_fundamental,
            prepared.y_validation,
            config.batch_size,
            shuffle=False,
            random_state=config.random_state,
            pin_memory=pin_memory,
        )

        print(
            f"\n=== Training {spec.candidate} "
            f"({spec.model_name}, seed={config.random_state}) ===",
            flush=True,
        )
        started = time.time()
        model, history, class_weights = fit_model(
            model,
            train_loader,
            validation_loader,
            prepared.y_train,
            config,
            device,
        )
        validation_probabilities, validation_labels = predict_probabilities(
            model,
            validation_loader,
            device,
        )
        elapsed = time.time() - started
        if not (validation_labels == prepared.y_validation).all():
            raise RuntimeError("Validation loader changed row order.")

        metrics, report, monthly, spread, predictions = evaluate_probabilities(
            spec.model_name,
            "validation",
            prepared.validation_frame,
            prepared.y_validation,
            validation_probabilities,
        )
        metrics.update(
            {
                "candidate": spec.candidate,
                "random_state": config.random_state,
                "fit_inference_seconds": elapsed,
            }
        )
        metric_rows.append(metrics)
        reports[f"{spec.candidate}_seed{config.random_state}_validation"] = report
        history["model"] = spec.model_name
        history["candidate"] = spec.candidate
        history["random_state"] = config.random_state
        histories.append(history)
        monthly["model"] = spec.model_name
        monthly["candidate"] = spec.candidate
        monthly["random_state"] = config.random_state
        monthly["split"] = "validation"
        spread["model"] = spec.model_name
        spread["candidate"] = spec.candidate
        spread["random_state"] = config.random_state
        spread["split"] = "validation"
        monthly_tables.append(monthly)
        spread_tables.append(spread)

        if metrics["macro_f1"] > best_validation[0]:
            best_validation = (metrics["macro_f1"], spec.candidate, predictions)

        if evaluate_test:
            test_loader = build_loader(
                prepared.x_test_technical,
                prepared.x_test_fundamental,
                prepared.y_test,
                config.batch_size,
                shuffle=False,
                random_state=config.random_state,
                pin_memory=pin_memory,
            )
            test_probabilities, test_labels = predict_probabilities(
                model,
                test_loader,
                device,
            )
            if not (test_labels == prepared.y_test).all():
                raise RuntimeError("Test loader changed row order.")
            test_metrics, test_report, test_monthly, test_spread, _ = (
                evaluate_probabilities(
                    spec.model_name,
                    "test",
                    prepared.test_frame,
                    prepared.y_test,
                    test_probabilities,
                )
            )
            test_metrics.update(
                {
                    "candidate": spec.candidate,
                    "random_state": config.random_state,
                    "fit_inference_seconds": elapsed,
                }
            )
            metric_rows.append(test_metrics)
            reports[f"{spec.candidate}_seed{config.random_state}_test"] = test_report
            test_monthly["model"] = spec.model_name
            test_monthly["candidate"] = spec.candidate
            test_monthly["random_state"] = config.random_state
            test_monthly["split"] = "test"
            test_spread["model"] = spec.model_name
            test_spread["candidate"] = spec.candidate
            test_spread["random_state"] = config.random_state
            test_spread["split"] = "test"
            monthly_tables.append(test_monthly)
            spread_tables.append(test_spread)

        if save_models:
            checkpoint = {
                "model_name": spec.model_name,
                "candidate": spec.candidate,
                "config": config.to_dict(),
                "class_weights": class_weights,
                "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "data_metadata": prepared.metadata,
            }
            torch.save(
                checkpoint,
                directories["models"]
                / f"{spec.candidate}_seed{config.random_state}.pt",
            )

    metrics_frame = pd.DataFrame(metric_rows)
    history_frame = pd.concat(histories, ignore_index=True)
    monthly_frame = pd.concat(monthly_tables, ignore_index=True)
    spread_frame = pd.concat(spread_tables, ignore_index=True)
    summary = _summary_table(metrics_frame)

    metrics_frame.to_csv(directories["tables"] / "metrics.csv", index=False)
    history_frame.to_csv(directories["tables"] / "training_history.csv", index=False)
    monthly_frame.to_csv(directories["tables"] / "monthly_rank_ic.csv", index=False)
    spread_frame.to_csv(directories["tables"] / "return_spread.csv", index=False)
    summary.to_csv(directories["tables"] / "validation_seed_summary.csv")
    write_json(directories["tables"] / "classification_reports.json", reports)
    write_json(
        directories["tables"] / "run_metadata.json",
        {
            "data": prepared.metadata,
            "evaluate_test": evaluate_test,
            "device": str(device),
            "models": [
                {
                    "candidate": spec.candidate,
                    "model_name": spec.model_name,
                    "config": spec.config.to_dict(),
                }
                for spec in model_specs
            ],
        },
    )

    _plot_metric_summary(metrics_frame, directories["figures"] / "metrics.png")
    _plot_history(history_frame, directories["figures"] / "validation_loss.png")
    ConfusionMatrixDisplay.from_predictions(
        prepared.y_validation,
        best_validation[2],
        display_labels=["bottom", "middle", "top"],
        cmap="Blues",
        values_format="d",
    )
    plt.title(f"Best Validation Macro F1: {best_validation[1]}")
    plt.tight_layout()
    plt.savefig(directories["figures"] / "best_confusion_matrix.png", dpi=200)
    plt.close()

    print("\nValidation seed summary:")
    print(summary.to_string())
    return metrics_frame, summary
