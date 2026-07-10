import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss

from .data import DATE_COL, ID_COL, TARGET_COL


def top_bottom_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bottom_mask = y_true == 0
    top_mask = y_true == 2
    return {
        "bottom_bucket_accuracy": float(np.mean(y_pred[bottom_mask] == 0)),
        "top_bucket_accuracy": float(np.mean(y_pred[top_mask] == 2)),
    }


def rank_ic(score, target):
    score = np.asarray(score)
    if np.unique(score).size < 2:
        return np.nan
    correlation = spearmanr(score, target, nan_policy="omit").correlation
    return float(correlation) if pd.notna(correlation) else np.nan


def monthly_rank_ic(frame, score_column):
    rows = []
    for month, group in frame.groupby(frame[DATE_COL].dt.to_period("M")):
        if group[score_column].nunique() < 2 or group[TARGET_COL].nunique() < 2:
            continue
        rows.append(
            {
                "month": month.to_timestamp(),
                "rank_ic": rank_ic(group[score_column], group[TARGET_COL]),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def top_minus_bottom_spread(frame, score_column, quantile=0.30):
    rows = []
    previous_weights = None
    for month, group in frame.groupby(frame[DATE_COL].dt.to_period("M")):
        if len(group) < 20 or group[score_column].nunique() < 2:
            continue
        low = group[score_column].quantile(quantile)
        high = group[score_column].quantile(1.0 - quantile)
        top_return = group.loc[group[score_column] >= high, TARGET_COL].mean()
        bottom_return = group.loc[group[score_column] <= low, TARGET_COL].mean()
        top_ids = group.loc[group[score_column] >= high, ID_COL].tolist()
        bottom_ids = group.loc[group[score_column] <= low, ID_COL].tolist()
        current_weights = {
            identifier: 1.0 / len(top_ids) for identifier in top_ids
        }
        for identifier in bottom_ids:
            current_weights[identifier] = current_weights.get(identifier, 0.0) - (
                1.0 / len(bottom_ids)
            )
        if previous_weights is None:
            turnover = np.nan
        else:
            identifiers = set(previous_weights) | set(current_weights)
            turnover = 0.5 * sum(
                abs(current_weights.get(identifier, 0.0) - previous_weights.get(identifier, 0.0))
                for identifier in identifiers
            )
        rows.append(
            {
                "month": month.to_timestamp(),
                "top_return": top_return,
                "bottom_return": bottom_return,
                "top_minus_bottom": top_return - bottom_return,
                "turnover": turnover,
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows)


def portfolio_statistics(spread, periods_per_year=12, return_scale=100.0):
    if spread.empty:
        return {
            "long_short_sharpe": np.nan,
            "average_turnover": np.nan,
            "max_drawdown": np.nan,
        }
    returns = spread["top_minus_bottom"].dropna() / return_scale
    volatility = returns.std(ddof=1)
    sharpe = (
        np.sqrt(periods_per_year) * returns.mean() / volatility
        if len(returns) > 1 and volatility > 0
        else np.nan
    )
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "long_short_sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "average_turnover": float(spread["turnover"].mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
    }


def evaluate_probabilities(model_name, split_name, frame, labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    ordinal_score = probabilities[:, 2] - probabilities[:, 0]
    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "cross_entropy": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "rank_ic": rank_ic(ordinal_score, frame[TARGET_COL]),
        "rank_ic_top_probability": rank_ic(probabilities[:, 2], frame[TARGET_COL]),
    }
    metrics.update(top_bottom_accuracy(labels, predictions))

    scored = frame[[DATE_COL, ID_COL, TARGET_COL, "label"]].copy()
    score_column = f"{model_name}_score"
    scored[score_column] = ordinal_score
    monthly = monthly_rank_ic(scored, score_column)
    spread = top_minus_bottom_spread(scored, score_column)
    metrics["average_monthly_rank_ic"] = (
        float(monthly["rank_ic"].mean()) if len(monthly) else np.nan
    )
    metrics["positive_ic_month_fraction"] = (
        float((monthly["rank_ic"] > 0).mean()) if len(monthly) else np.nan
    )
    metrics["average_top_minus_bottom_spread"] = (
        float(spread["top_minus_bottom"].mean()) if len(spread) else np.nan
    )
    metrics.update(portfolio_statistics(spread))
    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return metrics, report, monthly, spread, predictions
