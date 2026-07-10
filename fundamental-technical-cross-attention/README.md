# Fundamental-Technical Cross-Attention for Stock Return Prediction

This package contains the reproducible Python implementation used for the final deep
learning project. It predicts bottom, middle, and top next-period return buckets from
market-based technical signals and firm-level fundamentals.

## Final Models

- `technical_only_mlp`: technical baseline
- `concat_mlp`: direct feature-concatenation baseline
- `direct_cross_attention`: unrestricted bidirectional attention
- `conservative_direct_cross_attention`: attention enters through a small learned
  residual strength; this is the selected proposed model

Residual, gated residual, fundamental-only, and late-fusion models are retained as
ablations in `src/models.py`.

## Repository Map

- `src/data.py`: licensed parquet loading, chronological splits, labels, imputation,
  and scaling
- `src/models.py`: all neural architectures
- `src/train.py`: deterministic loaders, class weighting, early stopping, and fitting
- `src/metrics.py`: Macro F1, Rank IC, monthly IC, spread, Sharpe, turnover,
  and max drawdown
- `src/experiment.py`: multi-model orchestration and artifact generation
- `scripts/`: command-line entry points
- `tests/`: leakage, reproducibility, model-shape, and metric tests

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Validation Run

```bash
python -m scripts.run_final_experiment \
  --data-path /path/to/90features_USstocks.parquet \
  --output-dir outputs/smoke_test \
  --n-rows 100000 \
  --epochs 2 \
  --patience 1 \
  --no-save-models
```

## Three-Seed 1M Comparison

```bash
python -m scripts.run_final_experiment \
  --data-path /path/to/90features_USstocks.parquet \
  --output-dir outputs/finalists_1m \
  --n-rows 1000000 \
  --seeds 17 42 101
```

## Final Full-Data Test

Run this only after model selection is complete. `--evaluate-test` evaluates the
previously untouched test period.

```bash
python -m scripts.run_final_experiment \
  --data-path /path/to/90features_USstocks.parquet \
  --output-dir outputs/final_full_data \
  --n-rows 0 \
  --seeds 42 \
  --evaluate-test
```

The command writes metrics, seed summaries, monthly Rank IC, return spreads,
classification reports, figures, run metadata, and optional model checkpoints.
