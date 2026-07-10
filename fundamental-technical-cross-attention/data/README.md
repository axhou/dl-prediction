# Data

The CRSP/Compustat-derived parquet file is not distributed in this repository because
it is subject to WRDS licensing restrictions.

Expected columns:

- `date`: observation date
- `id`: stock identifier
- `target`: next-period realized return
- technical and fundamental predictors described in `src/data.py`

Pass the local parquet path with `--data-path` when running an experiment.
