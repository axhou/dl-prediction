from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DATE_COL = "date"
ID_COL = "id"
TARGET_COL = "target"

TECHNICAL_CANDIDATES = [
    "beta",
    "betasq",
    "chmom",
    "dolvol",
    "idiovol",
    "indmom",
    "mom1m",
    "mom6m",
    "mom12m",
    "mom36m",
    "pricedelay",
    "turn",
    "aeavol",
    "baspread",
    "ill",
    "maxret",
    "retvol",
    "std_dolvol",
    "std_turn",
    "zerotrade",
]


@dataclass
class PreparedData:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    x_train_technical: np.ndarray
    x_train_fundamental: np.ndarray
    y_train: np.ndarray
    x_validation_technical: np.ndarray
    x_validation_fundamental: np.ndarray
    y_validation: np.ndarray
    x_test_technical: np.ndarray
    x_test_fundamental: np.ndarray
    y_test: np.ndarray
    metadata: dict


def make_label(values, lower_cutoff, upper_cutoff):
    labels = np.where(
        values <= lower_cutoff,
        0,
        np.where(values >= upper_cutoff, 2, 1),
    )
    return pd.Series(labels, index=values.index, dtype="int64")


def validate_temporal_splits(train_frame, validation_frame, test_frame):
    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise ValueError(
            "Temporal split is empty: "
            f"train={len(train_frame)}, validation={len(validation_frame)}, "
            f"test={len(test_frame)}"
        )
    if train_frame[DATE_COL].max() >= validation_frame[DATE_COL].min():
        raise ValueError("Train and validation dates overlap.")
    if validation_frame[DATE_COL].max() >= test_frame[DATE_COL].min():
        raise ValueError("Validation and test dates overlap.")


def _safe_path(path):
    return str(Path(path).expanduser().resolve()).replace("'", "''")


def _get_cutoffs(connection, parquet_path):
    dates = connection.execute(
        f"""
        SELECT DISTINCT CAST("{DATE_COL}" AS DATE) AS split_date
        FROM read_parquet('{parquet_path}')
        WHERE "{TARGET_COL}" IS NOT NULL
        ORDER BY split_date
        """
    ).df()
    unique_dates = pd.to_datetime(dates["split_date"]).to_numpy()
    if len(unique_dates) < 3:
        raise ValueError("At least three distinct dates are required.")
    train_cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.70)])
    validation_cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.85)])
    return train_cutoff, validation_cutoff


def _load_frame(connection, parquet_path, columns, n_rows, sample_seed):
    column_sql = ", ".join(f'"{column}"' for column in columns)
    train_cutoff, validation_cutoff = _get_cutoffs(connection, parquet_path)

    if n_rows and n_rows > 0:
        requested = [int(n_rows * 0.70), int(n_rows * 0.15)]
        requested.append(n_rows - sum(requested))
        conditions = [
            f'CAST("{DATE_COL}" AS DATE) <= DATE \'{train_cutoff.date()}\'',
            (
                f'CAST("{DATE_COL}" AS DATE) > DATE \'{train_cutoff.date()}\' '
                f'AND CAST("{DATE_COL}" AS DATE) <= DATE \'{validation_cutoff.date()}\''
            ),
            f'CAST("{DATE_COL}" AS DATE) > DATE \'{validation_cutoff.date()}\'',
        ]
        parts = []
        for quota, condition, seed in zip(
            requested,
            conditions,
            [sample_seed, sample_seed + 1, sample_seed + 2],
        ):
            parts.append(
                connection.execute(
                    f"""
                    SELECT *
                    FROM (
                        SELECT {column_sql}
                        FROM read_parquet('{parquet_path}')
                        WHERE "{TARGET_COL}" IS NOT NULL AND {condition}
                    )
                    USING SAMPLE reservoir({quota} ROWS) REPEATABLE({seed})
                    """
                ).df()
            )
        frame = pd.concat(parts, ignore_index=True)
        strategy = "time_stratified_70_15_15"
    else:
        frame = connection.execute(
            f"""
            SELECT {column_sql}
            FROM read_parquet('{parquet_path}')
            WHERE "{TARGET_COL}" IS NOT NULL
            """
        ).df()
        requested = None
        strategy = "full_data"

    return frame, train_cutoff, validation_cutoff, requested, strategy


def load_and_preprocess(data_path, n_rows=0, sample_seed=42):
    parquet_path = _safe_path(data_path)
    connection = duckdb.connect()
    schema = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    ).df()
    columns = schema["column_name"].tolist()
    feature_columns = [
        column
        for column in columns
        if column not in {DATE_COL, ID_COL, TARGET_COL, "__index_level_0__"}
    ]
    technical_columns = [
        column for column in TECHNICAL_CANDIDATES if column in feature_columns
    ]
    fundamental_columns = [
        column for column in feature_columns if column not in technical_columns
    ]
    if not technical_columns or not fundamental_columns:
        raise ValueError("Both technical and fundamental feature groups are required.")

    selected_columns = (
        [DATE_COL, ID_COL, TARGET_COL]
        + technical_columns
        + fundamental_columns
    )
    frame, train_cutoff, validation_cutoff, requested, strategy = _load_frame(
        connection,
        parquet_path,
        selected_columns,
        n_rows,
        sample_seed,
    )
    connection.close()

    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])
    for column in technical_columns + fundamental_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    float_columns = frame.select_dtypes(include=["float64"]).columns
    frame[float_columns] = frame[float_columns].astype("float32")
    frame = frame.sort_values(DATE_COL).reset_index(drop=True)

    train_frame = frame[frame[DATE_COL] <= train_cutoff].copy()
    validation_frame = frame[
        (frame[DATE_COL] > train_cutoff)
        & (frame[DATE_COL] <= validation_cutoff)
    ].copy()
    test_frame = frame[frame[DATE_COL] > validation_cutoff].copy()
    validate_temporal_splits(train_frame, validation_frame, test_frame)

    lower_cutoff = train_frame[TARGET_COL].quantile(0.30)
    upper_cutoff = train_frame[TARGET_COL].quantile(0.70)
    for split_frame in (train_frame, validation_frame, test_frame):
        split_frame["label"] = make_label(
            split_frame[TARGET_COL], lower_cutoff, upper_cutoff
        )

    technical_imputer = SimpleImputer(strategy="median")
    fundamental_imputer = SimpleImputer(strategy="median")
    technical_scaler = StandardScaler()
    fundamental_scaler = StandardScaler()

    x_train_technical = technical_scaler.fit_transform(
        technical_imputer.fit_transform(train_frame[technical_columns])
    )
    x_train_fundamental = fundamental_scaler.fit_transform(
        fundamental_imputer.fit_transform(train_frame[fundamental_columns])
    )

    def transform(split_frame):
        technical = technical_scaler.transform(
            technical_imputer.transform(split_frame[technical_columns])
        )
        fundamental = fundamental_scaler.transform(
            fundamental_imputer.transform(split_frame[fundamental_columns])
        )
        return technical.astype("float32"), fundamental.astype("float32")

    x_validation_technical, x_validation_fundamental = transform(validation_frame)
    x_test_technical, x_test_fundamental = transform(test_frame)

    metadata = {
        "data_path": str(Path(data_path)),
        "n_rows_loaded": int(len(frame)),
        "sampling_strategy": strategy,
        "sample_seed": sample_seed,
        "requested_split_rows": requested,
        "technical_columns": technical_columns,
        "fundamental_columns": fundamental_columns,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_cutoff": str(train_cutoff.date()),
        "validation_cutoff": str(validation_cutoff.date()),
        "lower_target_cutoff": float(lower_cutoff),
        "upper_target_cutoff": float(upper_cutoff),
    }

    return PreparedData(
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        x_train_technical=x_train_technical.astype("float32"),
        x_train_fundamental=x_train_fundamental.astype("float32"),
        y_train=train_frame["label"].to_numpy(),
        x_validation_technical=x_validation_technical,
        x_validation_fundamental=x_validation_fundamental,
        y_validation=validation_frame["label"].to_numpy(),
        x_test_technical=x_test_technical,
        x_test_fundamental=x_test_fundamental,
        y_test=test_frame["label"].to_numpy(),
        metadata=metadata,
    )
