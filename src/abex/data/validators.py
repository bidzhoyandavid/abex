"""Basic data-quality checks on a raw experiment dataframe."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    n_rows: int
    dtypes: dict[str, str]
    n_nulls: dict[str, int]
    n_duplicates: int
    group_cardinality: dict[str, int]
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


def validate(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    id_col: str | None = None,
) -> ValidationReport:
    """Run structural checks needed before any stats are computed.

    Raises no exceptions on data issues — issues are collected into the
    report so the caller (human or agent) decides how to proceed. Type and
    argument-shape problems (missing columns, wrong types) are still raised
    immediately, since no meaningful report can be built without them.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment dataframe to validate.
    group_col : str
        Name of the column holding the experiment group/variant label.
    metric_col : str
        Name of the column holding the metric values.
    id_col : str or None, optional
        Name of a unique-entity id column (e.g. user id), used to detect
        duplicated entities instead of fully duplicated rows. Default is None.

    Returns
    -------
    ValidationReport
        Dataclass with row/dtype/null/duplicate/cardinality summaries and a
        list of human-readable `issues`. `is_clean` is True iff `issues` is empty.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or `group_col`/`metric_col`/`id_col`
        is not a str (or None for `id_col`).
    KeyError
        If `group_col`, `metric_col`, or `id_col` is not a column of `df`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(group_col, str):
        raise TypeError(f"group_col must be a str, got {type(group_col).__name__}")
    if not isinstance(metric_col, str):
        raise TypeError(f"metric_col must be a str, got {type(metric_col).__name__}")
    if id_col is not None and not isinstance(id_col, str):
        raise TypeError(f"id_col must be a str or None, got {type(id_col).__name__}")

    for col in (group_col, metric_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found in dataframe")

    issues: list[str] = []

    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
    n_nulls = {col: int(df[col].isna().sum()) for col in df.columns}

    if n_nulls[metric_col] > 0:
        issues.append(f"{n_nulls[metric_col]} null values in metric column {metric_col!r}")
    if n_nulls[group_col] > 0:
        issues.append(f"{n_nulls[group_col]} null values in group column {group_col!r}")

    if id_col is not None:
        if id_col not in df.columns:
            raise KeyError(f"column {id_col!r} not found in dataframe")
        n_duplicates = int(df.duplicated(subset=[id_col]).sum())
        if n_duplicates > 0:
            issues.append(f"{n_duplicates} duplicated rows by id column {id_col!r}")
    else:
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            issues.append(f"{n_duplicates} fully duplicated rows")

    group_cardinality = df[group_col].value_counts(dropna=False).to_dict()
    group_cardinality = {str(k): int(v) for k, v in group_cardinality.items()}
    if len(group_cardinality) < 2:
        issues.append(f"group column {group_col!r} has fewer than 2 distinct values")

    return ValidationReport(
        n_rows=len(df),
        dtypes=dtypes,
        n_nulls=n_nulls,
        n_duplicates=n_duplicates,
        group_cardinality=group_cardinality,
        issues=issues,
    )
