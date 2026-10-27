import numpy as np
import pandas as pd
from scipy import stats


def clean_data(df):
    df = df.copy()
    if "Cups_Per_Day" not in df.columns:
        raise ValueError("CSV must contain a 'Cups_Per_Day' column.")

    df["Cups_Per_Day"] = pd.to_numeric(df["Cups_Per_Day"], errors="coerce")
    df = df.dropna(subset=["Cups_Per_Day"])
    df = df[df["Cups_Per_Day"] >= 0]

    for col in ["Age", "Money_Per_Week", "Days_Per_Week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def descriptive_statistics(data):
    x = data["Cups_Per_Day"].dropna()
    if len(x) == 0:
        raise ValueError("No valid consumption values.")
    return {
        "count": int(x.count()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "mode": float(x.mode().iloc[0]),
        "variance": float(x.var(ddof=1)) if len(x) > 1 else 0.0,
        "std_dev": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "minimum": float(x.min()),
        "maximum": float(x.max())
    }


def bootstrap_confidence_interval(data, confidence=0.95, repetitions=5000, seed=42):
    x = np.asarray(data["Cups_Per_Day"].dropna(), dtype=float)
    if len(x) < 2:
        raise ValueError("At least two observations are required.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(repetitions)
    ])
    alpha = 1 - confidence
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def clt_simulation(data, sample_size=30, repetitions=1000, seed=42):
    x = np.asarray(data["Cups_Per_Day"].dropna(), dtype=float)
    if len(x) < 2:
        raise ValueError("At least two observations are required.")
    if not 1 <= sample_size <= len(x):
        raise ValueError("Sample size must be between 1 and dataset size.")
    rng = np.random.default_rng(seed)
    return np.array([
        rng.choice(x, size=sample_size, replace=True).mean()
        for _ in range(repetitions)
    ])


def one_sample_t_test(data, hypothesized_mean=2):
    x = data["Cups_Per_Day"].dropna().to_numpy()
    if len(x) < 2:
        raise ValueError("At least two observations are required.")
    result = stats.ttest_1samp(x, popmean=hypothesized_mean)
    return {
        "sample_mean": float(np.mean(x)),
        "hypothesized_mean": float(hypothesized_mean),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue)
    }


def pearson_correlation(data, x_col="Cups_Per_Day", y_col="Money_Per_Week"):
    pair = data[[x_col, y_col]].dropna()
    if len(pair) < 3:
        raise ValueError("At least three complete pairs are required.")
    if pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        raise ValueError("Both variables must have variation.")
    result = stats.pearsonr(pair[x_col], pair[y_col])
    return {"r": float(result.statistic), "p_value": float(result.pvalue), "n": len(pair)}


def group_comparison(data, group_col="Coffee_Type"):
    if group_col not in data.columns:
        raise ValueError(f"Column {group_col} is missing.")
    groups = [
        group["Cups_Per_Day"].dropna().to_numpy()
        for _, group in data.groupby(group_col)
    ]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        raise ValueError("Need at least two groups with two observations each.")
    result = stats.f_oneway(*groups)
    return {
        "f_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "groups_compared": len(groups)
    }
