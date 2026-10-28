"""
Backend for the Great American Coffee Taste Test dashboard (SI Lab project).

Practical map
  P1  Web scraping ............ scrape_wikipedia_table, clean_scraped_table
  P2  Descriptive stats ....... descriptive_statistics
  P3  Distributions ........... fit_poisson, binomial_table, fit_normal
  P4  Sampling + CLT .......... draw_sample, clt_simulation
  P5  CI, z-test, t-test ...... confidence_interval_t, bootstrap_ci, z_test, t_test
  P6  Chi-square .............. chi_square_test
  P7  Regression .............. train_regressors, predict_one
  P8  Classification .......... train_classifiers
  Bonus: pearson_correlation, anova_by_group
"""
import re

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

COFFEES = ["a", "b", "c", "d"]
CUPS_MAP = {"less than 1": 0.5, "1": 1, "2": 2, "3": 3, "4": 4, "more than 4": 5}


# ----------------------------------------------------------------------------
# Cleaning helpers
# ----------------------------------------------------------------------------
def _numbers(text):
    return [float(x) for x in re.findall(r"\d+\.?\d*", str(text).replace(",", ""))]


def _age_midpoint(text):
    """'25-34 years old' -> 29.5, '<18 years old' -> 16, '>65 years old' -> 70."""
    if pd.isna(text):
        return np.nan
    nums = _numbers(text)
    if not nums:
        return np.nan
    t = str(text)
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if "<" in t or "under" in t.lower():
        return nums[0] - 2
    return nums[0] + 5


def _spend_midpoint(text):
    """'$20-$40' -> 30, '<$20' -> 10, '>$100' -> 120."""
    if pd.isna(text):
        return np.nan
    nums = _numbers(text)
    if not nums:
        return np.nan
    t = str(text)
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if "<" in t or "under" in t.lower():
        return nums[0] / 2
    return nums[0] * 1.2


def clean_survey(raw):
    """Clean the raw survey; returns (clean_df, report_dict)."""
    df = raw.copy()
    report = {"rows_raw": len(df)}
    df.columns = [c.strip() for c in df.columns]

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()

    if "cups" in df.columns:
        low = df["cups"].astype("string").str.lower()
        df["cups_num"] = low.map(CUPS_MAP)
        df["cups_num"] = df["cups_num"].fillna(pd.to_numeric(low, errors="coerce")).astype(float)
        df["heavy_drinker"] = np.where(df["cups_num"].isna(), np.nan, (df["cups_num"] >= 3).astype(float))
    if "age" in df.columns:
        df["age_mid"] = df["age"].map(_age_midpoint).astype(float)
        order = (df.dropna(subset=["age_mid"]).groupby("age")["age_mid"].first().sort_values().index.tolist())
        df["age_group"] = pd.Categorical(df["age"], categories=order, ordered=True)
    if "total_spend" in df.columns:
        df["spend_mid"] = df["total_spend"].map(_spend_midpoint).astype(float)
    if "expertise" in df.columns:
        df["expertise"] = pd.to_numeric(df["expertise"], errors="coerce")

    pref_cols = []
    for l in COFFEES:
        for m in ["bitterness", "acidity", "personal_preference"]:
            col = f"coffee_{l}_{m}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        pc = f"coffee_{l}_personal_preference"
        if pc in df.columns:
            pref_cols.append(pc)
    if pref_cols:
        df["avg_preference"] = df[pref_cols].mean(axis=1)

    missing = df.isna().sum()
    report["missing_top"] = missing[missing > 0].sort_values(ascending=False).head(10)
    report["rows_clean"] = len(df)
    return df.reset_index(drop=True), report


def numeric_columns(df):
    skip = {"heavy_drinker"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def categorical_columns(df, max_levels=15):
    out = []
    for c in df.columns:
        if c in ("submission_id",) or str(c).endswith("_notes"):
            continue
        if df[c].dtype == "object" or str(df[c].dtype) in ("string", "category"):
            if 2 <= df[c].nunique() <= max_levels:
                out.append(c)
    return out


# ----------------------------------------------------------------------------
# P2  Descriptive statistics
# ----------------------------------------------------------------------------
def descriptive_statistics(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 2:
        raise ValueError("Need at least two numeric observations.")
    q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
    modes = x.mode()
    return {
        "count": int(x.count()), "mean": float(x.mean()), "median": float(x.median()),
        "mode": float(modes.iloc[0]) if len(modes) else np.nan,
        "variance": float(x.var(ddof=1)), "std_dev": float(x.std(ddof=1)),
        "minimum": float(x.min()), "maximum": float(x.max()),
        "Q1": float(q1), "Q2 (median)": float(q2), "Q3": float(q3), "IQR": float(q3 - q1),
        "skewness": float(stats.skew(x)), "kurtosis": float(stats.kurtosis(x)),
    }


# ----------------------------------------------------------------------------
# P3  Distributions
# ----------------------------------------------------------------------------
def fit_poisson(counts):
    """Fit Poisson to non-negative integer counts; returns lambda and observed/expected table."""
    x = np.round(pd.to_numeric(counts, errors="coerce").dropna()).astype(int)
    x = x[x >= 0]
    lam = float(x.mean())
    values = np.arange(0, int(x.max()) + 1)
    observed = np.array([(x == v).mean() for v in values])
    expected = stats.poisson.pmf(values, lam)
    table = pd.DataFrame({"cups": values, "observed": observed, "poisson_expected": expected})
    return lam, float(x.var(ddof=1) / lam) if lam > 0 else np.nan, table


def binomial_table(n, p):
    k = np.arange(0, n + 1)
    return pd.DataFrame({"k": k, "pmf": stats.binom.pmf(k, n, p), "cdf": stats.binom.cdf(k, n, p)})


def fit_normal(series):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    sample = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
    shapiro_p = float(stats.shapiro(sample).pvalue)
    return mu, sigma, shapiro_p


# ----------------------------------------------------------------------------
# P4  Sampling and CLT
# ----------------------------------------------------------------------------
def draw_sample(df, method, n, strata_col=None, replace=False, seed=42, n_clusters=2):
    rng = np.random.default_rng(seed)
    n = int(min(n, len(df))) if not replace else int(n)
    if method == "Simple random":
        return df.sample(n=n, replace=replace, random_state=seed)
    if method == "Systematic":
        step = max(len(df) // n, 1)
        start = int(rng.integers(0, step))
        return df.iloc[start::step].head(n)
    if method == "Stratified (proportional)":
        if strata_col is None or strata_col not in df.columns:
            raise ValueError("Pick a stratification column.")
        frac = n / len(df)
        parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed)
                 for _, g in df.dropna(subset=[strata_col]).groupby(strata_col, observed=True)]
        return pd.concat(parts)
    if method == "Cluster":
        if strata_col is None or strata_col not in df.columns:
            raise ValueError("Pick a cluster column.")
        groups = df[strata_col].dropna().unique()
        chosen = rng.choice(groups, size=min(n_clusters, len(groups)), replace=False)
        return df[df[strata_col].isin(chosen)]
    raise ValueError("Unknown sampling method.")


def clt_simulation(series, sample_size=30, repetitions=1000, seed=42):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(x) < 2:
        raise ValueError("Need at least two observations.")
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(x, size=sample_size, replace=True).mean() for _ in range(repetitions)])


# ----------------------------------------------------------------------------
# P5  Confidence interval, z-test, t-test
# ----------------------------------------------------------------------------
def confidence_interval_t(series, confidence=0.95):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(x)
    mean, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean - t_crit * se), float(mean + t_crit * se), float(t_crit), float(se)


def bootstrap_ci(series, confidence=0.95, repetitions=5000, seed=42):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(repetitions)])
    a = 1 - confidence
    lo, hi = np.quantile(means, [a / 2, 1 - a / 2])
    return float(lo), float(hi)


def z_test(series, mu0, sigma):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    z = (x.mean() - mu0) / (sigma / np.sqrt(len(x)))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z_statistic": float(z), "p_value": float(p)}


def t_test(series, mu0):
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    r = stats.ttest_1samp(x, popmean=mu0)
    return {"t_statistic": float(r.statistic), "p_value": float(r.pvalue), "df": len(x) - 1}


# ----------------------------------------------------------------------------
# P6  Chi-square test of independence
# ----------------------------------------------------------------------------
def chi_square_test(df, col1, col2):
    table = pd.crosstab(df[col1], df[col2])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both variables need at least two categories.")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan
    exp_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    return {"table": table, "expected": exp_df, "chi2": float(chi2), "p_value": float(p),
            "dof": int(dof), "cramers_v": cramers_v, "low_expected_share": float((expected < 5).mean())}


# ----------------------------------------------------------------------------
# P7  Regression
# ----------------------------------------------------------------------------
def regression_features(df, coffee):
    cols = [f"coffee_{coffee}_bitterness", f"coffee_{coffee}_acidity", "expertise", "cups_num", "age_mid"]
    return [c for c in cols if c in df.columns]


def train_regressors(df, coffee="d", test_size=0.2, seed=42):
    target = f"coffee_{coffee}_personal_preference"
    features = regression_features(df, coffee)
    if target not in df.columns or len(features) < 2:
        raise ValueError("Required rating columns are missing from the dataset.")
    data = df[features + [target]].dropna()
    if len(data) < 50:
        raise ValueError("Not enough complete rows for regression.")
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target],
                                                        test_size=test_size, random_state=seed)
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso Regression": make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
    }
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        rows.append([name, mean_absolute_error(y_test, y_pred), mse, float(np.sqrt(mse)), r2_score(y_test, y_pred)])
    results = pd.DataFrame(rows, columns=["Model", "MAE", "MSE", "RMSE", "R2"]).sort_values("R2", ascending=False)
    lin = models["Linear Regression"]
    coefs = pd.Series(lin[-1].coef_, index=features, name="standardised_coefficient")
    return {"results": results, "models": models, "features": features, "target": target,
            "y_test": y_test.to_numpy(), "predictions": preds, "coefficients": coefs,
            "defaults": data[features].median().to_dict(), "n_rows": len(data)}


def predict_one(bundle, model_name, values):
    row = pd.DataFrame([{f: values[f] for f in bundle["features"]}])
    return float(bundle["models"][model_name].predict(row)[0])


# ----------------------------------------------------------------------------
# P8  Classification
# ----------------------------------------------------------------------------
def classification_setup(df, target):
    rating_feats = [c for c in df.columns if re.fullmatch(r"coffee_[abcd]_(bitterness|acidity|personal_preference)", c)]
    if target == "prefer_overall":
        feats = rating_feats + [c for c in ["expertise", "cups_num", "age_mid"] if c in df.columns]
    elif target == "heavy_drinker":
        feats = rating_feats + [c for c in ["expertise", "age_mid", "spend_mid"] if c in df.columns]
    else:
        raise ValueError("Unsupported target.")
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in dataset.")
    return feats


def train_classifiers(df, target="prefer_overall", test_size=0.2, seed=42):
    feats = classification_setup(df, target)
    data = df[feats + [target]].dropna()
    y = data[target].astype(str)
    if y.nunique() < 2 or len(data) < 100:
        raise ValueError("Not enough complete rows or classes for classification.")
    X_train, X_test, y_train, y_test = train_test_split(
        data[feats], y, test_size=test_size, random_state=seed, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC()),
    }
    labels = sorted(y.unique())
    rows, matrices = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append([name, accuracy_score(y_test, pred),
                     precision_score(y_test, pred, average="weighted", zero_division=0),
                     recall_score(y_test, pred, average="weighted", zero_division=0),
                     f1_score(y_test, pred, average="weighted", zero_division=0)])
        matrices[name] = confusion_matrix(y_test, pred, labels=labels)
    results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"]).sort_values(
        "Accuracy", ascending=False)
    return {"results": results, "matrices": matrices, "labels": labels, "features": feats,
            "n_rows": len(data), "majority_baseline": float(y.value_counts(normalize=True).max())}


# ----------------------------------------------------------------------------
# Bonus
# ----------------------------------------------------------------------------
def pearson_correlation(df, x_col, y_col):
    pair = df[[x_col, y_col]].dropna()
    if len(pair) < 3 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        raise ValueError("Need at least 3 pairs and variation in both variables.")
    r = stats.pearsonr(pair[x_col], pair[y_col])
    return {"r": float(r.statistic), "p_value": float(r.pvalue), "n": len(pair)}


def anova_by_group(df, value_col, group_col, min_group=5):
    groups = [g[value_col].dropna().to_numpy() for _, g in df.groupby(group_col, observed=True)]
    groups = [g for g in groups if len(g) >= min_group]
    if len(groups) < 2:
        raise ValueError("Need at least two groups with enough observations.")
    r = stats.f_oneway(*groups)
    return {"f_statistic": float(r.statistic), "p_value": float(r.pvalue), "groups": len(groups)}


# ----------------------------------------------------------------------------
# P1  Web scraping
# ----------------------------------------------------------------------------
def scrape_wikipedia_table(url, table_index=0, timeout=20):
    """Download a page with requests and parse the n-th <table class='wikitable'> with BeautifulSoup."""
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "SI-Lab-Student-Project/1.0 (educational use)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        raise ValueError("No <table class='wikitable'> found on this page.")
    if table_index >= len(tables):
        raise ValueError(f"Page has only {len(tables)} wikitable(s).")
    table = tables[table_index]
    rows = table.find_all("tr")
    header = [re.sub(r"\[.*?\]", "", c.get_text(" ", strip=True)) for c in rows[0].find_all(["th", "td"])]
    data = []
    for tr in rows[1:]:
        cells = [re.sub(r"\[.*?\]", "", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if len(cells) == len(header):
            data.append(cells)
    return pd.DataFrame(data, columns=header), len(tables)


def clean_scraped_table(raw, country_col=None, value_col=None):
    """Rename columns, drop duplicates, convert numbers, handle missing values (Lab-2 Part A)."""
    df = raw.copy()
    country_col = country_col or df.columns[0]
    value_col = value_col or df.columns[1]
    df = df[[country_col, value_col]].rename(columns={country_col: "Country", value_col: "Production"})
    before = len(df)
    df = df.drop_duplicates()
    dups = before - len(df)
    df["Country"] = df["Country"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["Production"] = pd.to_numeric(df["Production"].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    missing = int(df["Production"].isna().sum())
    df = df.dropna(subset=["Production"]).reset_index(drop=True)
    return df, {"duplicates_removed": dups, "missing_removed": missing}
