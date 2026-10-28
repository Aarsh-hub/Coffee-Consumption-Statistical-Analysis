import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import statistics_backend as sb

KAGGLE_URL = "https://www.kaggle.com/datasets/datalab351/great-american-coffee-taste-test"
TIDY_URL = ("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/"
            "data/2024/2024-05-14/coffee_survey.csv")
COFFEE_LABELS = {"a": "Coffee A", "b": "Coffee B", "c": "Coffee C", "d": "Coffee D"}

st.set_page_config(page_title="Coffee Consumption Statistical Analysis", page_icon="☕", layout="wide")
st.title("☕ Coffee Consumption Statistical Analysis")
st.caption("Based on the Great American Coffee Taste Test survey (~4,000 respondents, 2023)")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner="Loading dataset...")
def load_default():
    local = ["coffee_survey.csv"] + sorted(glob.glob("data/*.csv")) + sorted(glob.glob("*survey*.csv"))
    for path in local:
        if os.path.exists(path):
            return pd.read_csv(path), f"local file: {path}"
    return pd.read_csv(TIDY_URL), "TidyTuesday mirror (same survey)"


st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload coffee survey CSV (optional)", type=["csv"])
try:
    if uploaded is not None:
        raw, source = pd.read_csv(uploaded), "uploaded file"
    else:
        raw, source = load_default()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.markdown(f"Download it from [Kaggle]({KAGGLE_URL}) and either upload it in the sidebar "
                "or save it next to `app.py` as `coffee_survey.csv`.")
    st.stop()

df, report = sb.clean_survey(raw)
st.sidebar.success(f"{report['rows_clean']:,} survey responses loaded")
st.sidebar.markdown(f"[Dataset on Kaggle]({KAGGLE_URL})")

NUM = sb.numeric_columns(df)
CAT = sb.categorical_columns(df)
if len(df) < 50:
    st.error("The dataset has too few rows to analyse.")
    st.stop()

tabs = st.tabs(["Web Scraping", "Descriptive & Graphs", "Distributions", "Sampling & CLT",
                "Confidence & Tests", "Chi-Square", "Regression", "Classification", "Correlation & ANOVA"])
(t_scrape, t_desc, t_dist, t_samp, t_ci, t_chi, t_reg, t_clf, t_bonus) = tabs

# ------------------------------------------------------------ P1 web scraping
with t_scrape:
    st.subheader("Web Scraping")
    st.write("Scrapes a coffee-related HTML table from Wikipedia, converts it to a DataFrame and cleans it "
             "(rename columns, remove duplicates, handle missing values, convert numbers).")
    url = st.text_input("Page URL", "https://en.wikipedia.org/wiki/List_of_countries_by_coffee_production")
    idx = st.number_input("Table index on the page", min_value=0, value=0, step=1)
    if st.button("Scrape table"):
        try:
            raw_tbl, n_tables = sb.scrape_wikipedia_table(url, int(idx))
            st.session_state["scraped"] = raw_tbl
            st.caption(f"Found {n_tables} wikitable(s) on the page.")
        except Exception as error:
            st.error(f"Scraping failed: {error}")
    if "scraped" in st.session_state:
        raw_tbl = st.session_state["scraped"]
        st.write("**Raw scraped table (first 10 rows)**")
        st.dataframe(raw_tbl.head(10))
        st.write(f"Shape: {raw_tbl.shape}")
        c1, c2 = st.columns(2)
        ccol = c1.selectbox("Country column", raw_tbl.columns, index=0)
        vcol = c2.selectbox("Numeric column", raw_tbl.columns, index=min(1, len(raw_tbl.columns) - 1))
        try:
            clean_tbl, info = sb.clean_scraped_table(raw_tbl, ccol, vcol)
            st.write(f"Duplicates removed: {info['duplicates_removed']} | "
                     f"Rows with missing/non-numeric values removed: {info['missing_removed']}")
            st.dataframe(clean_tbl.head(10))
            st.download_button("Download cleaned CSV", clean_tbl.to_csv(index=False).encode(),
                               "coffee_production.csv", "text/csv")
            st.write("**Descriptive statistics**")
            st.dataframe(pd.Series(sb.descriptive_statistics(clean_tbl["Production"])).to_frame("Value"))
            top = clean_tbl.nlargest(10, "Production")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top["Country"][::-1], top["Production"][::-1])
            ax.set_xlabel("Production (tonnes)")
            ax.set_title("Top 10 producers")
            show(fig)
        except Exception as error:
            st.warning(f"Could not clean the selected columns: {error}")

# ---------------------------------------------------- P2 descriptive & graphs
with t_desc:
    st.subheader("Descriptive Statistics and Graphs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows after cleaning", f"{report['rows_clean']:,}")
    c2.metric("Duplicates removed", report["duplicates_removed"])
    c3.metric("Columns", df.shape[1])
    with st.expander("Missing values (top 10 columns)"):
        st.dataframe(report["missing_top"].to_frame("missing"))
    with st.expander("Dataset preview"):
        st.dataframe(df.head(20))
        st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "cleaned_coffee_survey.csv")

    var = st.selectbox("Numeric variable", NUM, index=NUM.index("cups_num") if "cups_num" in NUM else 0)
    st.dataframe(pd.Series(sb.descriptive_statistics(df[var])).to_frame("Value"))
    with st.expander("pandas describe()"):
        st.dataframe(df[NUM].describe())

    x = df[var].dropna()
    g1, g2 = st.columns(2)
    with g1:
        st.write("**Histogram**")
        fig, ax = plt.subplots()
        ax.hist(x, bins=min(20, max(5, x.nunique())), edgecolor="black")
        ax.set_xlabel(var)
        ax.set_ylabel("Respondents")
        show(fig)
    with g2:
        st.write("**Box plot (outliers)**")
        fig, ax = plt.subplots()
        ax.boxplot(x, vert=False)
        ax.set_xlabel(var)
        show(fig)
    cat = st.selectbox("Categorical variable", CAT, index=CAT.index("favorite") if "favorite" in CAT else 0)
    counts = df[cat].value_counts().head(10)
    g3, g4 = st.columns(2)
    with g3:
        st.write("**Bar chart**")
        fig, ax = plt.subplots()
        ax.barh(counts.index.astype(str)[::-1], counts.values[::-1])
        ax.set_xlabel("Respondents")
        show(fig)
    with g4:
        st.write("**Pie chart**")
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", textprops={"fontsize": 7})
        show(fig)
    g5, g6 = st.columns(2)
    with g5:
        st.write("**Scatter plot**")
        sx = st.selectbox("X", NUM, index=0, key="sx")
        sy = st.selectbox("Y", NUM, index=min(1, len(NUM) - 1), key="sy")
        pair = df[[sx, sy]].dropna()
        fig, ax = plt.subplots()
        ax.scatter(pair[sx] + np.random.default_rng(1).normal(0, 0.05, len(pair)),
                   pair[sy] + np.random.default_rng(2).normal(0, 0.05, len(pair)), alpha=0.3, s=12)
        ax.set_xlabel(sx)
        ax.set_ylabel(sy)
        show(fig)
    with g6:
        st.write("**Line chart: mean by age group**")
        if "age_group" in df.columns:
            lv = st.selectbox("Value", NUM, index=NUM.index(var), key="lv")
            trend = df.groupby("age_group", observed=True)[lv].mean()
            fig, ax = plt.subplots()
            ax.plot(trend.index.astype(str), trend.values, marker="o")
            ax.set_ylabel(f"Mean {lv}")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
            show(fig)

# --------------------------------------------------------- P3 distributions
with t_dist:
    st.subheader("Probability Distributions")
    if "cups_num" in df.columns:
        st.write("#### Poisson: cups of coffee per day")
        lam, dispersion, table = sb.fit_poisson(df["cups_num"])
        st.write(f"Estimated λ = **{lam:.3f}**. Variance/mean ratio = **{dispersion:.2f}** "
                 "(a Poisson variable has ratio ≈ 1).")
        fig, ax = plt.subplots()
        w = 0.4
        ax.bar(table["cups"] - w / 2, table["observed"], w, label="Observed")
        ax.bar(table["cups"] + w / 2, table["poisson_expected"], w, label="Poisson(λ)")
        ax.set_xlabel("Cups per day (rounded; 'less than 1' counted as 0)")
        ax.set_ylabel("Proportion")
        ax.legend()
        show(fig)
    if "prefer_overall" in df.columns:
        st.write("#### Binomial: respondents preferring a coffee")
        labels = sorted(df["prefer_overall"].dropna().unique())
        pick = st.selectbox("Coffee", labels, index=len(labels) - 1)
        p = float((df["prefer_overall"].dropna() == pick).mean())
        n = st.slider("Number of respondents n", 5, 100, 20)
        k = st.slider("k (number preferring it)", 0, n, min(n, int(round(n * p))))
        bt = sb.binomial_table(n, p)
        st.write(f"p = {p:.3f} (share of respondents). P(X = {k}) = **{bt.loc[k, 'pmf']:.4f}**, "
                 f"P(X ≤ {k}) = **{bt.loc[k, 'cdf']:.4f}**")
        fig, ax = plt.subplots()
        ax.bar(bt["k"], bt["pmf"], color="tab:red")
        ax.axvline(k, linestyle="--", color="black")
        ax.set_xlabel("k")
        ax.set_ylabel("P(X = k)")
        show(fig)
    if "avg_preference" in df.columns:
        st.write("#### Normal: average taste-test preference per respondent")
        mu, sigma, sp = sb.fit_normal(df["avg_preference"])
        fig, ax = plt.subplots()
        ax.hist(df["avg_preference"].dropna(), bins=25, density=True, edgecolor="black")
        xs = np.linspace(df["avg_preference"].min(), df["avg_preference"].max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), linewidth=2, label=f"Normal({mu:.2f}, {sigma:.2f})")
        ax.set_xlabel("Average preference rating (1-5)")
        ax.legend()
        show(fig)
        st.caption(f"Shapiro-Wilk p = {sp:.4g}. A small p suggests the data are not exactly normal.")
    st.caption("Uniform distribution: see the sampling tab, where random numbers are drawn uniformly for sampling.")

# ------------------------------------------------------- P4 sampling and CLT
with t_samp:
    st.subheader("Sampling Methods and the Central Limit Theorem")
    var_s = st.selectbox("Variable", NUM, index=NUM.index("avg_preference") if "avg_preference" in NUM else 0,
                         key="samp_var")
    method = st.selectbox("Sampling method",
                          ["Simple random", "Systematic", "Stratified (proportional)", "Cluster"])
    size = st.slider("Sample size", 20, min(1000, len(df)), 200)
    strat = None
    if method in ("Stratified (proportional)", "Cluster"):
        default = CAT.index("age") if "age" in CAT else 0
        strat = st.selectbox("Stratum / cluster column", CAT, index=default)
    replace = st.checkbox("With replacement (simple random only)", value=False) if method == "Simple random" else False
    try:
        sample = sb.draw_sample(df, method, size, strata_col=strat, replace=replace)
        pop_mean, s_mean = df[var_s].mean(), sample[var_s].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Population mean", f"{pop_mean:.3f}")
        c2.metric("Sample mean", f"{s_mean:.3f}")
        c3.metric("Relative error", f"{abs(s_mean - pop_mean) / abs(pop_mean) * 100:.2f}%")
        st.caption(f"Sample size drawn: {len(sample)}")
        fig, ax = plt.subplots()
        ax.hist(df[var_s].dropna(), bins=20, alpha=0.5, density=True, label="Population")
        ax.hist(sample[var_s].dropna(), bins=20, alpha=0.5, density=True, label="Sample")
        ax.set_xlabel(var_s)
        ax.legend()
        show(fig)
    except Exception as error:
        st.warning(str(error))

    st.write("#### Central Limit Theorem")
    n_clt = st.slider("Sample size per draw", 1, 200, 30)
    means = sb.clt_simulation(df[var_s], sample_size=n_clt, repetitions=2000)
    fig, ax = plt.subplots()
    ax.hist(means, bins=30, density=True, edgecolor="black", label="Sample means")
    se = df[var_s].std(ddof=1) / np.sqrt(n_clt)
    xs = np.linspace(means.min(), means.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, df[var_s].mean(), se), linewidth=2, label="Normal from CLT")
    ax.axvline(df[var_s].mean(), linestyle="--", color="black", label="Population mean")
    ax.set_xlabel(f"Mean of {var_s}")
    ax.legend()
    show(fig)
    st.caption(f"SD of sample means = {means.std(ddof=1):.4f}; theoretical σ/√n = {se:.4f}.")

# ------------------------------------------------------------ P5 CI and tests
with t_ci:
    st.subheader("Confidence Intervals and Hypothesis Tests")
    var_c = st.selectbox("Variable", NUM, index=NUM.index("avg_preference") if "avg_preference" in NUM else 0,
                         key="ci_var")
    conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{v:.0%}")
    lo, hi, tcrit, se = sb.confidence_interval_t(df[var_c], conf)
    blo, bhi = sb.bootstrap_ci(df[var_c], conf)
    st.write(f"t-based {conf:.0%} CI: **{lo:.3f} to {hi:.3f}** (t* = {tcrit:.3f}, SE = {se:.4f})")
    st.write(f"Bootstrap {conf:.0%} CI: **{blo:.3f} to {bhi:.3f}**")
    fig, ax = plt.subplots(figsize=(8, 2.5))
    m = df[var_c].mean()
    ax.errorbar(m, 1, xerr=[[m - lo], [hi - m]], fmt="o", capsize=8)
    ax.axvline(m, linestyle="--")
    ax.set_yticks([])
    ax.set_xlabel(var_c)
    show(fig)

    mu0 = st.number_input("Hypothesized mean μ₀", value=float(round(df[var_c].mean(), 1)), step=0.1)
    alpha = 0.05
    tt = sb.t_test(df[var_c], mu0)
    st.write(f"**One-sample t-test** (H₀: μ = {mu0}): t = {tt['t_statistic']:.4f}, "
             f"df = {tt['df']}, p = {tt['p_value']:.5f}")
    (st.success if tt["p_value"] < alpha else st.info)(
        "Reject H₀ at 5%." if tt["p_value"] < alpha else "Fail to reject H₀ at 5%.")
    sigma = st.number_input("Assumed population σ for the z-test", min_value=0.01,
                            value=float(round(df[var_c].std(ddof=1), 2)), step=0.1)
    zt = sb.z_test(df[var_c], mu0, sigma)
    st.write(f"**One-sample z-test**: z = {zt['z_statistic']:.4f}, p = {zt['p_value']:.5f}")
    (st.success if zt["p_value"] < alpha else st.info)(
        "Reject H₀ at 5%." if zt["p_value"] < alpha else "Fail to reject H₀ at 5%.")
    xs = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots()
    ax.plot(xs, stats.norm.pdf(xs))
    for crit in (-1.96, 1.96):
        ax.axvline(crit, linestyle="--", color="gray")
    ax.axvline(float(np.clip(zt["z_statistic"], -4, 4)), color="red", label=f"z = {zt['z_statistic']:.2f}")
    ax.legend()
    show(fig)

# ------------------------------------------------------------ P6 chi-square
with t_chi:
    st.subheader("Chi-Square Test of Independence")
    if len(CAT) < 2:
        st.warning("Need at least two categorical columns.")
    else:
        a = st.selectbox("Variable 1", CAT, index=CAT.index("age") if "age" in CAT else 0)
        rest = [c for c in CAT if c != a]
        b = st.selectbox("Variable 2", rest, index=rest.index("prefer_overall") if "prefer_overall" in rest else 0)
        try:
            res = sb.chi_square_test(df.dropna(subset=[a, b]), a, b)
            st.write("**Contingency table**")
            st.dataframe(res["table"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("χ² statistic", f"{res['chi2']:.3f}")
            c2.metric("p-value", f"{res['p_value']:.5f}")
            c3.metric("Degrees of freedom", res["dof"])
            c4.metric("Cramér's V", f"{res['cramers_v']:.3f}")
            if res["p_value"] <= 0.05:
                st.success(f"Reject H₀: {a} and {b} are associated.")
            else:
                st.info(f"Fail to reject H₀: no evidence that {a} and {b} are related.")
            if res["low_expected_share"] > 0.2:
                st.warning(f"{res['low_expected_share']:.0%} of expected counts are below 5, so the chi-square "
                           "approximation may be unreliable. Pick variables with fewer categories.")
            with st.expander("Expected frequencies"):
                st.dataframe(res["expected"].round(2))
        except Exception as error:
            st.warning(str(error))

# -------------------------------------------------------------- P7 regression
with t_reg:
    st.subheader("Regression Models and Prediction")
    coffee = st.selectbox("Predict preference for", list(COFFEE_LABELS), index=3,
                          format_func=lambda k: COFFEE_LABELS[k])
    try:
        bundle = sb.train_regressors(df, coffee)
        st.caption(f"Target: {bundle['target']} | features: {', '.join(bundle['features'])} | "
                   f"{bundle['n_rows']:,} complete rows | 80/20 train-test split")
        st.dataframe(bundle["results"].round(4))
        best = bundle["results"].iloc[0]["Model"]
        st.write(f"Best model by R²: **{best}**")
        if bundle["results"]["R2"].max() < 0.3:
            st.info("R² is low: taste ratings depend on many things not in the data, so this is expected.")
        fig, ax = plt.subplots()
        ax.scatter(bundle["y_test"], bundle["predictions"][best], alpha=0.3, s=12)
        ax.plot([1, 5], [1, 5], "r--")
        ax.set_xlabel("Actual rating")
        ax.set_ylabel("Predicted rating")
        show(fig)
        st.write("Standardised linear coefficients")
        st.dataframe(bundle["coefficients"].round(4).to_frame())

        st.write("#### Predict a rating")
        chosen = st.selectbox("Model", list(bundle["models"]), index=list(bundle["models"]).index(best))
        ranges = {"expertise": (1, 10, 1), "cups_num": (0, 5, 1), "age_mid": (16, 70, 1)}
        cols = st.columns(len(bundle["features"]))
        values = {}
        for col, feat in zip(cols, bundle["features"]):
            lo_, hi_, step = ranges.get(feat, (1, 5, 1))
            default = float(np.clip(bundle["defaults"][feat], lo_, hi_))
            values[feat] = col.slider(feat, float(lo_), float(hi_), default, float(step))
        pred = sb.predict_one(bundle, chosen, values)
        st.metric("Predicted preference (1-5)", f"{float(np.clip(pred, 1, 5)):.2f}")
    except Exception as error:
        st.warning(str(error))

# ---------------------------------------------------------- P8 classification
with t_clf:
    st.subheader("Classification Models")
    target = st.radio("Target to predict", ["prefer_overall", "heavy_drinker"], horizontal=True,
                      format_func=lambda t: "Preferred coffee (A/B/C/D)" if t == "prefer_overall"
                      else "Heavy drinker (3+ cups/day)")
    if st.button("Train all five models"):
        try:
            with st.spinner("Training..."):
                clf = sb.train_classifiers(df, target)
            st.session_state["clf"] = (target, clf)
        except Exception as error:
            st.warning(str(error))
    if "clf" in st.session_state and st.session_state["clf"][0] == target:
        clf = st.session_state["clf"][1]
        st.caption(f"{clf['n_rows']:,} complete rows | 80/20 stratified split | "
                   f"majority-class baseline accuracy = {clf['majority_baseline']:.3f}")
        st.dataframe(clf["results"].round(4))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(clf["results"]["Model"], clf["results"]["Accuracy"])
        ax.axhline(clf["majority_baseline"], linestyle="--", color="red", label="Majority baseline")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=20)
        show(fig)
        model_name = st.selectbox("Confusion matrix for", list(clf["matrices"]))
        cm = clf["matrices"][model_name]
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(clf["labels"])))
        ax.set_yticks(range(len(clf["labels"])))
        ax.set_xticklabels(clf["labels"])
        ax.set_yticklabels(clf["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        show(fig)
        if target == "prefer_overall":
            st.caption("Note: the preferred coffee is largely determined by the preference ratings used as "
                       "features, so high accuracy here is partly by construction.")

# ------------------------------------------------------------------- bonus
with t_bonus:
    st.subheader("Correlation and ANOVA")
    c1, c2 = st.columns(2)
    xcol = c1.selectbox("X variable", NUM, index=0, key="px")
    ycol = c2.selectbox("Y variable", NUM, index=min(1, len(NUM) - 1), key="py")
    try:
        pr = sb.pearson_correlation(df, xcol, ycol)
        st.write(f"Pearson r = **{pr['r']:.3f}**, p = {pr['p_value']:.5f}, n = {pr['n']}")
        st.caption("Correlation measures linear association, not causation.")
    except Exception as error:
        st.warning(str(error))
    vcol = st.selectbox("Numeric outcome", NUM, index=NUM.index("avg_preference") if "avg_preference" in NUM else 0,
                        key="av")
    gcol = st.selectbox("Grouping variable", CAT, index=CAT.index("age") if "age" in CAT else 0, key="ag")
    try:
        an = sb.anova_by_group(df, vcol, gcol)
        st.write(f"F = **{an['f_statistic']:.3f}**, p = {an['p_value']:.5f} ({an['groups']} groups)")
    except Exception as error:
        st.warning(str(error))

st.divider()
st.caption(f"Data: Great American Coffee Taste Test (James Hoffmann and Cometeer, 2023). "
           f"Kaggle: {KAGGLE_URL}. Survey participants are self-selected, so results describe this sample, "
           "not all Americans.")
