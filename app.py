import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statistics_backend import (
    clean_data, descriptive_statistics, bootstrap_confidence_interval,
    clt_simulation, one_sample_t_test, pearson_correlation, group_comparison
)

st.set_page_config(page_title="Coffee Consumption Analyzer", page_icon="☕", layout="wide")
st.title("☕ Coffee Consumption Statistical Analysis")
st.caption("Statistical Inference Lab Project | Synthetic dataset for demonstration")

@st.cache_data
def load_default_data():
    return pd.read_csv("coffee_data.csv")

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload coffee CSV", type=["csv"])
try:
    raw_data = pd.read_csv(uploaded_file) if uploaded_file is not None else load_default_data()
    data = clean_data(raw_data)
except Exception as error:
    st.error(f"Could not load or clean the dataset: {error}")
    st.stop()

if len(data) < 2:
    st.error("Please provide at least two valid observations.")
    st.stop()

st.sidebar.header("Analysis Settings")
hypothesized_mean = st.sidebar.number_input("Hypothesized mean (cups/day)", min_value=0.0, value=2.0, step=0.5)
confidence = st.sidebar.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1,
                                  format_func=lambda x: f"{x:.0%}")

summary = descriptive_statistics(data)
ci_lower, ci_upper = bootstrap_confidence_interval(data, confidence=confidence)

overview, inference, relationships, dataset = st.tabs([
    "Overview", "Statistical Inference", "Relationships", "Dataset"
])

with overview:
    st.subheader("Consumption Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", summary["count"])
    c2.metric("Mean cups/day", f'{summary["mean"]:.2f}')
    c3.metric("Median cups/day", f'{summary["median"]:.2f}')
    c4.metric("Standard deviation", f'{summary["std_dev"]:.2f}')

    left, right = st.columns(2)
    with left:
        st.write("#### Daily consumption distribution")
        fig, ax = plt.subplots()
        bins = np.arange(-0.5, data["Cups_Per_Day"].max() + 1.5, 1)
        ax.hist(data["Cups_Per_Day"], bins=bins, edgecolor="black")
        ax.set_xlabel("Cups per day")
        ax.set_ylabel("Participants")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        if "Coffee_Type" in data.columns:
            st.write("#### Average consumption by coffee type")
            st.bar_chart(data.groupby("Coffee_Type")["Cups_Per_Day"].mean())

    st.write("#### Descriptive statistics")
    st.dataframe(pd.DataFrame(summary, index=["Value"]).T, use_container_width=True)

with inference:
    st.subheader("Confidence Interval")
    st.write(f"{confidence:.0%} bootstrap confidence interval for mean consumption: **{ci_lower:.2f} to {ci_upper:.2f} cups/day**.")
    st.caption("This interval describes uncertainty in the mean under the assumption that the observations reasonably represent the target population.")

    st.subheader("One-sample t-test")
    st.write(f"H₀: Population mean = {hypothesized_mean:.2f} cups/day.")
    st.write(f"H₁: Population mean ≠ {hypothesized_mean:.2f} cups/day.")
    test = one_sample_t_test(data, hypothesized_mean)
    a, b = st.columns(2)
    a.metric("t-statistic", f'{test["t_statistic"]:.4f}')
    b.metric("p-value", f'{test["p_value"]:.5f}')
    if test["p_value"] < 0.05:
        st.success("Reject H₀ at the 5% significance level: the data provide evidence that the population mean differs from the hypothesized mean.")
    else:
        st.info("Fail to reject H₀ at the 5% significance level: the data do not provide sufficient evidence that the population mean differs from the hypothesized mean.")

    st.subheader("Central Limit Theorem simulation")
    max_size = min(100, len(data))
    sample_size = st.slider("Sample size", min_value=1, max_value=max_size, value=min(30, max_size))
    means = clt_simulation(data, sample_size=sample_size)
    fig, ax = plt.subplots()
    ax.hist(means, bins=30, edgecolor="black")
    ax.axvline(data["Cups_Per_Day"].mean(), linestyle="--", label="Dataset mean")
    ax.set_xlabel("Mean cups/day in repeated samples")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with relationships:
    st.subheader("Pearson correlation")
    if {"Cups_Per_Day", "Money_Per_Week"}.issubset(data.columns):
        try:
            corr = pearson_correlation(data)
            a, b = st.columns(2)
            a.metric("Pearson r", f'{corr["r"]:.3f}')
            b.metric("p-value", f'{corr["p_value"]:.5f}')
            fig, ax = plt.subplots()
            ax.scatter(data["Cups_Per_Day"], data["Money_Per_Week"], alpha=0.65)
            ax.set_xlabel("Cups per day")
            ax.set_ylabel("Weekly spending")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Correlation measures linear association, not causation.")
        except ValueError as error:
            st.warning(str(error))

    st.subheader("One-way ANOVA by coffee type")
    if "Coffee_Type" in data.columns:
        try:
            result = group_comparison(data)
            a, b = st.columns(2)
            a.metric("F-statistic", f'{result["f_statistic"]:.4f}')
            b.metric("p-value", f'{result["p_value"]:.5f}')
            if result["p_value"] < 0.05:
                st.write("At the 5% level, there is evidence that at least one coffee-type group mean differs, subject to ANOVA assumptions.")
            else:
                st.write("At the 5% level, there is insufficient evidence that the group means differ.")
        except ValueError as error:
            st.warning(str(error))

with dataset:
    st.subheader("Dataset Explorer")
    st.write(f"Valid rows: {len(data)}")
    st.dataframe(data, use_container_width=True)
    st.download_button("Download cleaned CSV", data.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_coffee_data.csv", mime="text/csv")

st.divider()
st.caption("Educational demonstration. The bundled dataset is synthetic and should not be presented as real survey data.")
