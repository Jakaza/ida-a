# -*- coding: utf-8 -*-
"""Census 2022 Household Analysis - Streamlit app"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Census 2022 Household Analysis", layout="wide")
st.title("Census 2022 Household Analysis")
st.caption("Exploratory data analysis of the Stats SA Census 2022 household sample (F19).")


# ---------------------------------------------------------------- helpers
def show_fig(fig):
    """Render a matplotlib figure in Streamlit and free memory."""
    st.pyplot(fig)
    plt.close(fig)


def pick(df, *names):
    """Return the first column name that exists in df, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None


def countplot(df, x, title, xlabel, hue=None, legend_title=None, size=(8, 5)):
    fig, ax = plt.subplots(figsize=size)
    sns.countplot(x=x, hue=hue, data=df, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if hue and legend_title:
        ax.legend(title=legend_title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    show_fig(fig)


# ---------------------------------------------------------------- data
@st.cache_data
def load_and_clean(path):
    raw = pd.read_csv(path)
    initial_shape = raw.shape
    df = raw.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df = df.drop_duplicates()

    if "derh_hhage" in df.columns:
        df["derh_hhage"] = df["derh_hhage"].astype(int)
        df = df[(df["derh_hhage"] >= 0) & (df["derh_hhage"] <= 100)]

    if "derh_hhsex" in df.columns:
        df["derh_hhsex"] = df["derh_hhsex"].replace({1: "Male", 2: "Female"})

    # the household size column may be called derh_hhsize or derh_hsize
    size_col = "derh_hhsize" if "derh_hhsize" in df.columns else (
        "derh_hsize" if "derh_hsize" in df.columns else None
    )
    if size_col:
        df = df[df[size_col] <= 20]

    return raw.head(), initial_shape, df


try:
    raw_head, initial_shape, df = load_and_clean("Census2022sample_F19.csv")
except FileNotFoundError:
    st.error(
        "Census2022sample_F19.csv was not found. Commit it to the same folder "
        "as this script in your GitHub repo and reboot the app."
    )
    st.stop()

# column names (handles the hhsize / hsize naming difference)
AGE = pick(df, "derh_hhage")
SEX = pick(df, "derh_hhsex")
SIZE = pick(df, "derh_hhsize", "derh_hsize")
WATER = pick(df, "h05_waterpiped")
ADULT_H = pick(df, "a4_adult_hunger")
CHILD_H = pick(df, "a5_child_hunger")
DWELL = pick(df, "h02_maindwelling")
TENURE = pick(df, "h03_tenure")
EDU = pick(df, "educationlevel")
ASSETS = [c for c in ["h12_refrigerator", "h12_computer", "h12_motor_car", "h12_cellphone"] if c in df.columns]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Distributions",
        "Cross-tabulations",
        "Summary statistics",
        "Asset ownership",
    ],
)

with st.sidebar.expander("Show available columns"):
    st.write(df.columns.tolist())


# ---------------------------------------------------------------- sections
if section == "Overview":
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Initial rows", f"{initial_shape[0]:,}")
    c2.metric("Rows after cleaning", f"{df.shape[0]:,}")
    c3.metric("Columns", df.shape[1])

    st.markdown("**First 5 rows (raw)**")
    st.dataframe(raw_head)

    st.markdown("**Descriptive statistics (cleaned)**")
    st.dataframe(df.describe(include="all").T)

    if SEX:
        st.markdown("**Household head sex**")
        st.dataframe(df[SEX].value_counts().rename("count"))
    if WATER:
        st.markdown("**Piped water access (code)**")
        st.dataframe(df[WATER].value_counts().rename("count"))

elif section == "Distributions":
    st.subheader("Distributions")

    if AGE:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[AGE], bins=20, kde=True, ax=ax)
        ax.set_title("Age Distribution of Household Heads")
        ax.set_xlabel("Age of Household Head")
        ax.set_ylabel("Count")
        show_fig(fig)

    if SIZE:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[SIZE], bins=10, discrete=True, ax=ax)
        ax.set_title("Household Size Distribution")
        ax.set_xlabel("Household Size")
        ax.set_ylabel("Count")
        show_fig(fig)

    col1, col2 = st.columns(2)
    with col1:
        if ADULT_H:
            countplot(df, ADULT_H, "Adult Hunger Frequency", "Adult Hunger Code", size=(6, 4))
        if SEX:
            countplot(df, SEX, "Household Head Sex Distribution", "Sex", size=(6, 4))
    with col2:
        if CHILD_H:
            countplot(df, CHILD_H, "Child Hunger Frequency", "Child Hunger Code", size=(6, 4))
        if WATER:
            countplot(df, WATER, "Access to Piped Water", "Water Piped Code", size=(6, 4))

elif section == "Cross-tabulations":
    st.subheader("Cross-tabulations")

    if SEX and WATER:
        st.markdown("### Household head sex vs access to piped water")
        st.dataframe(pd.crosstab(df[SEX], df[WATER]))
        countplot(df, SEX, "Access to Piped Water by Household Head Sex",
                  "Household Head Sex", hue=WATER, legend_title="Water Piped Code")

    if SIZE and ADULT_H:
        st.markdown("### Household size vs adult hunger")
        st.dataframe(pd.crosstab(df[SIZE], df[ADULT_H]))
        countplot(df, SIZE, "Adult Hunger by Household Size",
                  "Household Size", hue=ADULT_H, legend_title="Adult Hunger Code", size=(10, 6))

    if SEX and ADULT_H:
        st.markdown("### Household head sex vs adult hunger")
        st.dataframe(pd.crosstab(df[SEX], df[ADULT_H]))
        countplot(df, SEX, "Adult Hunger by Household Head Sex",
                  "Household Head Sex", hue=ADULT_H, legend_title="Adult Hunger Code")

    if DWELL and TENURE:
        st.markdown("### Dwelling type vs tenure")
        st.dataframe(pd.crosstab(df[DWELL], df[TENURE]))
        countplot(df, DWELL, "Dwelling Type vs Tenure",
                  "Dwelling Type Code", hue=TENURE, legend_title="Tenure Code", size=(12, 6))

    if WATER and ADULT_H:
        st.markdown("### Piped water access vs adult hunger")
        st.dataframe(pd.crosstab(df[WATER], df[ADULT_H]))
        countplot(df, WATER, "Adult Hunger by Water Access",
                  "Water Piped Code", hue=ADULT_H, legend_title="Adult Hunger Code")

    if EDU and TENURE:
        st.markdown("### Education level vs tenure")
        st.dataframe(pd.crosstab(df[EDU], df[TENURE]))

elif section == "Summary statistics":
    st.subheader("Summary statistics")

    if SEX and SIZE:
        st.markdown("**Mean household size by sex of household head**")
        st.dataframe(df.groupby(SEX)[SIZE].mean().rename("mean household size"))

    if TENURE and AGE:
        st.markdown("**Average age of household head by tenure type**")
        st.dataframe(df.groupby(TENURE)[AGE].mean().rename("mean age"))

elif section == "Asset ownership":
    st.subheader("Asset ownership")

    if not ASSETS:
        st.info("No asset columns (h12_*) found in the dataset.")
    for col in ASSETS:
        st.markdown(f"**Ownership distribution for `{col}`**")
        left, right = st.columns([1, 2])
        with left:
            st.dataframe(df[col].value_counts().rename("count"))
        with right:
            countplot(df, col, f"Ownership of {col}", f"{col} Code", size=(6, 4))
