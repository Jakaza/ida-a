
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Census 2022 Data Analysis",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

st.title("📊 Census 2022 Data Analysis")
st.write(
    "Explore household demographics, hunger, water access, "
    "dwelling types, tenure and asset ownership."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_FILE = Path(__file__).parent / "Census2022sample_F19.csv"

if not DATA_FILE.exists():
    st.error(
        "CSV file not found. Please upload "
        "'Census2022sample_F19.csv' to your GitHub repository "
        "in the same folder as notebookc.py."
    )
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

st.subheader("1. Original Dataset")

col1, col2 = st.columns(2)
col1.metric("Original rows", f"{df.shape[0]:,}")
col2.metric("Original columns", f"{df.shape[1]:,}")

with st.expander("Preview original data"):
    st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# CLEAN COLUMN NAMES
# --------------------------------------------------
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Support either household-size spelling if present.
if "derh_hsize" not in df.columns and "derh_hhsize" in df.columns:
    df.rename(columns={"derh_hhsize": "derh_hsize"}, inplace=True)

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
st.subheader("2. Data Cleaning")

# Convert known numeric fields when present.
expected_numeric = [
    "derh_hhage",
    "derh_hhsize",
    "derh_hsize",
    "derh_hhsex",
    "h05_waterpiped",
    "a4_adult_hunger",
    "a5_child_hunger",
    "h02_maindwelling",
    "h03_tenure",
    "h12_refrigerator",
    "h12_computer",
    "h12_motor_car",
    "h12_cellphone",
]

for col in expected_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Identify numeric and categorical columns.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(
    exclude=[np.number]
).columns.tolist()

# Fill missing numeric values with the median.
for col in numeric_cols:
    if df[col].notna().any():
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values.
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# Remove duplicate rows.
rows_before = len(df)
df = df.drop_duplicates()
duplicates_removed = rows_before - len(df)

# Validate household-head age.
if "derh_hhage" in df.columns:
    df = df[
        df["derh_hhage"].between(0, 100)
    ]

# Map household-head sex codes.
if "derh_hhsex" in df.columns:
    df["derh_hhsex"] = df["derh_hhsex"].replace({
        1: "Male",
        2: "Female"
    })

# Remove unrealistic household sizes.
if "derh_hsize" in df.columns:
    df = df[df["derh_hsize"].le(20)]

elif "derh_hhsize" in df.columns:
    df = df[df["derh_hhsize"].le(20)]

col1, col2, col3 = st.columns(3)
col1.metric("Cleaned rows", f"{len(df):,}")
col2.metric("Duplicates removed", f"{duplicates_removed:,}")
col3.metric("Columns available", f"{df.shape[1]:,}")

with st.expander("View cleaned dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("Summary statistics"):
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def show_countplot(data, column, title, xlabel=None,
                   hue=None, legend_title=None):
    """Display a count plot if the required columns exist."""

    required = [column]
    if hue is not None:
        required.append(hue)

    missing = [c for c in required if c not in data.columns]

    if missing:
        st.warning(
            f"Cannot display '{title}'. Missing column(s): "
            + ", ".join(missing)
        )
        return

    plot_data = data[required].dropna()

    if plot_data.empty:
        st.info(f"No data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.countplot(
        data=plot_data,
        x=column,
        hue=hue,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    if hue is not None:
        ax.legend(title=legend_title or hue)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_histogram(data, column, title, xlabel,
                   bins=20, discrete=False):
    """Display a histogram if the column exists."""

    if column not in data.columns:
        st.warning(f"Cannot display '{title}': '{column}' is missing.")
        return

    values = pd.to_numeric(data[column], errors="coerce").dropna()

    if values.empty:
        st.info(f"No numeric data available for {title}.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        values,
        bins=bins,
        discrete=discrete,
        kde=(not discrete and values.nunique() > 5),
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_crosstab(data, row, col, title):
    """Display a cross-tabulation and its chart."""

    if row not in data.columns or col not in data.columns:
        st.warning(
            f"Cannot display '{title}'. "
            f"Required columns: {row}, {col}."
        )
        return

    table = pd.crosstab(data[row], data[col])

    st.markdown(f"**{title}**")
    st.dataframe(table, use_container_width=True)

    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    table.plot(kind="bar", ax=ax)

    ax.set_xlabel(row)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title=col)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 3. HOUSEHOLD DEMOGRAPHICS
# --------------------------------------------------
st.header("3. Household Demographics")

col1, col2 = st.columns(2)

with col1:
    show_histogram(
        df,
        "derh_hhage",
        "Age Distribution of Household Heads",
        "Age of Household Head",
        bins=20
    )

with col2:
    size_col = (
        "derh_hsize" if "derh_hsize" in df.columns
        else "derh_hhsize"
    )

    show_histogram(
        df,
        size_col,
        "Household Size Distribution",
        "Household Size",
        bins=10,
        discrete=True
    )

show_countplot(
    df,
    "derh_hhsex",
    "Household Head Sex Distribution",
    "Sex"
)

if "derh_hhsex" in df.columns:
    st.markdown("**Household Head Sex Counts**")
    st.dataframe(
        df["derh_hhsex"].value_counts().rename_axis("Sex").reset_index(
            name="Count"
        ),
        use_container_width=True
    )

# --------------------------------------------------
# 4. HUNGER ANALYSIS
# --------------------------------------------------
st.header("4. Hunger Analysis")

col1, col2 = st.columns(2)

with col1:
    show_countplot(
        df,
        "a4_adult_hunger",
        "Adult Hunger Frequency",
        "Adult Hunger Code"
    )

with col2:
    show_countplot(
        df,
        "a5_child_hunger",
        "Child Hunger Frequency",
        "Child Hunger Code"
    )

# --------------------------------------------------
# 5. WATER ACCESS
# --------------------------------------------------
st.header("5. Access to Piped Water")

show_countplot(
    df,
    "h05_waterpiped",
    "Access to Piped Water",
    "Water Piped Code"
)

# Household head sex versus water access.
show_crosstab(
    df,
    "derh_hhsex",
    "h05_waterpiped",
    "Household Head Sex vs Access to Piped Water"
)

show_countplot(
    df,
    "derh_hhsex",
    "Access to Piped Water by Household Head Sex",
    "Household Head Sex",
    hue="h05_waterpiped",
    legend_title="Water Piped Code"
)

# --------------------------------------------------
# 6. HOUSEHOLD SIZE AND HUNGER
# --------------------------------------------------
st.header("6. Household Size and Hunger")

show_crosstab(
    df,
    size_col,
    "a4_adult_hunger",
    "Household Size vs Adult Hunger"
)

show_countplot(
    df,
    size_col,
    "Adult Hunger by Household Size",
    "Household Size",
    hue="a4_adult_hunger",
    legend_title="Adult Hunger Code"
)

# --------------------------------------------------
# 7. SEX AND ADULT HUNGER
# --------------------------------------------------
st.header("7. Household Head Sex and Adult Hunger")

show_crosstab(
    df,
    "derh_hhsex",
    "a4_adult_hunger",
    "Household Head Sex vs Adult Hunger"
)

show_countplot(
    df,
    "derh_hhsex",
    "Adult Hunger by Household Head Sex",
    "Household Head Sex",
    hue="a4_adult_hunger",
    legend_title="Adult Hunger Code"
)

# --------------------------------------------------
# 8. DWELLING TYPE AND TENURE
# --------------------------------------------------
st.header("8. Dwelling Type and Tenure")

show_crosstab(
    df,
    "h02_maindwelling",
    "h03_tenure",
    "Dwelling Type vs Tenure"
)

show_countplot(
    df,
    "h02_maindwelling",
    "Dwelling Type vs Tenure",
    "Dwelling Type Code",
    hue="h03_tenure",
    legend_title="Tenure Code"
)

# --------------------------------------------------
# 9. SUMMARY STATISTICS
# --------------------------------------------------
st.header("9. Summary Statistics")

if "derh_hhsex" in df.columns and size_col in df.columns:
    st.markdown("**Mean Household Size by Sex of Household Head**")
    st.dataframe(
        df.groupby("derh_hhsex")[size_col]
        .mean()
        .reset_index(name="Average Household Size"),
        use_container_width=True
    )

if "h03_tenure" in df.columns and "derh_hhage" in df.columns:
    st.markdown("**Average Age of Household Head by Tenure Type**")
    st.dataframe(
        df.groupby("h03_tenure")["derh_hhage"]
        .mean()
        .reset_index(name="Average Age"),
        use_container_width=True
    )

# --------------------------------------------------
# 10. ASSET OWNERSHIP
# --------------------------------------------------
st.header("10. Household Asset Ownership")

asset_cols = [
    "h12_refrigerator",
    "h12_computer",
    "h12_motor_car",
    "h12_cellphone"
]

for col in asset_cols:
    if col not in df.columns:
        st.warning(f"Asset column '{col}' is not in the dataset.")
        continue

    st.subheader(f"Ownership of {col}")

    counts = (
        df[col]
        .value_counts(dropna=False)
        .rename_axis("Code")
        .reset_index(name="Count")
    )

    st.dataframe(counts, use_container_width=True)

    show_countplot(
        df,
        col,
        f"Ownership Distribution: {col}",
        f"{col} Code"
    )

# --------------------------------------------------
# 11. WATER ACCESS AND ADULT HUNGER
# --------------------------------------------------
st.header("11. Water Access and Adult Hunger")

show_crosstab(
    df,
    "h05_waterpiped",
    "a4_adult_hunger",
    "Adult Hunger by Water Access"
)

show_countplot(
    df,
    "h05_waterpiped",
    "Adult Hunger by Water Access",
    "Water Piped Code",
    hue="a4_adult_hunger",
    legend_title="Adult Hunger Code"
)

# --------------------------------------------------
# 12. EDUCATION AND TENURE, IF AVAILABLE
# --------------------------------------------------
st.header("12. Education and Tenure")

if "educationlevel" in df.columns and "h03_tenure" in df.columns:
    show_crosstab(
        df,
        "educationlevel",
        "h03_tenure",
        "Education Level vs Tenure"
    )
else:
    st.info(
        "Education and tenure analysis was skipped because "
        "the dataset does not contain both 'educationlevel' "
        "and 'h03_tenure'."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Census 2022 Sample — Exploratory Data Analysis")
