
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Video Game Sales Analysis",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Video Game Sales Analysis")
st.write(
    "Explore video game sales across platforms, genres, "
    "publishers and regions."
)

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your vgsales.csv file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your vgsales.csv file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

# --------------------------------------------------
# STANDARDIZE COLUMN NAMES
# --------------------------------------------------

df.columns = df.columns.str.strip()

# Match column names without depending on capitalization
column_map = {
    col.lower(): col for col in df.columns
}

required_columns = [
    "Rank", "Name", "Platform", "Year", "Genre",
    "Publisher", "NA_Sales", "EU_Sales",
    "JP_Sales", "Other_Sales", "Global_Sales"
]

missing_columns = [
    col for col in required_columns
    if col.lower() not in column_map
]

if missing_columns:
    st.error(
        "The uploaded dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )
    st.write("Columns found in your file:", df.columns.tolist())
    st.stop()

# Rename columns to the standard dataset names
df = df.rename(
    columns={
        column_map[col.lower()]: col
        for col in required_columns
    }
)

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

numeric_columns = [
    "Rank", "Year", "NA_Sales", "EU_Sales",
    "JP_Sales", "Other_Sales", "Global_Sales"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep missing years as missing.
# Fill missing text values so they can be grouped.
df["Publisher"] = df["Publisher"].fillna("Unknown")
df["Genre"] = df["Genre"].fillna("Unknown")
df["Platform"] = df["Platform"].fillna("Unknown")
df["Name"] = df["Name"].fillna("Unknown")

# Sales columns: treat missing sales values as zero
sales_columns = [
    "NA_Sales", "EU_Sales", "JP_Sales",
    "Other_Sales", "Global_Sales"
]

df[sales_columns] = df[sales_columns].fillna(0)

# --------------------------------------------------
# DARK CHART STYLE
# --------------------------------------------------

def style_chart(fig, ax, title, xlabel, ylabel):
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.set_title(title, color="white", fontsize=14, pad=15)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#777777")

    ax.grid(axis="y", alpha=0.2, color="white")
    fig.tight_layout()


def show_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def make_bar(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=series.index.astype(str),
        y=series.values,
        ax=ax,
        color="#35A7A0"
    )

    style_chart(fig, ax, title, xlabel, ylabel)
    ax.tick_params(axis="x", rotation=45)

    show_plot(fig)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Data Preparation",
        "EDA",
        "Top Selling Games",
        "Platform Performance",
        "Genre Analysis",
        "Publisher Performance",
        "Regional Sales Analysis"
    ]
)

# --------------------------------------------------
# 1. DATA PREPARATION
# --------------------------------------------------

if page == "Data Preparation":

    st.header("1. Data Preparation")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("Missing Values")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column Name", "Missing Values"]

    st.dataframe(missing, use_container_width=True, hide_index=True)

    st.subheader("Data Types")

    dtypes = df.dtypes.astype(str).reset_index()
    dtypes.columns = ["Column Name", "Data Type"]

    st.dataframe(dtypes, use_container_width=True, hide_index=True)

    st.subheader("Column Names")
    st.write(df.columns.tolist())


# --------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

elif page == "EDA":

    st.header("2. Exploratory Data Analysis")

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.subheader("Sales by Genre and Region")

    genre_region = df.groupby("Genre").agg(
        NA_Total=("NA_Sales", "sum"),
        NA_Average=("NA_Sales", "mean"),
        EU_Total=("EU_Sales", "sum"),
        EU_Average=("EU_Sales", "mean"),
        JP_Total=("JP_Sales", "sum"),
        JP_Average=("JP_Sales", "mean"),
        Other_Total=("Other_Sales", "sum"),
        Other_Average=("Other_Sales", "mean")
    ).round(2).reset_index()

    st.dataframe(
        genre_region,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Platform Sales")

    platform_stats = df.groupby("Platform").agg(
        Total_Sales=("Global_Sales", "sum"),
        Average_Sales=("Global_Sales", "mean")
    ).round(2).reset_index()

    st.dataframe(
        platform_stats,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Genre Global Sales")

    genre_stats = df.groupby("Genre").agg(
        Total_Sales=("Global_Sales", "sum"),
        Average_Sales=("Global_Sales", "mean")
    ).round(2).sort_values(
        "Total_Sales", ascending=False
    ).reset_index()

    st.dataframe(
        genre_stats,
        use_container_width=True,
        hide_index=True
    )

    # Global sales histogram
    st.subheader("Global Sales Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Global_Sales",
        bins=30,
        color="#35A7A0",
        ax=ax
    )

    style_chart(
        fig, ax,
        "Distribution of Global Sales",
        "Global Sales (Millions)",
        "Number of Games"
    )

    show_plot(fig)

    # Global sales boxplot
    st.subheader("Global Sales Box Plot")

    fig, ax = plt.subplots(figsize=(10, 3))

    sns.boxplot(
        data=df,
        x="Global_Sales",
        color="#E69F56",
        ax=ax
    )

    style_chart(
        fig, ax,
        "Global Sales Outliers",
        "Global Sales (Millions)",
        ""
    )

    show_plot(fig)

    # Regional histograms
    st.subheader("Regional Sales Distributions")

    regions = [
        ("NA_Sales", "North America"),
        ("EU_Sales", "Europe"),
        ("JP_Sales", "Japan"),
        ("Other_Sales", "Other Regions")
    ]

    for col, region_name in regions:

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.histplot(
            data=df,
            x=col,
            bins=30,
            color="#6386D8",
            ax=ax
        )

        style_chart(
            fig, ax,
            f"{region_name} Sales Distribution",
            "Sales (Millions)",
            "Number of Games"
        )

        show_plot(fig)

    st.subheader("Games by Genre")

    genre_count = df["Genre"].value_counts()

    make_bar(
        genre_count,
        "Number of Games by Genre",
        "Genre",
        "Number of Games"
    )


# --------------------------------------------------
# 3. TOP SELLING GAMES
# --------------------------------------------------

elif page == "Top Selling Games":

    st.header("3. Top Selling Games")

    st.subheader("Top 10 Games by Global Sales")

    top_games = df.sort_values(
        "Global_Sales",
        ascending=False
    )[["Name", "Platform", "Genre", "Global_Sales"]].head(10)

    st.dataframe(
        top_games,
        use_container_width=True,
        hide_index=True
    )

    total_sales = df["Global_Sales"].sum()

    st.metric(
        "Total Global Sales",
        f"{total_sales:,.2f} Million Units"
    )


# --------------------------------------------------
# 4. PLATFORM PERFORMANCE
# --------------------------------------------------

elif page == "Platform Performance":

    st.header("4. Platform Performance")

    st.subheader("Top 10 Platforms by Global Sales")

    platform_sales = (
        df.groupby("Platform")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    platform_table = platform_sales.reset_index()
    platform_table.columns = ["Platform", "Total Global Sales"]

    st.dataframe(
        platform_table.round(2),
        use_container_width=True,
        hide_index=True
    )

    make_bar(
        platform_sales,
        "Top 10 Platforms by Global Sales",
        "Platform",
        "Global Sales (Millions)"
    )

    st.subheader("Platforms with Most Top-Selling Games Over Time")

    valid_years = df.dropna(subset=["Year"]).copy()

    if not valid_years.empty:

        top_games_year = valid_years.loc[
            valid_years.groupby("Year")["Global_Sales"].idxmax()
        ]

        platform_count = top_games_year["Platform"].value_counts()

        make_bar(
            platform_count,
            "Platforms with the Most Top-Selling Games Over Time",
            "Platform",
            "Number of Years"
        )

    else:
        st.info("No valid year information is available.")


# --------------------------------------------------
# 5. GENRE ANALYSIS
# --------------------------------------------------

elif page == "Genre Analysis":

    st.header("5. Genre Analysis")

    st.subheader("Global Sales by Genre")

    genre_sales = (
        df.groupby("Genre")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    genre_table = genre_sales.reset_index()
    genre_table.columns = ["Genre", "Global Sales"]

    st.dataframe(
        genre_table.round(2),
        use_container_width=True,
        hide_index=True
    )

    make_bar(
        genre_sales,
        "Global Sales by Genre",
        "Genre",
        "Global Sales (Millions)"
    )

    # Genre vs Platform
    st.subheader("Genre Sales Across Platforms")

    genre_platform = pd.pivot_table(
        df,
        index="Genre",
        columns="Platform",
        values="Global_Sales",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(
        genre_platform.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    genre_platform.plot(
        kind="bar",
        ax=ax,
        colormap="tab20"
    )

    style_chart(
        fig, ax,
        "Genre Sales Across Platforms",
        "Genre",
        "Global Sales (Millions)"
    )

    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        title="Platform",
        bbox_to_anchor=(1.02, 1),
        facecolor="#0E1117",
        labelcolor="white"
    )

    fig.tight_layout()
    show_plot(fig)

    # Genre vs Region
    st.subheader("Genre Sales Across Regions")

    genre_region_total = df.groupby("Genre")[
        sales_columns[:4]
    ].sum()

    st.dataframe(
        genre_region_total.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_region_total.plot(
        kind="bar",
        ax=ax,
        colormap="Set2"
    )

    style_chart(
        fig, ax,
        "Genre Sales Across Regions",
        "Genre",
        "Total Sales (Millions)"
    )

    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        title="Region",
        facecolor="#0E1117",
        labelcolor="white"
    )

    fig.tight_layout()
    show_plot(fig)

    # Genre vs Year
    st.subheader("Genre Sales Over the Years")

    year_data = df.dropna(subset=["Year"])

    genre_year = (
        year_data.groupby(["Year", "Genre"])["Global_Sales"]
        .sum()
        .reset_index()
    )

    if not genre_year.empty:

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.lineplot(
            data=genre_year,
            x="Year",
            y="Global_Sales",
            hue="Genre",
            ax=ax
        )

        style_chart(
            fig, ax,
            "Global Sales by Genre Over the Years",
            "Year",
            "Global Sales (Millions)"
        )

        ax.legend(
            title="Genre",
            bbox_to_anchor=(1.02, 1),
            facecolor="#0E1117",
            labelcolor="white"
        )

        fig.tight_layout()
        show_plot(fig)

    else:
        st.info("No valid year information is available.")


# --------------------------------------------------
# 6. PUBLISHER PERFORMANCE
# --------------------------------------------------

elif page == "Publisher Performance":

    st.header("6. Publisher Performance")

    st.subheader("Top Publisher by Global Sales in Each Genre")

    publisher_genre = (
        df.groupby(["Genre", "Publisher"])["Global_Sales"]
        .sum()
        .reset_index()
    )

    result = publisher_genre.loc[
        publisher_genre.groupby("Genre")["Global_Sales"].idxmax()
    ].sort_values(
        "Global_Sales",
        ascending=False
    ).reset_index(drop=True)

    result = result.rename(
        columns={"Global_Sales": "Total Global Sales"}
    )

    st.dataframe(
        result.round(2),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=result,
        x="Genre",
        y="Total Global Sales",
        hue="Publisher",
        dodge=False,
        ax=ax
    )

    style_chart(
        fig, ax,
        "Top Publisher by Global Sales in Each Genre",
        "Genre",
        "Global Sales (Millions)"
    )

    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        title="Publisher",
        bbox_to_anchor=(1.02, 1),
        facecolor="#0E1117",
        labelcolor="white"
    )

    fig.tight_layout()
    show_plot(fig)

    # Publisher vs Region
    st.subheader("Publisher Performance Across Regions")

    publisher_region = df.groupby("Publisher")[
        sales_columns[:4]
    ].sum()

    top10_publishers = (
        publisher_region.sum(axis=1)
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    publisher_region = publisher_region.loc[
        top10_publishers
    ]

    st.dataframe(
        publisher_region.round(2).reset_index(),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    publisher_region.plot(
        kind="bar",
        ax=ax,
        colormap="Set2"
    )

    style_chart(
        fig, ax,
        "Top 10 Publishers Across Regions",
        "Publisher",
        "Total Sales (Millions)"
    )

    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        title="Region",
        facecolor="#0E1117",
        labelcolor="white"
    )

    fig.tight_layout()
    show_plot(fig)


# --------------------------------------------------
# 7. REGIONAL SALES ANALYSIS
# --------------------------------------------------

elif page == "Regional Sales Analysis":

    st.header("7. Regional Sales Analysis")

    region_sales = df[sales_columns[:4]].sum()

    st.subheader("Total Sales Across Regions")

    region_table = region_sales.reset_index()
    region_table.columns = ["Region", "Total Sales (Millions)"]

    region_table["Region"] = region_table["Region"].replace({
        "NA_Sales": "North America",
        "EU_Sales": "Europe",
        "JP_Sales": "Japan",
        "Other_Sales": "Other Regions"
    })

    st.dataframe(
        region_table.round(2),
        use_container_width=True,
        hide_index=True
    )

    make_bar(
        region_sales,
        "Total Sales Across Regions",
        "Region",
        "Total Sales (Millions)"
    )

    # Regional sales by genre
    st.subheader("Regional Sales by Genre")

    genre_region = df.groupby("Genre")[
        sales_columns[:4]
    ].sum()

    st.dataframe(
        genre_region.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_region.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="Set2"
    )

    style_chart(
        fig, ax,
        "Regional Sales by Genre",
        "Genre",
        "Total Sales (Millions)"
    )

    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        title="Region",
        facecolor="#0E1117",
        labelcolor="white"
    )

    fig.tight_layout()
    show_plot(fig)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Video Game Sales Analysis | "
    "Python, Pandas, NumPy, Seaborn, Matplotlib & Streamlit"
)
