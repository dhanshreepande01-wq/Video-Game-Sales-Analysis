
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Video Game Sales Analysis",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Video Game Sales Analysis")
st.write(
    "Explore video game sales across platforms, "
    "genres, publishers and regions."
)

# ==================================================
# 2. FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload your vgsales.csv file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your video game sales CSV file.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

# ==================================================
# 3. CLEAN COLUMN NAMES
# ==================================================

# Remove extra spaces from column names
df.columns = df.columns.astype(str).str.strip()

# Match column names even if they have different
# spaces, capital letters or special characters.

def normalize_column(column):
    return re.sub(r"[^a-z0-9]", "", column.lower())

aliases = {
    "Rank": ["rank", "ranking"],
    "Name": ["name", "game", "game_name", "title"],
    "Platform": ["platform", "console"],
    "Year": ["year", "year_of_release", "release_year"],
    "Genre": ["genre", "game_genre"],
    "Publisher": ["publisher", "game_publisher"],
    "NA_Sales": [
        "na_sales", "north_america_sales",
        "north_american_sales"
    ],
    "EU_Sales": [
        "eu_sales", "europe_sales",
        "european_sales"
    ],
    "JP_Sales": [
        "jp_sales", "japan_sales",
        "japanese_sales"
    ],
    "Other_Sales": [
        "other_sales", "rest_of_world_sales"
    ],
    "Global_Sales": [
        "global_sales", "worldwide_sales",
        "total_sales"
    ]
}

existing_columns = {
    normalize_column(col): col for col in df.columns
}

rename_columns = {}

for standard_name, possible_names in aliases.items():
    for possible_name in possible_names:
        key = normalize_column(possible_name)

        if key in existing_columns:
            actual_name = existing_columns[key]
            rename_columns[actual_name] = standard_name
            break

df = df.rename(columns=rename_columns)

# ==================================================
# 4. CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "Name",
    "Platform",
    "Year",
    "Genre",
    "Publisher",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error("Some required columns are missing.")

    st.write("**Missing columns:**")
    st.write(missing_columns)

    st.write("**Columns found in your CSV:**")
    st.write(df.columns.tolist())

    st.info(
        "Please check that you uploaded the correct "
        "video game sales dataset."
    )
    st.stop()

# ==================================================
# 5. DATA PREPARATION
# ==================================================

data = df.copy()

# Convert numerical columns
numeric_columns = [
    "Year",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]

if "Rank" in data.columns:
    numeric_columns.append("Rank")

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Replace missing categorical values
data["Name"] = data["Name"].fillna("Unknown")
data["Platform"] = data["Platform"].fillna("Unknown")
data["Genre"] = data["Genre"].fillna("Unknown")
data["Publisher"] = data["Publisher"].fillna("Unknown")

# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def show_table(table):
    if table is None or table.empty:
        st.info("No data available.")
    else:
        st.table(table)


def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def make_bar(series, title, xlabel, ylabel):
    series = series.dropna()

    if series.empty:
        st.info("No data available for this chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=series.index.astype(str),
        y=series.values,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    show_plot(fig)


def get_genre_region():
    columns = [
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales"
    ]

    return data.groupby("Genre")[columns].sum()


# ==================================================
# 7. SIDEBAR NAVIGATION
# ==================================================

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

# ==================================================
# 8. DATA PREPARATION
# ==================================================

if page == "Data Preparation":

    st.header("1. Data Preparation")

    st.subheader("Dataset Preview")
    show_table(data.head(10))

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{len(data):,}")
    col2.metric("Total Columns", data.shape[1])
    col3.metric("Duplicate Rows", f"{data.duplicated().sum():,}")

    st.subheader("Missing Values")

    missing = data.isnull().sum().rename("Missing Values")
    show_table(missing.to_frame())

    st.subheader("Data Types")

    types = data.dtypes.astype(str).rename("Data Type")
    show_table(types.to_frame())

    st.subheader("Column Names")
    st.write(data.columns.tolist())


# ==================================================
# 9. EXPLORATORY DATA ANALYSIS
# ==================================================

elif page == "EDA":

    st.header("2. Exploratory Data Analysis")

    st.subheader("Descriptive Statistics")
    show_table(data.describe().round(2))

    st.subheader("Sales by Genre and Region")

    genre_stats = data.groupby("Genre")[
        ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    ].agg(["sum", "mean"]).round(2)

    show_table(genre_stats)

    st.subheader("Platform Sales")

    platform_stats = data.groupby("Platform")[
        "Global_Sales"
    ].agg(["sum", "mean"]).round(2)

    show_table(platform_stats)

    st.subheader("Genre Global Sales")

    genre_global = data.groupby("Genre")[
        "Global_Sales"
    ].agg(["sum", "mean"]).round(2)

    show_table(genre_global)

    st.subheader("Global Sales Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=data,
        x="Global_Sales",
        bins=30,
        ax=ax
    )
    ax.set_title("Distribution of Global Sales")
    ax.set_xlabel("Global Sales (Millions)")
    ax.set_ylabel("Frequency")
    show_plot(fig)

    st.subheader("Global Sales Box Plot")

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.boxplot(data=data, x="Global_Sales", ax=ax)
    ax.set_title("Global Sales Outliers")
    show_plot(fig)

    regions = [
        ("NA_Sales", "North America Sales Distribution"),
        ("EU_Sales", "Europe Sales Distribution"),
        ("JP_Sales", "Japan Sales Distribution"),
        ("Other_Sales", "Other Regions Sales Distribution")
    ]

    for column, title in regions:

        fig, ax = plt.subplots(figsize=(9, 4))

        sns.histplot(
            data=data,
            x=column,
            bins=30,
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("Sales (Millions)")
        show_plot(fig)

    st.subheader("Games by Genre")

    genre_count = data["Genre"].value_counts()

    make_bar(
        genre_count,
        "Games by Genre",
        "Genre",
        "Number of Games"
    )


# ==================================================
# 10. TOP SELLING GAMES
# ==================================================

elif page == "Top Selling Games":

    st.header("3. Top Selling Games")

    st.subheader("Top 10 Games by Global Sales")

    top_games = data.sort_values(
        "Global_Sales",
        ascending=False
    )[
        ["Name", "Platform", "Year", "Global_Sales"]
    ].head(10)

    show_table(top_games)

    total_sales = data["Global_Sales"].sum()

    st.metric(
        "Total Global Sales",
        f"{total_sales:,.2f} Million Units"
    )

    st.caption(
        "Sales figures are expressed in millions of units."
    )


# ==================================================
# 11. PLATFORM PERFORMANCE
# ==================================================

elif page == "Platform Performance":

    st.header("4. Platform Performance")

    st.subheader("Top 10 Platforms by Global Sales")

    platform_sales = (
        data.groupby("Platform")["Global_Sales"]
        .sum()
        .nlargest(10)
    )

    show_table(
        platform_sales.rename(
            "Total Global Sales (Millions)"
        ).to_frame()
    )

    make_bar(
        platform_sales,
        "Top 10 Platforms by Global Sales",
        "Platform",
        "Global Sales (Millions)"
    )

    st.subheader("Platforms with Most Top-Selling Games Over Time")

    valid_years = data[
        (data["Year"].notna()) &
        (data["Year"] > 0) &
        (data["Global_Sales"].notna())
    ]

    if not valid_years.empty:

        yearly_top = valid_years.loc[
            valid_years.groupby("Year")["Global_Sales"].idxmax()
        ]

        platform_year_count = yearly_top["Platform"].value_counts()

        make_bar(
            platform_year_count,
            "Platforms with Most Top-Selling Games Over Time",
            "Platform",
            "Number of Years"
        )

    else:
        st.info("No valid year data available.")


# ==================================================
# 12. GENRE ANALYSIS
# ==================================================

elif page == "Genre Analysis":

    st.header("5. Genre Analysis")

    st.subheader("Global Sales by Genre")

    genre_sales = (
        data.groupby("Genre")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    show_table(genre_sales.to_frame())

    make_bar(
        genre_sales,
        "Global Sales by Genre",
        "Genre",
        "Global Sales (Millions)"
    )

    st.subheader("Genre Sales Across Platforms")

    genre_platform = pd.pivot_table(
        data=data,
        index="Genre",
        columns="Platform",
        values="Global_Sales",
        aggfunc="sum",
        fill_value=0
    )

    show_table(genre_platform)

    fig, ax = plt.subplots(figsize=(14, 6))

    genre_platform.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Genre Sales Across Platforms")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Global Sales (Millions)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(
        title="Platform",
        bbox_to_anchor=(1.02, 1)
    )

    show_plot(fig)

    st.subheader("Genre Sales Across Regions")

    genre_region = get_genre_region()

    show_table(genre_region)

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_region.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Genre Sales Across Regions")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Total Sales (Millions)")
    ax.tick_params(axis="x", rotation=45)

    show_plot(fig)

    st.subheader("Genre Sales Over the Years")

    yearly_genre = data[
        (data["Year"].notna()) &
        (data["Year"] > 0)
    ]

    genre_year = (
        yearly_genre.groupby(["Year", "Genre"])["Global_Sales"]
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

        ax.set_title("Global Sales by Genre Over the Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Global Sales (Millions)")

        show_plot(fig)

    else:
        st.info("No valid year information available.")


# ==================================================
# 13. PUBLISHER PERFORMANCE
# ==================================================

elif page == "Publisher Performance":

    st.header("6. Publisher Performance")

    st.subheader("Top Publisher by Global Sales in Each Genre")

    publisher_genre = (
        data.groupby(["Genre", "Publisher"])["Global_Sales"]
        .sum()
        .reset_index()
    )

    publisher_genre = publisher_genre.dropna(
        subset=["Global_Sales"]
    )

    if not publisher_genre.empty:

        result = publisher_genre.loc[
            publisher_genre.groupby("Genre")[
                "Global_Sales"
            ].idxmax()
        ].sort_values(
            "Global_Sales",
            ascending=False
        )

        show_table(result)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=result,
            x="Genre",
            y="Global_Sales",
            hue="Publisher",
            dodge=False,
            ax=ax
        )

        ax.set_title("Top Publisher by Global Sales in Each Genre")
        ax.set_xlabel("Genre")
        ax.set_ylabel("Global Sales (Millions)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(
            title="Publisher",
            bbox_to_anchor=(1.02, 1)
        )

        show_plot(fig)

    else:
        st.info("No publisher sales information available.")

    st.subheader("Publisher Performance Across Regions")

    publisher_region = data.groupby("Publisher")[
        ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    ].sum()

    top10 = (
        publisher_region.sum(axis=1)
        .nlargest(10)
        .index
    )

    publisher_region = publisher_region.loc[top10]

    show_table(publisher_region)

    fig, ax = plt.subplots(figsize=(12, 6))

    publisher_region.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Top 10 Publisher Performance Across Regions")
    ax.set_xlabel("Publisher")
    ax.set_ylabel("Total Sales (Millions)")
    ax.tick_params(axis="x", rotation=45)

    show_plot(fig)


# ==================================================
# 14. REGIONAL SALES ANALYSIS
# ==================================================

elif page == "Regional Sales Analysis":

    st.header("7. Regional Sales Analysis")

    regions = [
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales"
    ]

    st.subheader("Total Sales Across Regions")

    region_sales = data[regions].sum()

    show_table(
        region_sales.rename(
            "Total Sales (Millions)"
        ).to_frame()
    )

    make_bar(
        region_sales,
        "Total Sales Across Regions",
        "Region",
        "Total Sales (Millions)"
    )

    st.subheader("Regional Sales by Genre")

    genre_region = get_genre_region()

    show_table(genre_region)

    fig, ax = plt.subplots(figsize=(12, 6))

    genre_region.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title("Regional Sales by Genre")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Total Sales (Millions)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Region")

    show_plot(fig)


# ==================================================
# 15. FOOTER
# ==================================================

st.divider()

st.success("Video Game Sales Analysis Completed!")

st.caption(
    "Developed using Python, Pandas, Seaborn, "
    "Matplotlib and Streamlit."
)
